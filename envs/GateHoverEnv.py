"""Fly from a random start in the room to a hover point in front of the gate and hold it.

Phase 4 (state-based teacher, n_frames = 0): the actor sees the true state relative to the hover
point plus its own previous commands; the critic additionally sees the randomised physical
parameters (asymmetric actor-critic).

Phase 5 (camera policy, n_frames > 0): the actor sees the last n_frames grayscale camera images
("image", uint8, newest first) plus its previous commands ("actions"); the critic sees exactly the
teacher's inputs ("state", "privileged_state"). "corners" holds the training labels for the
auxiliary corner task (where the opening's corners are in the newest frame); only the PPO loss
reads it.

Gate geometry, layout and camera projection come from tools/gate_geometry.py (repo root must be
on sys.path, as in exps/examples/run.py). The gate pose comes from a scene's .gate.json sidecar.
"""
import json
import math
from typing import Dict, Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces

from tools import gate_geometry as gg
from .base.droneGymEnv import DroneGymEnvsBase
from ..utils.type import TensorDict

DEFAULT_START = {
    "margin_wall": 0.4,          # m from walls and ceiling
    "z": [0.5, 2.5],             # m
    "min_gate_distance": 0.6,    # m in front of the gate plane
    "yaw_noise_deg": 20.0,       # heading noise around "pointing at the opening centre"
    "tilt_deg": 5.0,             # roll and pitch
    "speed": 0.3,                # m/s per axis
    "rate": 0.3,                 # rad/s per axis
    "require_opening_in_view": True,
}
DEFAULT_SCENE = "VisFly/datasets/visfly-beta/configs/scenes/gate_room"
DEFAULT_AUGMENT = {
    # camera-like image changes. "Per episode" values are drawn per drone at each reset (exposure,
    # camera response, focus); the others change every frame
    "gain": [0.5, 1.3],          # per episode: brightness factor (the gate room renders bright already)
    "gamma": [0.7, 1.4],         # per episode: response curve
    "offset": [-20.0, 20.0],     # per episode: black level, gray values
    "blur_sigma": [0.0, 1.0],    # per episode: Gaussian blur, px at image_size
    "noise_std": [1.0, 6.0],     # per episode: level of per-pixel noise, gray values
    "frame_gain": 0.05,          # per frame: relative brightness jitter (auto exposure)
}
DEFAULT_REWARD = {
    # every term that is paid each step is positive or small, so that staying alive is always worth
    # more than crashing (a per-step distance penalty made early crashes attractive in the first run)
    "progress": 2.0,             # per m of distance reduction
    "closeness": 0.1,            # bonus exp(-d_weighted / closeness_scale) every step
    "closeness_scale": 1.0,      # m
    "axis_weight": [1.0, 1.0, 1.0],  # distance along the gate normal, sideways, height
    "hold": 0.1,                 # bonus exp(-(d / hold_sigma)^2) every step
    "hold_sigma": 0.1,           # m
    "velocity": 0.02,            # per m/s, weighted by closeness to the hover point
    "rate": 0.005,               # per rad/s
    "view": 0.05,                # (1 - cos of angle between camera axis and opening centre)
    "smooth": 0.02,              # squared change of the normalised command
    "crash": 10.0,               # once, on failure
}


class GateStartGenerator:
    """Start states in front of the gate with the camera pointing roughly at the opening.
    Implements the interface DroneEnvsBase expects from a state generator."""

    def __init__(self, gate_pos, gate_yaw, room, start, seed=0):
        self.gate_pos, self.gate_yaw, self.room, self.s = gate_pos, gate_yaw, room, start
        self.rng = np.random.default_rng(seed)
        kp = gg.gate_keypoints()
        self.opening_w = np.array(gg.gate_to_world(kp["opening_front"], gate_pos, gate_yaw))
        self.centre_w = np.array(gg.gate_to_world(kp["opening_center"], gate_pos, gate_yaw))[0]

    def to(self, device):
        return self

    def _sample(self):
        s, r, rng = self.s, self.room, self.rng
        m = s["margin_wall"]
        c, sn = math.cos(self.gate_yaw), math.sin(self.gate_yaw)
        for _ in range(1000):
            p = np.array([rng.uniform(r["x"][0] + m, r["x"][1] - m), rng.uniform(r["y"][0] + m, r["y"][1] - m),
                          rng.uniform(*s["z"])])
            rel = p[:2] - np.array(self.gate_pos[:2])
            if c * rel[0] + sn * rel[1] > -s["min_gate_distance"]:  # gate-frame x: must be in front
                continue
            to_c = self.centre_w - p
            yaw = math.atan2(to_c[1], to_c[0]) + math.radians(rng.uniform(-1, 1) * s["yaw_noise_deg"])
            tilt = math.radians(s["tilt_deg"])
            q = gg.yaw_quat(yaw, rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt))
            if s["require_opening_in_view"]:
                uv, depth = gg.project_points(self.opening_w, p, q, (320, 320))
                if (depth <= 0).any() or (uv < 8).any() or (uv > 312).any():
                    continue
            vel = rng.uniform(-1, 1, 3) * s["speed"]
            rate = rng.uniform(-1, 1, 3) * s["rate"]
            return p, np.array(q), vel, rate
        raise RuntimeError("no valid start state found; check the start settings")

    def safe_generate(self, num=1, **kwargs):
        p, q, v, w = zip(*[self._sample() for _ in range(num)])
        f = lambda x: th.as_tensor(np.array(x), dtype=th.float32)
        return f(p), f(q), f(v), f(w)


class GateHoverEnv(DroneGymEnvsBase):
    """Actions: [thrust, roll rate, pitch rate, yaw rate] in [-1, 1]. Rates map linearly to
    +- max_rate. Thrust is centred on hover: 0 = hover thrust (nominal mass), +1 = maximum thrust,
    -1 = zero thrust, piecewise linear in between (see policy_to_dynamics). The deployment script
    must use the same mapping."""

    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = False,
            requires_grad: bool = False,
            random_kwargs: dict = None,
            dynamics_kwargs: dict = {},
            scene_kwargs: dict = {},
            sensor_kwargs: list = [],
            device: str = "cpu",
            max_episode_steps: int = 300,
            tensor_output: bool = False,
            gate_pose_file: str = "VisFly/datasets/visfly-beta/configs/scenes/gate_room/gate_room_0.gate.json",
            hover_distance: float = 1.0,
            n_prev_actions: int = 4,
            start_kwargs: Optional[Dict] = None,
            reward_kwargs: Optional[Dict] = None,
            drone_radius: float = 0.07,
            max_tilt_deg: float = 60.0,
            n_frames: int = 0,
            image_size: int = 128,
            render_size: int = 128,
            augment=None,
            image_device: Optional[str] = None,
    ):
        """n_frames > 0 switches to camera observations (phase 5). render_size is the rendered
        resolution; larger values are downsampled to image_size (pixel averaging, like the
        real 320 x 320 -> image_size step). augment: dict overriding DEFAULT_AUGMENT, or False.
        image_device: where grayscale conversion, downsampling and augmentation run (default: the
        GPU if available; on the CPU they took 70 of 128 ms per step with 64 drones)."""
        with open(gate_pose_file) as f:
            pose = json.load(f)
        self.gate_pos, self.gate_yaw = pose["gate_pos"], pose["gate_yaw"]
        self.room = gg.load_json(gg.ROOM_SPEC)
        self.start_cfg = {**DEFAULT_START, **(start_kwargs or {})}
        self.reward_cfg = {**DEFAULT_REWARD, **(reward_kwargs or {})}
        self.n_prev_actions = n_prev_actions
        self.drone_radius = drone_radius
        self.cos_max_tilt = math.cos(math.radians(max_tilt_deg))
        self.n_frames, self.image_size, self.render_size = n_frames, image_size, render_size
        self.image_device = th.device(image_device or ("cuda" if th.cuda.is_available() else "cpu"))
        self.augment_cfg = None if augment is False else {**DEFAULT_AUGMENT, **(augment or {})}
        is_collision_reset = True
        if n_frames > 0:
            import habitat_sim
            visual = True
            fwd, left, up = gg.CAMERA_OFFSET_BODY
            sensor_kwargs = [{"sensor_type": habitat_sim.SensorType.COLOR, "uuid": "color",
                              "resolution": [render_size, render_size], "hfov": gg.CAMERA_HFOV_DEG,
                              "position": [-left, up, -fwd]}]  # Habitat axes: x right, y up, -z forward
            scene_kwargs = {"path": DEFAULT_SCENE, **scene_kwargs}
            is_collision_reset = False  # get_failure covers walls and gate (with the crash penalty)

        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs or {},
            dynamics_kwargs=dynamics_kwargs,
            sensor_kwargs=sensor_kwargs,
            scene_kwargs=scene_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
            is_collision_reset=is_collision_reset,
        )
        # replace the default start generators
        gen = GateStartGenerator(self.gate_pos, self.gate_yaw, self.room, self.start_cfg, seed=seed)
        self.envs.stateGenerators = [gen] * self.num_agent

        # gate frame and hover point as tensors (world frame)
        c, s = math.cos(self.gate_yaw), math.sin(self.gate_yaw)
        self.R_gate = th.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # gate -> world
        self.hover_point = th.tensor(gg.hover_point(self.gate_pos, self.gate_yaw, hover_distance), dtype=th.float32)
        self._opening_w = th.tensor(  # corners TL, TR, BR, BL as seen from the front
            gg.gate_to_world(gg.gate_keypoints()["opening_front"], self.gate_pos, self.gate_yaw), dtype=th.float32)
        self.opening_centre = th.tensor(
            gg.gate_to_world(gg.gate_keypoints()["opening_center"], self.gate_pos, self.gate_yaw)[0], dtype=th.float32)
        self._build_collision_boxes()

        dyn = self.envs.dynamics
        self.hover_thrust = dyn._hover_action[0, 0].item()  # VisFly's normalised thrust at hover
        self.hover_action = th.zeros(4)                     # in the centred action space
        self._prev_actions = self.hover_action.repeat(self.num_agent, n_prev_actions, 1)
        self._prev_dist = th.zeros(self.num_agent)
        n_priv = dyn.randomized_params.shape[1]
        obs_spaces = {
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(15 + 4 * n_prev_actions,), dtype=np.float32),
            "privileged_state": spaces.Box(low=-np.inf, high=np.inf, shape=(n_priv,), dtype=np.float32),
        }
        if n_frames > 0:
            obs_spaces["image"] = spaces.Box(low=0, high=255, shape=(n_frames, image_size, image_size), dtype=np.uint8)
            obs_spaces["corners"] = spaces.Box(low=-2, high=2, shape=(12,), dtype=np.float32)
            if n_prev_actions > 0:
                obs_spaces["actions"] = spaces.Box(low=-1, high=1, shape=(4 * n_prev_actions,), dtype=np.float32)
            self._frames = th.zeros((self.num_agent, n_frames, image_size, image_size), dtype=th.uint8)
            self._aug = {}
            self._sample_augmentation(th.arange(self.num_agent))
            self._push_frame = False   # step(): a new frame was rendered for every drone
            self._fill_frames = None   # reset: these drones start a new stack from their first frame
            self._img_rng = th.Generator(device=self.image_device).manual_seed(seed)
        self.observation_space = spaces.Dict(obs_spaces)

    # ---------- geometry helpers ----------
    def _build_collision_boxes(self):
        """Gate parts as boxes in the gate frame, inflated by the drone radius."""
        r = self.drone_radius
        self._boxes = []
        for p in gg.gate_parts():
            a = math.radians(p["rot_y_deg"])
            Ry = th.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], dtype=th.float32)
            half = th.tensor(p["size"], dtype=th.float32) / 2 + r
            hole = None
            if p["cutout"]:
                hole = (th.tensor(p["cutout"]["center"], dtype=th.float32),
                        th.tensor(p["cutout"]["size"], dtype=th.float32) / 2 - r)
            self._boxes.append((th.tensor(p["center"], dtype=th.float32), Ry, half, hole))

    def to_gate_frame(self, p_world):
        return (p_world - th.tensor(self.gate_pos, dtype=th.float32)) @ self.R_gate

    def gate_collision(self, pos):
        pg = self.to_gate_frame(pos)
        hit = th.zeros(len(pos), dtype=th.bool)
        for centre, Ry, half, hole in self._boxes:
            local = (pg - centre) @ Ry  # into the part's (possibly rotated) frame
            inside = (local.abs() < half).all(dim=1)
            if hole is not None:
                hc, hh = hole
                in_hole = ((pg[:, 1:] - hc[1:]).abs() < hh[1:]).all(dim=1)
                inside = inside & ~in_hole
            hit = hit | inside
        return hit

    def _rotation_matrices(self):
        q = self.orientation.cpu()  # w, x, y, z
        w, x, y, z = q.unbind(1)
        return th.stack([
            th.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
            th.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
            th.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1),
        ], 1)  # body -> world, (N, 3, 3)

    # ---------- camera images ----------
    def corner_labels(self):
        """Labels for the auxiliary corner task, (N, 12): the opening's corners TL, TR, BR, BL in the
        newest frame as (u, v) in normalised image coordinates (the image spans [-1, 1], u right,
        v down; pixels = (u + 1) * image_size / 2), then 4 flags: 1 if the corner is in front of
        the camera and inside the image. Coordinates of corners that are not visible are 0.
        Same pinhole model as tools/gate_geometry.project_points (checked in tools/corner_check.py)."""
        R = self._rotation_matrices()                                       # body -> world
        p = self._opening_w[None] - self.position.cpu()[:, None]            # (N, 4, 3), world
        p_cam = th.einsum("nkw,nwb->nkb", p, R) - th.tensor(gg.CAMERA_OFFSET_BODY, dtype=th.float32)
        depth = p_cam[..., 0]                                               # body: x forward, y left, z up
        f = 1.0 / math.tan(math.radians(gg.CAMERA_HFOV_DEG) / 2)           # square image: same for u and v
        d = depth.clamp(min=1e-3)
        uv = th.stack([-f * p_cam[..., 1] / d, -f * p_cam[..., 2] / d], -1)
        visible = (depth > 0.05) & (uv.abs() <= 1).all(-1)
        uv = th.where(visible[..., None], uv, th.zeros_like(uv))
        return th.cat([uv.reshape(-1, 8), visible.float()], 1)

    def _sample_augmentation(self, idx):
        n, c = len(idx), self.augment_cfg
        if c is None:
            values = {"gain": th.ones(n), "gamma": th.ones(n), "offset": th.zeros(n), "blur_sigma": th.zeros(n),
                      "noise_std": th.zeros(n)}
        else:
            values = {k: th.empty(n).uniform_(*c[k]) for k in ("gain", "gamma", "offset", "blur_sigma", "noise_std")}
        for k, v in values.items():
            self._aug.setdefault(k, th.zeros(self.num_agent))[idx] = v

    def _blur(self, x, sigma):
        """Gaussian blur with a different sigma per image. x: (n, 1, H, W) float."""
        r = 3
        t = th.arange(-r, r + 1, dtype=th.float32, device=x.device)
        k = th.exp(-0.5 * (t / sigma.clamp(min=1e-3)[:, None]) ** 2)
        k = k / k.sum(1, keepdim=True)                       # (n, 7); sigma ~0 gives a delta
        n = len(x)
        y = F.conv2d(F.pad(x.transpose(0, 1), (r, r, 0, 0), mode="replicate"), k[:, None, None, :], groups=n)
        y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k[:, None, :, None], groups=n)
        return y.transpose(0, 1)

    def _color_images(self, idx):
        """Rendered images of drones idx as a uint8 tensor (n, R, R, C). VisFly keeps them as a
        transposed (N, 3, R, R) view of Habitat's contiguous (N, R, R, 4) RGBA array; gathering from
        the view costs ~20 ms per step with 64 drones, so read the array underneath when possible."""
        view = self.envs.sensor_obs["color"]
        base = view.base
        if isinstance(base, np.ndarray) and base.ndim == 4 and base.shape[:3] == (self.num_agent, *view.shape[2:]) \
                and base.flags["C_CONTIGUOUS"] and np.shares_memory(base, view):
            return th.from_numpy(base if len(idx) == self.num_agent else base[idx.numpy()])
        return th.from_numpy(np.ascontiguousarray(view[idx.numpy()].transpose(0, 2, 3, 1)))

    def _render_frames(self, idx):
        """Current camera images of drones idx as augmented uint8 grayscale (n, S, S)."""
        dev = self.image_device
        rgb = self._color_images(idx).to(dev).permute(0, 3, 1, 2)[:, :3].float()  # (n, 3, R, R), 0..255
        gray = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2])[:, None] / 255.0
        if self.render_size != self.image_size:
            gray = F.adaptive_avg_pool2d(gray, self.image_size)
        if self.augment_cfg is not None:
            a = {k: v[idx].to(dev)[:, None, None, None] for k, v in self._aug.items()}
            gray = self._blur(gray, a["blur_sigma"][:, 0, 0, 0])
            jitter = 1 + self.augment_cfg["frame_gain"] * (
                2 * th.rand(len(idx), generator=self._img_rng, device=dev) - 1)[:, None, None, None]
            gray = (gray * a["gain"] * jitter).clamp(0, 1) ** a["gamma"]
            gray = gray * 255 + a["offset"] + a["noise_std"] * th.randn(gray.shape, generator=self._img_rng, device=dev)
        else:
            gray = gray * 255
        return gray[:, 0].round().clamp(0, 255).to(th.uint8).cpu()

    def _update_frames(self):
        if self._push_frame:  # after a step: shift every stack by one, newest frame first
            self._frames = th.cat([self._render_frames(th.arange(self.num_agent))[:, None], self._frames[:, :-1]], 1)
            self._push_frame = False
        if self._fill_frames is not None:  # after a reset: no history yet, repeat the first frame
            idx = self._fill_frames
            self._frames[idx] = self._render_frames(idx)[:, None].expand(-1, self.n_frames, -1, -1)
            self._fill_frames = None

    # ---------- VisFly interface ----------
    def get_observation(self, indices=None, predicted_obs=None) -> TensorDict:
        teacher = self._teacher_observation()
        if self.n_frames == 0:
            return TensorDict(teacher)
        self._update_frames()
        obs = {**teacher, "image": self._frames.clone()}
        if self.n_prev_actions > 0:
            obs["actions"] = self._prev_actions.reshape(self.num_agent, -1).clone()
        obs["corners"] = self.corner_labels()
        return TensorDict(obs)

    def _teacher_observation(self):
        pos, vel, rate = self.position.cpu(), self.velocity.cpu(), self.angular_velocity.cpu()
        R = self._rotation_matrices()
        R_rel = self.R_gate.T @ R                       # body -> gate frame
        err = (pos - self.hover_point) @ self.R_gate    # position error in the gate frame
        state = th.cat([
            err / 2.0,
            (vel @ self.R_gate) / 2.0,
            R_rel[:, :, 0], R_rel[:, :, 1],             # 6D orientation (first two columns)
            rate / 3.0,
            self._prev_actions.reshape(self.num_agent, -1),
        ], dim=1)
        return {"state": state, "privileged_state": self.envs.dynamics.randomized_params.cpu()}

    def get_success(self) -> th.Tensor:
        return th.zeros(self.num_agent, dtype=th.bool)  # hold until the time limit, never end early

    def get_failure(self) -> th.Tensor:
        pos = self.position.cpu()
        r, rad = self.room, self.drone_radius
        out_of_room = (pos[:, 0] < r["x"][0] + rad) | (pos[:, 0] > r["x"][1] - rad) | \
                      (pos[:, 1] < r["y"][0] + rad) | (pos[:, 1] > r["y"][1] - rad) | \
                      (pos[:, 2] < r["z"][0] + 0.05) | (pos[:, 2] > r["z"][1] - rad)
        tilted = self._rotation_matrices()[:, 2, 2] < self.cos_max_tilt
        return out_of_room | tilted | self.gate_collision(pos)

    def _distance(self):
        err = (self.position.cpu() - self.hover_point) @ self.R_gate
        w = th.tensor(self.reward_cfg["axis_weight"])
        return err.norm(dim=1), (err * w).norm(dim=1)

    def get_reward(self, predicted_obs=None) -> Dict:
        c = self.reward_cfg
        d, d_weighted = self._distance()
        vel, rate = self.velocity.cpu(), self.angular_velocity.cpu()
        R = self._rotation_matrices()
        to_c = self.opening_centre - self.position.cpu()
        cos_view = (R[:, :, 0] * to_c).sum(1) / to_c.norm(dim=1).clamp(min=1e-6)
        terms = {
            "progress": c["progress"] * (self._prev_dist - d),
            "closeness": c["closeness"] * th.exp(-d_weighted / c["closeness_scale"]),
            "hold": c["hold"] * th.exp(-(d / c["hold_sigma"]) ** 2),
            "velocity": -c["velocity"] * vel.norm(dim=1) * th.exp(-d / 0.5),
            "rate": -c["rate"] * rate.norm(dim=1),
            "view": -c["view"] * (1 - cos_view),
            "smooth": -c["smooth"] * ((self._prev_actions[:, 0] - self._prev_actions[:, 1]) ** 2).sum(1),
            "crash": -c["crash"] * self._failure.float() if self._failure is not None else th.zeros(self.num_agent),
        }
        reward = sum(terms.values())
        # diagnostics (not part of the reward): steps within 5 cm, final-ish distance
        diag = {"in_5cm": (d < 0.05).float(), "in_10cm": (d < 0.10).float()}
        return {"reward": reward, **terms, **diag}

    # ---------- action centring ----------
    def policy_to_dynamics(self, a):
        """Centred policy action -> VisFly's normalised action (thrust 0 -> hover)."""
        a = th.as_tensor(a, dtype=th.float32).clone()
        h, t = self.hover_thrust, a[..., 0]
        a[..., 0] = th.where(t >= 0, h + t * (1 - h), h + t * (1 + h))
        return a

    def dynamics_to_policy(self, a):
        a = th.as_tensor(a, dtype=th.float32).clone()
        h, t = self.hover_thrust, a[..., 0]
        a[..., 0] = th.where(t >= h, (t - h) / (1 - h), (t - h) / (1 + h))
        return a

    # ---------- bookkeeping for previous commands and progress ----------
    def step(self, _action, is_test=False):
        a = th.as_tensor(_action, dtype=th.float32).cpu()
        self._prev_actions = th.cat([a.unsqueeze(1), self._prev_actions[:, :-1]], dim=1)
        self._push_frame = self.n_frames > 0
        out = super().step(self.policy_to_dynamics(a).clamp(-1, 1), is_test=is_test)
        self._prev_dist = self._distance()[0]  # reset drones already have their new start here
        return out

    def reset(self, *args, **kwargs):
        self._prev_actions = self.hover_action.repeat(self.num_agent, self.n_prev_actions, 1)
        self._failure = th.zeros(self.num_agent, dtype=th.bool)
        if self.n_frames > 0:
            self._sample_augmentation(th.arange(self.num_agent))
            self._fill_frames = th.arange(self.num_agent)
        obs = super().reset(*args, **kwargs)
        self._prev_dist = self._distance()[0]
        return obs

    def reset_agent_by_id(self, agent_indices=None, state=None, reset_obs=None):
        idx = th.as_tensor(agent_indices, dtype=th.long)
        self._prev_actions[idx] = self.hover_action  # the delay buffer is filled with hover, too
        if self.n_frames > 0:
            self._sample_augmentation(idx)
            self._fill_frames = idx
        return super().reset_agent_by_id(agent_indices, state=state, reset_obs=reset_obs)
