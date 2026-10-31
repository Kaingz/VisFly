"""Fly from a random start in the room to a hover point in front of the gate and hold it.

Phase 4 (state-based teacher): the actor sees the true state relative to the hover point plus
its own previous commands; the critic additionally sees the randomised physical parameters
(asymmetric actor-critic). Camera observations come in phase 5.

Gate geometry, layout and camera projection come from tools/gate_geometry.py (repo root must be
on sys.path, as in exps/examples/run.py). The gate pose comes from a scene's .gate.json sidecar.
"""
import json
import math
from typing import Dict, Optional

import numpy as np
import torch as th
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
    ):
        with open(gate_pose_file) as f:
            pose = json.load(f)
        self.gate_pos, self.gate_yaw = pose["gate_pos"], pose["gate_yaw"]
        self.room = gg.load_json(gg.ROOM_SPEC)
        self.start_cfg = {**DEFAULT_START, **(start_kwargs or {})}
        self.reward_cfg = {**DEFAULT_REWARD, **(reward_kwargs or {})}
        self.n_prev_actions = n_prev_actions
        self.drone_radius = drone_radius
        self.cos_max_tilt = math.cos(math.radians(max_tilt_deg))

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
        )
        # replace the default start generators
        gen = GateStartGenerator(self.gate_pos, self.gate_yaw, self.room, self.start_cfg, seed=seed)
        self.envs.stateGenerators = [gen] * self.num_agent

        # gate frame and hover point as tensors (world frame)
        c, s = math.cos(self.gate_yaw), math.sin(self.gate_yaw)
        self.R_gate = th.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # gate -> world
        self.hover_point = th.tensor(gg.hover_point(self.gate_pos, self.gate_yaw, hover_distance), dtype=th.float32)
        self.opening_centre = th.tensor(
            gg.gate_to_world(gg.gate_keypoints()["opening_center"], self.gate_pos, self.gate_yaw)[0], dtype=th.float32)
        self._build_collision_boxes()

        dyn = self.envs.dynamics
        self.hover_thrust = dyn._hover_action[0, 0].item()  # VisFly's normalised thrust at hover
        self.hover_action = th.zeros(4)                     # in the centred action space
        self._prev_actions = self.hover_action.repeat(self.num_agent, n_prev_actions, 1)
        self._prev_dist = th.zeros(self.num_agent)
        n_priv = dyn.randomized_params.shape[1]
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(15 + 4 * n_prev_actions,), dtype=np.float32),
            "privileged_state": spaces.Box(low=-np.inf, high=np.inf, shape=(n_priv,), dtype=np.float32),
        })

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

    # ---------- VisFly interface ----------
    def get_observation(self, indices=None, predicted_obs=None) -> TensorDict:
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
        return TensorDict({
            "state": state,
            "privileged_state": self.envs.dynamics.randomized_params.cpu(),
        })

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
        out = super().step(self.policy_to_dynamics(a).clamp(-1, 1), is_test=is_test)
        self._prev_dist = self._distance()[0]  # reset drones already have their new start here
        return out

    def reset(self, *args, **kwargs):
        self._prev_actions = self.hover_action.repeat(self.num_agent, self.n_prev_actions, 1)
        self._failure = th.zeros(self.num_agent, dtype=th.bool)
        obs = super().reset(*args, **kwargs)
        self._prev_dist = self._distance()[0]
        return obs

    def reset_agent_by_id(self, agent_indices=None, state=None, reset_obs=None):
        idx = th.as_tensor(agent_indices, dtype=th.long)
        self._prev_actions[idx] = self.hover_action  # the delay buffer is filled with hover, too
        return super().reset_agent_by_id(agent_indices, state=state, reset_obs=reset_obs)
