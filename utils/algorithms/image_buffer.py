import numpy as np
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer


class ImageDictRolloutBuffer(DictRolloutBuffer):
    """DictRolloutBuffer that keeps uint8 observations (camera images) as uint8 instead of float32.

    SB3 stores every observation as float32, which makes image rollouts 4x larger than needed
    (e.g. 16k samples of 4 x 128 x 128 frames: 4.3 GB instead of 1.1 GB). The policy's
    preprocessing converts uint8 images to floats in [0, 1] on the GPU, batch by batch.
    Select it with `rollout_buffer_class: ImageDictRolloutBuffer` in the PPO config.
    """

    def reset(self) -> None:
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            dtype = np.uint8 if self.observation_space.spaces[key].dtype == np.uint8 else np.float32
            self.observations[key] = np.zeros((self.buffer_size, self.n_envs, *obs_input_shape), dtype=dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()  # BaseBuffer.reset, as in SB3's DictRolloutBuffer
