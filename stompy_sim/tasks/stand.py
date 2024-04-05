from collections import OrderedDict
from typing import Any, Dict, Union

import numpy as np
import torch
import sapien
from stompy_sim.agents.stompy.stompy import Stompy
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig, SceneConfig


@register_env("Stand-v0", max_episode_steps=100)
class StandEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["stompy"]
    agent: Union[Stompy]

    def __init__(self, *args, robot_uids="stompy", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_cfg(self):
        return SimConfig(scene_cfg=SceneConfig(solver_iterations=4, solver_velocity_iterations=0))

    @property
    def _sensor_configs(self):
        pose = sapien_utils.look_at(eye=[1, -1, 1.25], target=[0, 0, 0.7])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[1, -1, 1.25], target=[0, 0, 0.7])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.ground = ground.build_ground(self._scene)
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.agent.robot.set_pose(sapien.Pose(p=[0, 0, 1]))
        self.agent.robot.set_qpos(self.agent.init_standing_qpos)
    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: Dict):
        return OrderedDict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
