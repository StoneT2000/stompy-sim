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
from mani_skill.utils.structs.types import SimConfig, SceneConfig, GPUMemoryConfig
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
@register_env("Stand-v0", max_episode_steps=100)
class StandEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["stompy"]
    agent: Union[Stompy]

    def __init__(self, *args, robot_uids="stompy", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_cfg(self):
        return SimConfig(
            sim_freq=120, 
            control_freq=60,
            gpu_memory_cfg=GPUMemoryConfig(max_rigid_contact_count=2**21, max_rigid_patch_count=2**19),
            # scene configs copied form Isaac humanoid sim configs
            scene_cfg=SceneConfig(solver_iterations=4, solver_velocity_iterations=0, bounce_threshold=0.2)
        )

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
        # self.table_scene = TableSceneBuilder(self)
        # self.table_scene.build()
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=(0.1, 0.1, 0.8), )
        builder.add_box_visual(half_size=(0.1, 0.1, 0.8), material=sapien.render.RenderMaterial(base_color=[0.2, 0.3, 0.8, 1]))
        self.box1 = builder.build_kinematic(name="box1")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx) # number of envs resetting
            # self.table_scene.initialize(env_idx)
            self.agent.robot.set_pose(sapien.Pose(p=[0, 0, 1])) # place robot base pose at 1 meter high (z-axis)
            self.agent.robot.set_qpos(self.agent.init_standing_qpos)

            # randomize the box xy pos for fun
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.4
            xyz[:, 1] = -1.5
            xyz[:, 0] = torch.rand((b)) * 0.5 - 0.25
            self.box1.set_pose(Pose.create_from_pq(p=xyz))

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
