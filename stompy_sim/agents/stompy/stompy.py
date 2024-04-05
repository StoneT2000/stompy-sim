import os.path as osp
import numpy as np
import sapien
from mani_skill.utils import sapien_utils
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.sensors.camera import CameraConfig
from mani_skill.agents.registration import register_agent
from transforms3d import euler
import torch
@register_agent()
class Stompy(BaseAgent):
    uid = "stompy"
    urdf_path = osp.join(osp.dirname(__file__), "description/robot.urdf")
    urdf_config = dict()

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = ['joint_right_arm_1_x8_1_dof_x8', 'joint_left_arm_2_x8_1_dof_x8', 'joint_head_1_x4_1_dof_x4', 'joint_torso_1_x8_1_dof_x8', 'joint_right_arm_1_x8_2_dof_x8', 'joint_left_arm_2_x8_2_dof_x8', 'joint_legs_1_x8_1_dof_x8', 'joint_legs_1_x8_2_dof_x8', 'joint_head_1_x4_2_dof_x4', 'joint_right_arm_1_x6_1_dof_x6', 'joint_left_arm_2_x6_1_dof_x6', 'joint_legs_1_right_leg_1_x8_1_dof_x8', 'joint_legs_1_left_leg_1_x8_1_dof_x8', 'joint_right_arm_1_x6_2_dof_x6', 'joint_left_arm_2_x6_2_dof_x6', 'joint_legs_1_right_leg_1_x10_2_dof_x10', 'joint_legs_1_left_leg_1_x10_1_dof_x10', 'joint_legs_1_right_leg_1_knee_revolute', 'joint_legs_1_left_leg_1_knee_revolute', 'joint_right_arm_1_x4_1_dof_x4', 'joint_left_arm_2_x4_1_dof_x4', 'joint_legs_1_right_leg_1_x10_1_dof_x10', 'joint_legs_1_right_leg_1_ankle_revolute', 'joint_legs_1_left_leg_1_x10_2_dof_x10', 'joint_legs_1_left_leg_1_ankle_revolute', 'joint_legs_1_right_leg_1_x6_1_dof_x6', 'joint_legs_1_left_leg_1_x6_1_dof_x6', 'joint_legs_1_right_leg_1_x4_1_dof_x4', 'joint_legs_1_left_leg_1_x4_1_dof_x4', 'joint_right_arm_1_hand_1_x4_1_dof_x4', 'joint_left_arm_2_hand_1_x4_1_dof_x4', 'joint_right_arm_1_hand_1_slider_1', 'joint_right_arm_1_hand_1_slider_2', 'joint_left_arm_2_hand_1_slider_1', 'joint_left_arm_2_hand_1_slider_2', 'joint_right_arm_1_hand_1_x4_2_dof_x4', 'joint_left_arm_2_hand_1_x4_2_dof_x4']
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100
        self.gripper_joint_names = [
            "joint_right_arm_1_hand_1_slider_1",
            "joint_right_arm_1_hand_1_slider_2",
            "joint_left_arm_2_hand_1_slider_1",
            "joint_left_arm_2_hand_1_slider_2",
        ]
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100
        super().__init__(*args, fix_root_link=False, **kwargs)

    @property
    def init_standing_qpos(self):
        return torch.Tensor([
            [1.5, -1.5, -0.942, 0, 1.5, -1.5, -0.25, 0.25, 0.5, 0.65, -0.65, -0.5, -0.5, 0, 0, -.5, .5, 0.78, -0.78, 0.25, -0.25, 0, -0.4, 0, 0.4, 0, 0, -0.2, 0.2, 0.2, -2.2, 0, 0, 0, 0, 0, 0]
        ])

    @property
    def _controller_configs(self):
        # raise NotImplementedError()
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -0.034,  # a trick to have force when the object is thin
            0.0,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )
        return dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, grippers=gripper_pd_joint_pos, balance_passive_force=False,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos, grippers=gripper_pd_joint_pos, balance_passive_force=False,
            )
        )

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="head_camera",
                pose=sapien.Pose(p=[0.12, 0, 0.02], q=euler.euler2quat(-np.pi/2, 0, 0)),
                width=128,
                height=128,
                fov=1.57,
                near=0.01,
                far=100,
                # intrinsic = , # you can specify an intrinsic matrix directly here instead of defining the ther values
                entity_uid="link_head_1_head_1", # mount cameras relative to existing link IDs as so
            )
        ]

    def _after_init(self):
        pass
    def _load_articulation(self):
        """
        Load the robot articulation
        """
        loader = self.scene.create_urdf_loader()
        loader.name = self.uid
        if self._agent_idx is not None:
            loader.name = f"{self.uid}-agent-{self._agent_idx}"
        loader.fix_root_link = self.fix_root_link

        urdf_path = self.urdf_path

        urdf_config = sapien_utils.parse_urdf_config(self.urdf_config, self.scene)
        sapien_utils.check_urdf_config(urdf_config)

        # TODO(jigu): support loading multiple convex collision shapes
        sapien_utils.apply_urdf_config(loader, urdf_config)
        loader.disable_self_collisions = True
        self.robot = loader.load(urdf_path)
        assert self.robot is not None, f"Fail to load URDF from {urdf_path}"