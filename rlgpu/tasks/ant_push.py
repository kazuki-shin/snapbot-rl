# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import torch

from rlgpu.utils.torch_jit_utils import *
from rlgpu.tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import ipdb


class AntPush(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # Observations:
        # 0:3 - ant pose (x, y, theta)
        # 3:6 - box pose (x, y, theta)
        # 6:9 - goal pose (x, y, theta)
        self.cfg["env"]["numObservations"] = 6

        # Action: target abstracted high-level action
        # Forward, Backward, Left, Right, Turn Left, Turn Right
        self.cfg["env"]["numActions"] = 6

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actors_per_env = 2
        dofs_per_env = 8 # why 8? 
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, actors_per_env, 13)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, dofs_per_env, 2)

        self.root_states = vec_root_tensor

        ant_xy = vec_root_tensor[..., 0, 0:2] 
        ant_rotation = vec_root_tensor[..., 0, 3:7]
        ant_theta = get_euler_xyz(ant_rotation)[2] # extract z axis from quat 
        self.ant_positions = torch.cat((ant_xy, ant_theta.unsqueeze(1)), 1) # (x,y,theta)
        
        box_xy = vec_root_tensor[..., 1, 0:2] 
        box_rotation = vec_root_tensor[..., 1, 3:7]
        box_theta = get_euler_xyz(box_rotation)[2] # extract z axis from quat 
        self.box_positions = torch.cat((box_xy, box_theta.unsqueeze(1)), 1) # (x,y,theta)

        self.dof_states = vec_dof_tensor
        self.dof_positions = vec_dof_tensor[..., 0]
        self.dof_velocities = vec_dof_tensor[..., 1]

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()

        self.dof_position_targets = torch.zeros((self.num_envs, dofs_per_env), dtype=torch.float32, device=self.device, requires_grad=False)


    def create_sim(self):
        # set z-axis as the upaxis in sim
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        # create ground plane and all training envs
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../../assets"
        asset_file = "mjcf/nv_ant.xml"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.angular_damping = 0.0

        # load ant asset
        ant_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(ant_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)

        actuator_props = self.gym.get_asset_actuator_properties(ant_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self.joint_gears = to_torch(motor_efforts, device=self.device)

        # start pose of ant
        ant_start_pose = gymapi.Transform()
        ant_start_pose.p = gymapi.Vec3(*get_axis_params(0.44, self.up_axis_idx))

        # set box asset information  (density, dim)
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1.0
        width = height = depth = 0.5

        # starting pose of box object
        box_start_pose = gymapi.Transform()
        box_start_pose.p = gymapi.Vec3(5.0, 5.0, 1.0)
        # load box asset
        box_asset = self.gym.create_box(self.sim, width, height, depth, asset_options)

        # handles to keep track of actors and envs
        self.ant_handles = []
        self.box_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            ant_handle = self.gym.create_actor(env_ptr, ant_asset, ant_start_pose, "ant", i, 1, 0)
            box_handle = self.gym.create_actor(env_ptr, box_asset, box_start_pose, "box", i, 1, 0)

            # set color of ants to orange
            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.gym.set_rigid_body_color(env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.06, 0.38, 0.97))

            self.envs.append(env_ptr)
            self.ant_handles.append(ant_handle)
            self.box_handles.append(box_handle)


    def compute_reward(self):
         self.rew_buf[:], self.reset_buf[:] = compute_ant_reward(
            self.ant_positions,
            self.box_positions)

    def compute_observations(self):
        # print("~!~!~!~! Computing obs")

        self.obs_buf[..., 0:3] = self.ant_positions 
        self.obs_buf[..., 3:6] = self.box_positions
        # print("Ant position", self.obs_buf[0])
        # print("Box position", self.obs_buf[1])
        return self.obs_buf

    def reset(self, env_ids):
        pass

    def pre_physics_step(self, actions):
        self.dof_position_targets += 100
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_position_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_ant_reward(ant_positions, box_positions):
    reward = 1
    reset = 0
    return reward, reset
