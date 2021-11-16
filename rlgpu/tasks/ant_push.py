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

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

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
        self.cfg["env"]["numActions"] = 8

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        # get gym GPU state tensors
        actors_per_env = 2
        dofs_per_env = 8  # why 8?
        sensors_per_env = 4

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        vec_root_tensor = gymtorch.wrap_tensor(
            self.root_tensor).view(self.num_envs, actors_per_env, 13)
        vec_dof_tensor = gymtorch.wrap_tensor(
            self.dof_state_tensor).view(self.num_envs, dofs_per_env, 2)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(
            self.sensor_tensor).view(self.num_envs, sensors_per_env, 6)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.root_states = vec_root_tensor

        self.dof_states = vec_dof_tensor
        self.dof_positions = vec_dof_tensor[..., 0]
        self.dof_velocities = vec_dof_tensor[..., 1]

        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()

        self.dof_position_targets = torch.zeros(
            (self.num_envs, dofs_per_env), dtype=torch.float32, device=self.device, requires_grad=False)

        ant_xy = self.root_states[..., 0, 0:2]
        ant_rotation = self.root_states[..., 0, 3:7]
        ant_theta = get_euler_xyz(ant_rotation)[2]  # extract z axis from quat
        self.ant_positions = torch.cat(
            (ant_xy, ant_theta.unsqueeze(1)), 1)  # (x,y,theta)

        box_xy = self.root_states[..., 1, 0:2]
        box_rotation = self.root_states[..., 1, 3:7]
        box_theta = get_euler_xyz(box_rotation)[2]  # extract z axis from quat
        self.box_positions = torch.cat(
            (box_xy, box_theta.unsqueeze(1)), 1)  # (x,y,theta)

    def create_sim(self):
        # set z-axis as the upaxis in sim
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id,
                                      self.physics_engine, self.sim_params)

        # create ground plane and all training envs
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

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
        ant_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(ant_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)

        actuator_props = self.gym.get_asset_actuator_properties(ant_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self.joint_gears = to_torch(motor_efforts, device=self.device)

        # start pose of ant
        ant_start_pose = gymapi.Transform()
        ant_start_pose.p = gymapi.Vec3(
            *get_axis_params(0.44, self.up_axis_idx))

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)
        body_names = [self.gym.get_asset_rigid_body_name(
            ant_asset, i) for i in range(self.num_bodies)]
        extremity_names = [s for s in body_names if "foot" in s]
        self.extremities_index = torch.zeros(
            len(extremity_names), dtype=torch.long, device=self.device)

        # set box asset information  (density, dim)
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1.0
        width = height = depth = 0.7

        # starting pose of box object
        box_start_pose = gymapi.Transform()
        box_start_pose.p = gymapi.Vec3(5.0, 5.0, 3.0)
        # load box asset
        box_asset = self.gym.create_box(
            self.sim, width, height, depth, asset_options)

        # handles to keep track of actors and envs
        self.ant_handles = []
        self.obj_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            ant_handle = self.gym.create_actor(
                env_ptr, ant_asset, ant_start_pose, "ant", i, 1, 0)
            box_handle = self.gym.create_actor(
                env_ptr, box_asset, box_start_pose, "box", i, 0, 0)

            env_sensors = []
            for extr in extremity_names:
                extr_handle = self.gym.find_actor_rigid_body_handle(
                    env_ptr, ant_handle, extr)
                env_sensors.append(self.gym.create_force_sensor(
                    env_ptr, extr_handle, sensor_pose))
                self.sensors.append(env_sensors)

            # set color of ants to orange
            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.gym.set_rigid_body_color(
                env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.06, 0.38, 0.97))

            self.envs.append(env_ptr)
            self.ant_handles.append(ant_handle)
            self.obj_handles.append(box_handle)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, ant_handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(
            self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(
            self.dof_limits_upper, device=self.device)

        for i in range(len(extremity_names)):
            self.extremities_index[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.ant_handles[0], extremity_names[i])

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_ant_reward(
            self.ant_positions,
            self.box_positions,
            self.reset_buf,
            self.progress_buf)

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.obs_buf[:], self.ant_positions[:], self.box_positions[:] = compute_ant_observations(
            self.obs_buf, self.root_states, self.progress_buf)

    def reset(self, env_ids):
        self.root_states[env_ids] = self.initial_root_states[env_ids]

        # reset root state for ants and boxes in selected envs
        self.gym.set_actor_root_state_tensor(self.sim, self.root_tensor)

        # reset DOF states for bbots in selected envs
        self.dof_states[env_ids] = self.initial_dof_states[env_ids]
        self.gym.set_dof_state_tensor(self.sim, self.dof_state_tensor)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        self.actions = actions.clone().to(self.device)
        forces = self.actions * self.joint_gears * self.power_scale
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.progress_buf += 1

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_ant_reward(ant_positions, box_positions, reset_buf, progress_buf):
    reward = 1
    # calculate euclidean distance between ant and box
    ant_box_dist = torch.sqrt(torch.square(ant_positions[..., 0] - box_positions[..., 0]) +
                              torch.square(ant_positions[..., 1] - box_positions[..., 1]))
    reward = 1.0 / (1.0 + ant_box_dist)
    reset = torch.where(ant_positions[..., 0] >
                        1, torch.ones_like(reset_buf), reset_buf)
    return reward, reset


@torch.jit.script
def compute_ant_observations(obs_buf, root_states, progress_buf):
    ant_xy = root_states[..., 0, 0:2]
    ant_rotation = root_states[..., 0, 3:7]
    ant_theta = get_euler_xyz(ant_rotation)[2]  # extract z axis from quat
    ant_positions = torch.cat(
        (ant_xy, ant_theta.unsqueeze(1)), 1)  # (x,y,theta)

    box_xy = root_states[..., 1, 0:2]
    box_rotation = root_states[..., 1, 3:7]
    box_theta = get_euler_xyz(box_rotation)[2]  # extract z axis from quat
    box_positions = torch.cat(
        (box_xy, box_theta.unsqueeze(1)), 1)  # (x,y,theta)

    obs_buf[..., 0:3] = ant_positions
    obs_buf[..., 3:6] = box_positions

    if progress_buf[0] % 100 == 0:
        print("~!~!~!~! Computing obs")
        print("Ant/Box pos", obs_buf)
        # print(self.dof_states[0])

    return obs_buf, ant_positions, box_positions
