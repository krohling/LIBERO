import os
import json
from easydict import EasyDict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.venv import SubprocVectorEnv
from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.utils import torch_load_model, get_task_embs
from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy

import robomimic.utils.obs_utils as ObsUtils

MODALITY_CONFIG = EasyDict({
    "data": {
        "obs":{
            "modality":{
                "rgb":[
                "agentview_rgb",
                "eye_in_hand_rgb"
                ],
                "depth":[
                
                ],
                "low_dim":[
                "gripper_states",
                "joint_states"
                ]
            }
        },
        "obs_key_mapping":{
            "agentview_rgb":"agentview_image",
            "eye_in_hand_rgb":"robot0_eye_in_hand_image",
            "gripper_states":"robot0_gripper_qpos",
            "joint_states":"robot0_joint_pos"
        },
    }
})


EMBED_CONFIG = EasyDict({
    "task_embedding_format": "bert",
    "data": {
        "max_word_len": 25,
    },
    "policy": {
        "language_encoder": {
            "network_kwargs": {
                "input_size": 768,
            }
        }
    }
})


def obs_to_tensor(obs_tensor_dict):
    obs_tensor_dict = raw_obs_to_tensor_obs(obs_tensor_dict, torch.randn(1, 16), MODALITY_CONFIG)['obs']
    # Flatten each observation type
    img1 = obs_tensor_dict["agentview_rgb"].flatten(start_dim=1)
    img2 = obs_tensor_dict["eye_in_hand_rgb"].flatten(start_dim=1)
    gripper = obs_tensor_dict["gripper_states"].flatten(start_dim=1)
    joints = obs_tensor_dict["joint_states"].flatten(start_dim=1)

    # Concatenate everything along the last dimension
    return torch.cat([img1, img2, gripper, joints], dim=1)  # (batch, total_features)


def make_policy(config_path):
    # config_path = "./cleanrl/ppo_continuous_action_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    config = EasyDict(config)
    return BCTransformerPolicy(config, config['shape_meta'])

def make_libero_envs(
        num_envs=2, 
        task_suite_name='libero_object', 
        task_id=0,
        horizon=600,
        camera_dim=128,
    ):
    benchmark_dict = benchmark.get_benchmark_dict()
    
    task_suite = benchmark_dict[task_suite_name]()
    # task_emb = task_suite.get_task_emb(task_id)
    # print(f"task_emb: {task_emb}")
    task = task_suite.get_task(task_id)
    task_description = task.language
    task_emb = get_task_embs(EMBED_CONFIG, [task_description]).squeeze()
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    print(f"task_bddl_file: {task_bddl_file}")

    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
        f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": camera_dim,
        "camera_widths": camera_dim,
        "horizon": horizon
    }

    env = SubprocVectorEnv(
        [lambda: OffScreenRenderEnv(**env_args) for _ in range(num_envs)]
    )
    env.reset()
    env.seed(0)

    init_states_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    init_states = torch.load(init_states_path)
    indices = np.arange(num_envs) % init_states.shape[0]
    init_states_ = init_states[indices]
    obs = env.set_init_state(init_states_)

    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": MODALITY_CONFIG.data.obs.modality})

    dummy_action = [[0.]*7] * num_envs
    for step in range(1):
        obs, _, _, _ = env.step(dummy_action)
        obs_tensor = obs_to_tensor(obs)

    # Flatten the shape for the FCN
    observation_space_dim = np.prod(obs_tensor[0].shape)
    env.task_emb = task_emb
    env.single_observation_space = torch.zeros(observation_space_dim)
    env.single_action_space = torch.zeros(7)

    return env


class LiberoAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.task_emb = envs.task_emb
        self.critic = make_policy('./policies/critic_config.json')
        self.actor = make_policy('./policies/actor_stoch_head_config.json')

    def get_value(self, obs):
        obs = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)
        critic_input = self.critic.preprocess_input(obs, train_mode=False)
        q_value = self.critic(critic_input).squeeze()

        return q_value

    def get_action_and_value(self, obs, action=None):
        obs = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)
        actor_input = self.actor.preprocess_input(obs, train_mode=False)

        probs = self.actor(actor_input)
        if action is None:
            action = probs.sample()
            action = action.squeeze()
        
        action_log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)

        critic_input = self.critic.preprocess_input(obs, train_mode=False)
        q_value = self.critic(critic_input).squeeze()
        
        return action, action_log_prob, entropy, q_value

class StochHeadLiberoAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.task_emb = envs.task_emb
        self.critic = make_policy('./policies/critic_config.json')
        self.actor = make_policy('./policies/actor_stoch_head_config.json')
        self.actor.load_state_dict(torch_load_model('./checkpoints/actor_stoch_head_250.pth', 'cpu')[0])

    def get_value(self, obs):
        obs = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)
        critic_input = self.critic.preprocess_input(obs, train_mode=False)
        q_value = self.critic(critic_input).squeeze()

        return q_value

    def get_action_and_value(self, obs, action=None):
        obs = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)
        actor_input = self.actor.preprocess_input(obs, train_mode=False)

        probs = self.actor(actor_input)
        if action is None:
            action = probs.sample()
            action = action.squeeze()
        
        action_log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)

        critic_input = self.critic.preprocess_input(obs, train_mode=False)
        q_value = self.critic(critic_input).squeeze()
        
        return action, action_log_prob, entropy, q_value


class GMMHeadLiberoAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.task_emb = envs.task_emb
        self.critic = make_policy('./policies/critic_config.json')
        self.actor = make_policy('./policies/actor_gmm_head_config.json')
        self.actor.load_state_dict(torch_load_model('./checkpoints/actor_gmm_head_200.pth', 'cpu')[0])

    def get_value(self, obs):
        obs = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)
        critic_input = self.critic.preprocess_input(obs, train_mode=False)
        q_value = self.critic(critic_input).squeeze()

        return q_value

    def get_action_and_value(self, obs, action=None):
        obs = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)
        actor_input = self.actor.preprocess_input(obs, train_mode=False)

        probs = self.actor(actor_input)
        if action is None:
            action = probs.sample()
            action = action.squeeze()
        
        action_log_prob = probs.log_prob(action).sum(1)

        # GMM Dist does not support encropy computation
        # entropy = probs.entropy().sum(1)
        entropy = torch.zeros_like(action_log_prob)

        critic_input = self.critic.preprocess_input(obs, train_mode=False)
        q_value = self.critic(critic_input).squeeze()
        
        return action, action_log_prob, entropy, q_value


class DetHeadLiberoAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.task_emb = torch.randn(768)
        self.critic = make_policy('./policies/critic_config.json')
        self.actor = make_policy('./policies/actor_det_head_config.json')
        self.actor.load_state_dict(torch_load_model('./checkpoints/actor_det_head_20.pth', 'cpu')[0])
        # self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.actor_logstd = nn.Parameter(torch.full((1, np.prod(envs.single_action_space.shape)), 0.001))

    def get_value(self, obs):
        obs = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)
        critic_input = self.critic.preprocess_input(obs, train_mode=False)
        q_value = self.critic(critic_input).squeeze()

        return q_value

    def get_action_and_value(self, obs, action=None):
        obs = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)

        actor_input = self.actor.preprocess_input(obs, train_mode=False)
        action_mean = self.actor(actor_input).squeeze()
        # print(f"action_mean.shape: {action_mean.shape}")

        if action_mean.ndim == 1:
            action_mean = action_mean.unsqueeze(0)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        # print(f"action_logstd.shape: {action_logstd.shape}")

        action_std = torch.exp(action_logstd)
        # print(f"action_std.shape: {action_std.shape}")

        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample().squeeze()
        
        action_log_prob = probs.log_prob(action).sum(1)
        # print(f"action_log_prob.shape: {action_log_prob.shape}")

        entropy = probs.entropy().sum(1)
        # print(f"entropy.shape: {entropy.shape}")
        
        critic_input = self.critic.preprocess_input(obs, train_mode=False)
        q_value = self.critic(critic_input).squeeze()
        
        return action, action_log_prob, entropy, q_value




# class LiberoAgent(nn.Module):
#     def __init__(self, envs):
#         super().__init__()
#         self.task_emb = torch.randn(768)
#         self.critic = make_policy('./critic_config.json')
#         self.actor = make_policy('./actor_config.json')
#         self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

#     def get_value(self, obs):
#         obs = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)
#         critic_input = self.critic.preprocess_input(obs, train_mode=False)
#         q_value = self.critic(critic_input).squeeze()

#         return q_value

#     def get_action_and_value(self, obs, action=None):
#         obs = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)

#         actor_input = self.actor.preprocess_input(obs, train_mode=False)
#         action_mean = self.actor(actor_input).squeeze()
#         # print(f"action_mean.shape: {action_mean.shape}")

#         if action_mean.ndim == 1:
#             action_mean = action_mean.unsqueeze(0)
#         action_logstd = self.actor_logstd.expand_as(action_mean)
#         # print(f"action_logstd.shape: {action_logstd.shape}")

#         action_std = torch.exp(action_logstd)
#         # print(f"action_std.shape: {action_std.shape}")

#         probs = Normal(action_mean, action_std)
#         if action is None:
#             action = probs.sample().squeeze()
        
#         action_log_prob = probs.log_prob(action).sum(1)
#         # print(f"action_log_prob.shape: {action_log_prob.shape}")

#         entropy = probs.entropy().sum(1)
#         # print(f"entropy.shape: {entropy.shape}")
        
#         critic_input = self.critic.preprocess_input(obs, train_mode=False)
#         q_value = self.critic(critic_input).squeeze()
        
#         return action, action_log_prob, entropy, q_value