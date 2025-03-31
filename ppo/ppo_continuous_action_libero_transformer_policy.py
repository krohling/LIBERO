# python ppo_continuous_action_libero_transformer_policy.py --num_envs 5 --num_steps 600 --save_videos --track

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import json
import random
import time
from dataclasses import dataclass
from easydict import EasyDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.venv import SubprocVectorEnv, DummyVectorEnv
from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.utils import torch_load_model, get_task_embs
from libero.libero.utils.video_utils import VideoWriter
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


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_videos: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    video_folder: str = './videos'
    """the path to save videos"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to wandb"""

    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel environments for trajectory collection"""
    num_eval_envs: int = 5
    """the number of parallel environments for evaluation"""
    eval_frequency: int = 1
    """the number of iterations to run evaluation"""
    num_steps: int = 600
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    kl_loss_coef: float = 0.0
    """the coefficient of the KL loss"""

    libero_task_suite: str = 'libero_object'
    """the LIBERO task suite"""
    libero_task_id: int = 0
    """the LIBERO task id"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    checkpoint_path: str = None
    """the checkpoint to use for the actor network"""
    pretrain_value_iters: int = 0
    """the number of pretraining iterations for the value network"""
    reset_envs: bool = False
    """whether to reset the 'done' environments during trajectory collection"""
    critic_type: str = 'transformer'
    """the type of critic to use"""


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

    env = DummyVectorEnv(
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

    observation_space_dim = np.prod(obs_tensor[0].shape)
    env.task_emb = task_emb
    env.single_observation_space = torch.zeros(observation_space_dim)
    env.single_action_space = torch.zeros(7)

    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class StochLiberoAgent(nn.Module):
    def __init__(self, envs, checkpoint_path=None, critic_type='transformer'):
        super().__init__()
        self.task_emb = envs.task_emb
        self.critic_type = critic_type

        if critic_type == 'mlp':
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
        else:
            self.critic = make_policy('./critic_config.json')

        self.actor = make_policy('./actor_stoch_config.json')
        if checkpoint_path:
            print(f"Loading checkpoint: {checkpoint_path}")
            self.actor.load_state_dict(torch_load_model(checkpoint_path, 'cpu')[0])

    def get_value(self, obs):
        if self.critic_type == 'mlp':
            obs_tensor = obs_to_tensor(obs)
            q_value = self.critic(obs_tensor).squeeze()
        else:
            obs = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)
            critic_input = self.critic.preprocess_input(obs, train_mode=False)
            q_value = self.critic(critic_input).squeeze()

        return q_value

    def get_action_and_value(self, obs, action=None):
        obs_dict = raw_obs_to_tensor_obs(obs, self.task_emb, MODALITY_CONFIG)

        actor_input = self.actor.preprocess_input(obs_dict, train_mode=False)
        probs = self.actor(actor_input)
        if action is None:
            action = probs.sample()
            action = action.squeeze()
        
        action_log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)

        if self.critic_type == 'mlp':
            obs_tensor = obs_to_tensor(obs)
            q_value = self.critic(obs_tensor).squeeze()
        else:
            critic_input = self.critic.preprocess_input(obs_dict, train_mode=False)
            q_value = self.critic(critic_input).squeeze()
        
        return action, action_log_prob, entropy, q_value


def perform_evaluation(libero_agent, global_step, args):
    envs = make_libero_envs(args.num_eval_envs, args.libero_task_suite, args.libero_task_id, args.num_steps)
    next_obs = envs.reset()

    rewards = torch.zeros((args.num_steps, args.num_eval_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_eval_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_eval_envs)).to(device)

    video_writer = VideoWriter(args.video_folder, args.save_videos)
    next_done = torch.zeros(args.num_eval_envs).to(device)

    for step in range(0, args.num_steps):
        print(f"step: {step}")
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, _, _, value = libero_agent.get_action_and_value(next_obs)
            values[step] = value.flatten()

        next_obs, reward, next_done, _ = envs.step(action.cpu().numpy())

        video_writer.append_vector_obs(
            next_obs, next_done, camera_name="agentview_image"
        )
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_done = torch.Tensor(next_done).to(device)

        if next_done.sum() == args.num_envs:
            break
    
    envs.close()
    
    video_writer.save()
    if args.track:
        wandb.save(f"{args.video_folder}/video.mp4")
    
    with torch.no_grad():
        next_value = libero_agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
    
    success_rate = next_done.sum() / args.num_envs
    cum_rewards = rewards.sum(dim=0)
    max_cum_reward = cum_rewards.max()
    avg_cum_reward = cum_rewards.mean()
    avg_cum_returns = returns.sum(dim=0).mean()

    writer.add_scalar("charts/validation_success_rate", success_rate, global_step)
    writer.add_scalar("charts/validation_max_reward", max_cum_reward, global_step)
    writer.add_scalar("charts/validation_avg_reward", avg_cum_reward, global_step)
    writer.add_scalar("charts/validation_avg_return", avg_cum_returns, global_step)


    print(f"success_rate: {success_rate}")
    print(f"max_reward: {max_cum_reward}")
    print(f"avg_reward: {avg_cum_reward}")
    print(f"avg_return: {avg_cum_returns}")


def set_freeze_status(model, frozen=False):
    for param in model.parameters():
        param.requires_grad = not frozen


if __name__ == "__main__":
    # policy = make_policy('./actor_config.json')

    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.libero_task_suite}_{args.libero_task_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    MODALITY_CONFIG['device'] = str(device)

    # env setup
    envs = make_libero_envs(args.num_envs, args.libero_task_suite, args.libero_task_id, args.num_steps)
    libero_agent = StochLiberoAgent(
        envs, 
        checkpoint_path=args.checkpoint_path,
        critic_type=args.critic_type
    ).to(device)

    frozen_agent = StochLiberoAgent(
        envs, 
        checkpoint_path=args.checkpoint_path,
        critic_type=args.critic_type
    ).to(device)
    set_freeze_status(frozen_agent, frozen=True)
    
    optimizer = optim.Adam(libero_agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = np.empty((args.num_steps, args.num_envs), dtype=object)
    # obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # next_obs, _ = envs.reset(seed=args.seed)
    envs.reset()
    
    dummy_action = [[0.]*7] * args.num_envs
    next_obs, reward, done, info = envs.step(dummy_action)
    next_done = torch.zeros(args.num_envs).to(device)

    print(f"Running for {args.num_iterations} iterations")
    
    for iteration in range(1, args.num_iterations + 1):
        print(f"iteration: {iteration}")
        iteration_start = time.time()
        envs.reset()

        if iteration <= args.pretrain_value_iters:
            print("Freezing actor parameters. Only training value network for this iteration.")
            set_freeze_status(libero_agent.actor, frozen=True)
        else:
            set_freeze_status(libero_agent.actor, frozen=False)

        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            print(f"step: {step}")

            done_env_ids = np.nonzero(next_done.cpu().numpy())[0]

            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = libero_agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, next_done, infos = envs.step(action.cpu().numpy())

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = torch.Tensor(next_done).to(device)

            if len(done_env_ids) > 0 and args.reset_envs:
                print(f"Resetting done envs: {done_env_ids}")
                next_obs[done_env_ids] = envs.reset(done_env_ids)
                next_done[done_env_ids] = torch.zeros(len(done_env_ids)).to(device)

        if iteration % args.eval_frequency == 0:
            perform_evaluation(libero_agent, global_step, args)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = libero_agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = libero_agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # KL - Divergence loss
                _, bc_logprob, _, _ = frozen_agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                kl_loss = torch.nn.functional.kl_div(
                    input=bc_logprob, 
                    target=newlogprob,
                    reduction='batchmean', 
                    log_target=True
                )
                # kl_loss = (newlogprob - bc_logprob).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - (args.ent_coef * entropy_loss) + (v_loss * args.vf_coef) + (kl_loss * args.kl_loss_coef)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(libero_agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        print(f"iteration {iteration} took {time.time() - iteration_start:.2f} seconds")


    envs.close()
    writer.close()
