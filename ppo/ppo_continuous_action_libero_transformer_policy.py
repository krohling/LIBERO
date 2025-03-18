# python ppo_continuous_action_libero_transformer_policy.py --num_envs 5 --num_steps 600 --save_videos --track

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from libero.libero.utils.video_utils import VideoWriter

from policies import *


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
    wandb_project_name: str = "libero_rl"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_videos: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    video_folder: str = './videos'
    """the path to save videos"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = True
    """whether to upload the saved model to wandb"""

    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 600
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
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

    policy: str = "LiberoAgent"



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

    if args.policy == 'StochHeadLiberoAgent':
        print("Using StochHeadLiberoAgent")
        libero_agent = StochHeadLiberoAgent(envs).to(device)
    elif args.policy == 'DetHeadLiberoAgent':
        print("Using DetHeadLiberoAgent")
        libero_agent = DetHeadLiberoAgent(envs).to(device)
    elif args.policy == 'GMMHeadLiberoAgent':
        print("Using GMMHeadLiberoAgent")
        libero_agent = GMMHeadLiberoAgent(envs).to(device)
    else:
        libero_agent = LiberoAgent(envs).to(device)

    print(f"learning_rate: {args.learning_rate}")

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

    print(f"Running for {args.num_iterations} iterations")
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        print(f"iteration: {iteration}")
        iteration_start = time.time()
        next_obs = envs.reset()
        next_done = np.zeros(args.num_envs)

        video_dir = f"{args.video_folder}/{iteration}"
        video_writer = VideoWriter(video_dir, args.save_videos)
        
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        success_count = 0
        for step in range(0, args.num_steps):
            print(f"step: {step}")

            done_env_ids = np.nonzero(next_done)[0]
            if len(done_env_ids) > 0:
                success_count += len(done_env_ids)
                print("*************************")
                print(f"Resetting done envs: {done_env_ids}")
                next_obs[done_env_ids] = envs.reset(done_env_ids)
                next_done[done_env_ids] = torch.zeros(len(done_env_ids))
                dones[step-1][done_env_ids] = torch.ones(len(done_env_ids))

            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = torch.tensor(next_done).to(device).view(-1)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = libero_agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

            next_obs, reward, next_done, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            video_writer.append_vector_obs(
                next_obs, next_done, camera_name="agentview_image"
            )
            

        
        if args.save_videos:
            video_writer.save()
            video_filename = f"{video_dir}/video.mp4"
            print(f"Saved video to {video_filename}")
            if args.track:
                wandb.save(video_filename)

        
        # success_rate = next_done.sum() / args.num_envs
        cum_rewards = rewards.sum(dim=0)
        max_cum_reward = cum_rewards.max()
        avg_cum_reward = cum_rewards.mean()
        # print(f"success_rate: {success_rate}")
        print(f"success_count: {success_count}")
        print(f"max_reward: {max_cum_reward}")
        print(f"avg_reward: {avg_cum_reward}")
        # writer.add_scalar("charts/episodic_success_rate", success_rate, global_step)
        writer.add_scalar("charts/episodic_success_count", success_count, global_step)
        writer.add_scalar("charts/episodic_max_return", max_cum_reward, global_step)
        writer.add_scalar("charts/episodic_avg_return", avg_cum_reward, global_step)
        
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
                
                nextnonterminal = torch.tensor(nextnonterminal).to(device)
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
                # _, newlogprob, entropy, newvalue = libero_agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # print(f"pg_loss: {pg_loss}")
                # print(f"v_loss: {v_loss}")
                # loss = pg_loss + v_loss * args.vf_coef

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

        if args.save_model:
            model_path = f"runs/{run_name}/{args.policy}_{iteration}.pth"
            torch.save(libero_agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
            if args.track:
                wandb.save(model_path)

    #     from cleanrl_utils.evals.ppo_eval import evaluate

    #     episodic_returns = evaluate(
    #         model_path,
    #         make_env,
    #         args.env_id,
    #         eval_episodes=10,
    #         run_name=f"{run_name}-eval",
    #         Model=Agent,
    #         device=device,
    #         gamma=args.gamma,
    #     )
    #     for idx, episodic_return in enumerate(episodic_returns):
    #         writer.add_scalar("eval/episodic_return", episodic_return, idx)

    #     if args.upload_model:
    #         from cleanrl_utils.huggingface import push_to_hub

    #         repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
    #         repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
    #         push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
