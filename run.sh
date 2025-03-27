#!/bin/bash

export MUJOCO_GL=egl
echo "Retrieving dataset: libero_object"
python benchmark_scripts/download_libero_datasets.py --datasets libero_object 

if [[ "$TRAIN" == "RL" ]]; then
    echo "RL Training with Configuration: seed=${SEED:-1}, num_envs=${NUM_ENVS:-5}, num_steps=${NUM_STEPS:-600}, learning_rate=${LR:-0.00001}, total_timesteps=${TOTAL_TIMESTEPS:-1000000}, wandb_project_name=${WANDB_PROJECT:-libero_rl}, wandb_entity=${WANDB_ENTITY:-kevin_ai}"
    cd ppo
    xvfb-run -a python ppo_continuous_action_libero_transformer_policy.py --seed ${SEED:-1} --num_envs ${NUM_ENVS:-5} --num_steps ${NUM_STEPS:-600} --learning_rate ${LR:-0.00001} --total_timesteps ${TOTAL_TIMESTEPS:-1000000} --wandb_project_name ${WANDB_PROJECT:-libero_rl} --wandb_entity ${WANDB_ENTITY:-kevin_ai} --track --save_videos --save_model --upload_model 
    echo "***RL Training Complete***"
else
    echo "***Training LIBERO***"
    echo "Downgrading numpy to 1.22.4 workaround for Robomimic compatibility"
    pip install numpy==1.22.4
    echo "Training with Configuration: seed=${SEED:-0}, benchmark_name=${BENCHMARK_NAME:-LIBERO_OBJECT}, policy=${POLICY:-bc_transformer_policy}, lifelong=${LIFELONG:-base}, train.n_epochs=${EPOCHS:-50}, train.batch_size=${BATCH_SIZE:-128}, train.optimizer.kwargs.lr=${LR:-0.0001}"
    xvfb-run -a python libero/lifelong/main.py seed=${SEED:-0} benchmark_name=${BENCHMARK_NAME:-LIBERO_OBJECT} policy=${POLICY:-bc_transformer_policy} lifelong=${LIFELONG:-base} train.n_epochs=${EPOCHS:-50} train.batch_size=${BATCH_SIZE:-128} train.optimizer.kwargs.lr=${LR:-0.0001}

    echo "***Training LIBERO Complete***"
fi