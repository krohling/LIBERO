#!/bin/bash

if [[ "$TRAIN" == "RL" ]]; then
    echo "***RL Training***"
    # apt-get update && apt-get install -y \
    #     libgles2-mesa-dev \
    #     libgl1-mesa-dev \
    #     libglu-dev \
    #     libglew-dev \
    #     mesa-utils

    # apt-get install -y libosmesa6-dev patchelf
    # pip install --no-cache-dir --force-reinstall mujoco
    echo "Retrieving dataset: libero_object"
    python benchmark_scripts/download_libero_datasets.py --datasets libero_object 

    echo "RL Training with Configuration: seed=${SEED:-1}, num_envs=${NUM_ENVS:-5}, num_steps=${NUM_STEPS:-600}, learning_rate=${LR:-0.00001}, total_timesteps=${TOTAL_TIMESTEPS:-1000000}, policy=${POLICY:-StochHeadLiberoAgent}, wandb_project_name=${WANDB_PROJECT:-libero_rl}, wandb_entity=${WANDB_ENTITY:-kevin_ai}"
    cd ppo
    python ppo_continuous_action_libero_transformer_policy.py --seed ${SEED:-1} --num_envs ${NUM_ENVS:-5} --num_steps ${NUM_STEPS:-600} --learning_rate ${LR:-0.00001} --total_timesteps ${TOTAL_TIMESTEPS:-1000000} --policy ${POLICY:-StochHeadLiberoAgent} --wandb_project_name ${WANDB_PROJECT:-libero_rl} --wandb_entity ${WANDB_ENTITY:-kevin_ai} --track --save_videos --save_model --upload_model 

    echo "***RL Training Complete***"
else
    # export MUJOCO_GL=egl
    echo "***Training LIBERO***"

    echo "Retrieving dataset: libero_object"
    python benchmark_scripts/download_libero_datasets.py --datasets libero_object 

    echo "Downgrading numpy to 1.22.4 workaround for Robomimic bug"
    pip install numpy==1.22.4

    echo "Training with Configuration: seed=${SEED:-0}, benchmark_name=${BENCHMARK_NAME:-LIBERO_OBJECT}, policy=${POLICY:-bc_transformer_policy}, lifelong=${LIFELONG:-base}, train.n_epochs=${EPOCHS:-50}, train.batch_size=${BATCH_SIZE:-128}, train.optimizer.kwargs.lr=${LR:-0.0001}"
    xvfb-run -a python libero/lifelong/main.py seed=${SEED:-0} benchmark_name=${BENCHMARK_NAME:-LIBERO_OBJECT} policy=${POLICY:-bc_transformer_policy} lifelong=${LIFELONG:-base} train.n_epochs=${EPOCHS:-50} train.batch_size=${BATCH_SIZE:-128} train.optimizer.kwargs.lr=${LR:-0.0001}

    echo "***Training LIBERO Complete***"
fi

# if [ -n "$RUNPOD_POD_ID" ]; then
#     echo "Terminating Pod"
#     runpodctl remove pod $RUNPOD_POD_ID
# fi