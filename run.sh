echo "***Training LIBERO***"

echo "Retrieving dataset: libero_object"
python benchmark_scripts/download_libero_datasets.py --datasets libero_object 

export MUJOCO_GL=egl
echo "Training with Configuration: seed=0 benchmark_name=LIBERO_OBJECT policy=bc_transformer_policy lifelong=base"
xvfb-run -a python libero/lifelong/main.py seed=0 benchmark_name=LIBERO_OBJECT policy=bc_transformer_policy lifelong=base

# echo "Training with Configuration: seed=0 benchmark_name=LIBERO_OBJECT policy=bc_transformer_policy lifelong=er"
# xvfb-run -a python libero/lifelong/main.py seed=0 benchmark_name=LIBERO_OBJECT policy=bc_transformer_policy lifelong=er

# echo "Training with Configuration: seed=0 benchmark_name=LIBERO_OBJECT policy=bc_transformer_policy lifelong=ewc"
# xvfb-run -a python libero/lifelong/main.py seed=0 benchmark_name=LIBERO_OBJECT policy=bc_transformer_policy lifelong=ewc

# echo "Training with Configuration: seed=0 benchmark_name=LIBERO_OBJECT policy=bc_transformer_policy lifelong=packnet"
# xvfb-run -a python libero/lifelong/main.py seed=0 benchmark_name=LIBERO_OBJECT policy=bc_transformer_policy lifelong=packnet

# echo "Training with Configuration: seed=0 benchmark_name=LIBERO_OBJECT policy=bc_transformer_policy lifelong=multitask"
# xvfb-run -a python libero/lifelong/main.py seed=0 benchmark_name=LIBERO_OBJECT policy=bc_transformer_policy lifelong=multitask

# echo "Running evaluation for task 0 with EWC"
# python libero/lifelong/evaluate.py --benchmark libero_object \
#                                    --task_id 0 \
#                                    --algo ewc \
#                                    --policy bc_transformer_policy \
#                                    --seed 0 \
#                                    --load_task 9 \
#                                    --device_id 0 --save-videos

# echo "***Training Complete***"
# if [ -n "$RUNPOD_POD_ID" ]; then
#     echo "Terminating Pod"
#     runpodctl remove pod $RUNPOD_POD_ID
# fi