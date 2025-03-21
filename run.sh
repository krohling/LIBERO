echo "***Training LIBERO***"

echo "Retrieving dataset: libero_object"
python benchmark_scripts/download_libero_datasets.py --datasets libero_object 

export MUJOCO_GL=egl
echo "Training with Configuration: seed=0 benchmark_name=LIBERO_OBJECT policy=bc_transformer_policy lifelong=base"
xvfb-run -a python libero/lifelong/main.py seed=${SEED:-0} benchmark_name=${BENCHMARK_NAME:-LIBERO_OBJECT} policy=${POLICY:-bc_transformer_policy} lifelong=${LIFELONG:-base} train.n_epochs=${EPOCHS:-50} train.batch_size=${BATCH_SIZE:-128} train.optimizer.kwargs.lr=${LR:-0.0001}