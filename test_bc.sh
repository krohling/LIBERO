# If running fails due to numpy version, you may need to change the installed numpy version
# pip install numpy==1.22.4

export USE_STOCH_HEAD=True # Set to True to use StochPolicyHead for BC training
python libero/lifelong/main.py seed=0 benchmark_name=libero_object policy=bc_transformer_policy lifelong=base train.n_epochs=50 train.batch_size=2 train.optimizer.kwargs.lr=0.0001