mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate


conda create -n libero python=3.8.13 -y
conda activate libero 

git clone https://github.com/krohling/LIBERO.git
cd LIBERO
git fetch --all
git checkout -b bc_to_rl origin/bc_to_rl

pip install -r requirements.txt
pip install -r ppo/requirements-ppo.txt 
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install --upgrade imageio imageio[ffmpeg] imageio-ffmpeg
pip install -e .

python benchmark_scripts/download_libero_datasets.py --datasets libero_object 

# Run BC Training
# pip install numpy==1.22.4
# python libero/lifelong/main.py seed=0 benchmark_name=LIBERO_OBJECT policy=bc_transformer_policy lifelong=base train.n_epochs=50 train.batch_size=128 train.optimizer.kwargs.lr=0.0001

# Run PPO Training
# cd ppo
# pip install numpy==1.24.4
# python ppo_continuous_action_libero_transformer_policy.py --num_envs 5 --num_steps 600 --learning_rate 0.00001