# HPC_helper

## Get started
```bash
# load python 3.7 at HPC
# module load gcc python/3.7.9 StdEnv/2020 cudacore/.11.4.2 nccl/2.11.4 # CC
# module load gcc/7.5.0 python/3.7 nccl # Sockeye

# python venv
python -m venv venvhpc
source venvhpc/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

# download MNIST dataset
mkdir -p ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P ./mnist_data/MNIST/raw
```
## Go benchmarking
```bash
qsub run_benchmark.pbs
```