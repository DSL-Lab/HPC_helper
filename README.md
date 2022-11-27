# HPC_helper

## Get started
### Setup python environment
```bash
# load python 3.8 at HPC
# module load gcc/11.3.0 python/3.8.10 StdEnv/2020 cuda/11.7 nccl/2.12.12 # CC
# module load gcc/9.4.0 python/3.8.10 cuda/11.3.1 nccl/2.9.9-1-cuda11-3 # Sockeye

# python virtual environment
python -m venv venvhpc
source venvhpc/bin/activate
pip install -U pip
pip install -r setup/requirements.txt

# download MNIST dataset
mkdir -p ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P ./mnist_data/MNIST/raw
```
## Go benchmarking
```bash
# at Sockeye
qsub scripts/demo_sockeye.sh
```

## GPU profiling (*to-be-updated*)
```bash
python helper/benchmark_layernorm.py
```