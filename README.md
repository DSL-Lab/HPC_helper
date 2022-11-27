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
pip install -r setup/requirements_sockeye.txt   # if at Sockeye
pip install -r setup/requirements_cc.txt        # if at CC

# download MNIST dataset
mkdir -p ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P ./mnist_data/MNIST/raw
```
## Go training
**IMPORTANT**: please change the account and notification email address in the bash script before running.

```bash
# at Sockeye
qsub scripts/demo_sockeye.sh

# at CC
sbatch scripts/demo_cc.sh
```
Please check the training logs at `runs` for runtime comparison.

## GPU profiling (*to-be-updated*)
```bash
python helper/benchmark_layernorm.py
```

## Reference
#### Tutorial
* [Multi Node PyTorch Distributed Training Guide For People In A Hurry](https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide)
* [PyTorch with Multiple GPUs](https://docs.alliancecan.ca/wiki/PyTorch#PyTorch_with_Multiple_GPUs)

#### Helpful documentations
* [pytorch torchrun](https://pytorch.org/docs/stable/elastic/run.html)
* [mpirun man page](https://www.open-mpi.org/doc/v4.0/man1/mpirun.1.php)
* [SLURM srun page](https://slurm.schedmd.com/srun.html)
* [SLURM sbatch environment variables](https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES)
* [PBS qsub environment variables](https://opus.nci.org.au/display/Help/Useful+PBS+Environment+Variables)
* 