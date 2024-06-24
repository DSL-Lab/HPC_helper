# HPC_helper

This repository showcases a minimal example of using `PyTorch` distributed training on computing clusters, enabling you to run your training tasks on `N` nodes, each with `M` GPUs. It includes common use cases such as [DataParallel (DP)](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) or [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and offers support for [PBS](https://2020.help.altair.com/2020.1/PBSProfessional/PBSUserGuide2020.1.1.pdf) and [SLURM](https://slurm.schedmd.com/documentation.html) systems. Below, you'll find runnable code and scripts for UBC Sockeye, Vector Vaughan cluster, and Digital Research Alliance of Canada (formerly ComputeCanada) HPCs.

Last updated: Jun 23, 2024. Contact: Qi Yan, qi.yan@ece.ubc.ca 

## Get started

### Setup python environment
```bash
# load python 3.8 at HPC
# module load gcc/9.4.0 python/3.8.10 cuda/11.3.1 nccl/2.9.9-1-cuda11-3 # Sockeye
# module load python/3.8 cuda-11.7 # Vector
# module load gcc/11.3.0 python/3.8.10 StdEnv/2020 cuda/11.7 nccl/2.12.12 # CC

# python virtual environment
python -m venv venvhpc
source venvhpc/bin/activate
pip install -U pip
pip install -r setup/requirements_sockeye.txt   # if at Sockeye
pip install -r setup/requirements_cc.txt        # if at Vector or CC

# sanity check at Sockeye or CC
# you must enter an interactive session on Vector to tun this
python -c "import torch; print('Things are done.')"  

# download MNIST dataset
mkdir -p ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P ./mnist_data/MNIST/raw
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P ./mnist_data/MNIST/raw
```
On Alliance/CC clusters, you can only `pip install` python packages available on the system and `conda` is forbidden.
If you need to install additional packages, you can use the [`apptainer` container environment](https://docs.alliancecan.ca/wiki/Apptainer/en).
See the section below for details.

<details>
  <summary>apptainer instructions on Alliance/CC clusters</summary>
  The following instructions have been tested on the `narval` cluster. Similar steps work on other clusters like `cedar`, while the storage path may vary.

  ```bash
  ## pull image and create sandbox; recommended to do so at /scratch space for faster runtime
  module load apptainer-suid/1.1
  mkdir -p /lustre07/scratch/${USER}/venv && cd /lustre07/scratch/${USER}/venv
  apptainer pull --name pytorch220_official.sif docker://pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel
  apptainer build --sandbox venvhpc.sandbox pytorch220_official.sif

  ## get ready to enter the sandbox in an interactive shell
  export TMPDIR=/tmp/${USER}tmp
  mkdir -p ${TMPDIR}
  export APPTAINER_CACHEDIR=${TMPDIR}
  export APPTAINER_TMPDIR=${TMPDIR}

  ## bind the project, scratch, home directory to the sandbox; run `apptainer help run` to see meaning of each flag
  apptainer shell -C -B /project -B /scratch -B /home -W ${TMPDIR} --nv venvhpc.sandbox

  ## inside apptainer create conda env or use python venv; recommended to create conda env at /scratch space
  bash
  export USER=YOUR_USER_NAME  # change to your username
  conda create -p /lustre07/scratch/${USER}/venv/condaenvs/venvhpc python=3.8 -y
  conda activate /lustre07/scratch/${USER}/venv/condaenvs/venvhpc

  mkdir -p /lustre07/scratch/${USER}/venv/condaenvs/condacache
  conda config --add pkgs_dirs /lustre07/scratch/${USER}/venv/condaenvs/condacache

  ## pip install within the conda env
  pip install -U pip
  pip install -r setup/requirements_cc.txt

  ## sanity check
  python -c "import torch; print('Things are done.')"  

  ## follow the above "download MNIST dataset" section to load the dataset
  ```
  
  The apptainer sandbox is a containerized environment that allows you to install custom packages without root access. The `--bind` or `-B` flag is used to bind directories to the container. The sandbox itself contains only the necessary system libraries and the user's home directory. We still store the code and datasets on normal storage space.

</details>


### Go training
We showcase the use of distributed learning for a simple training task using ResNet50 as backbone.

**IMPORTANT**: 
* please change the account and notification email address in the bash script before running.
* the old Sockeye script is intended for OpenPBS system, which is no longer useful and kpet just for the sake of completeness.
* the Sockeye, Vector and CC scripts are intended for SLURM system, but we don't provide `preemption` support for Vector script.

```bash
# at Sockeye
sbatch scripts/demo_sockeye.sh

# at Vector
sbatch scripts/demo_vector.sh

# at CC
sbatch scripts/demo_cc.sh

# at CC with apptainer
## note: please change the paths in the script accordingly
sbatch scripts/demo_cc_apptainer.sh
```
Please check the training logs at `runs` for runtime comparison. Hear are five-epoch training time comparisons from my runs:

| #Nodes | #GPUs per node | PyTorch Distirbuted Method | Sockeye runtime | CC runtime                   | Vector runtime                    |
| ------ | -------------- | -------------------------- | --------------- | ---------------------------- | --------------------------------- |
| N=1    | M=1            | N/A                        | 363.4s          | 309.7s                       | 425.0s                            |
| N=1    | M=4            | DP                         | 103.5s          | 114.2s                       | 133.9s                            |
| N=1    | M=4            | DDP                        | 93.7s           | 85.2s                        | 113.4s                            |
| N=2    | M=4            | DDP                        | 55.7s           | 47.0s (mpirun); 47.4s (srun) | 60.9s (mpirun); 60.6s (srun)      |

In the demo script, we use Tesla V100-SXM2-32GB at Sockeye and CC, and RTX6000-24GB at Vector.
The single-precision performance in terms of FLOPS is 15.7 TFLOPS for V100-SXM2-32GB and 16.3 TFLOPS for RTX6000-24GB.
Therefore, the performance difference is mainly due to the GPU memory size.

## Distributed training rule of thumb

Generally, we could either use [DataParallel (DP)](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) or [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) protocol to start distributed training. DP is straightforward and only involves changes to a few lines of code. However, its efficiency is worse than DDP; please see [this page](https://pytorch.org/docs/stable/notes/cuda.html#use-nn-parallel-distributeddataparallel-instead-of-multiprocessing-or-nn-dataparallel) for why. Moreover, DP doesn't support multi-node distributed training. Therefore, it's better to always start with DDP despite its relatively higher complexity. The table belows shows the possible way to launch your distributed training jobs.


| #Nodes | #GPUs per node | PyTorch Distirbuted Method | Launch Method at Sockeye | Launch Method at CC |
|--------|----------------|----------------------------|---------------------------|----------------------|
| N=1    | M=1            | N/A                        | N/A                       | N/A                  |
| N=1    | M>1            | DDP or DP                 | torchrun                  | torchrun             |
| N>1    | M>1            | DDP                        | mpirun + python           | mpirun + python or srun + torchrun |


### Difference between PBS (old Sockeye) and SLURM (Vector and CC) systems
At PBS (old Sockeye) system, `mpirun + python` seems to be the only viable way to launch multi-node training. At SLURM (Vector and CC) system, we could use either `srun + torchrun` or `mpirun + python`. Essentially, both `mpirun` and `srun` are launching parallel jobs across different nodes *in one line of code*, and these two mechanisms are the key to scalable multi-node DDP training. We use the following example to show the crucial details to avoid errors.

**`mpirun + python` method explained**

Sample commands:
```bash
mpirun -np 8 \
--hostfile $PBS_NODEFILE --oversubscribe \
-x MASTER_ADDR=$(hostname) \
-x MASTER_PORT=$MASTER_PORT \
-x CUDA_VISIBLE_DEVICES=0,1,2,3 \
-x PATH \
-bind-to none -map-by :OVERSUBSCRIBE \
-mca pml ob1 -mca btl ^openib \
python main.py --batch_size=6144 --ddp -m=sockeye_demo_multiple_node_mpi_ddp
```
The `mpirun` is executed once, then the parallel jobs will be launched and their communications will be handled by PyTorch and `mpirun` altogether. The key is that we only need to **run `mpirun + python`  once on the master node**.

 `mpirun + python` comes with an option `-np` which specifies the number of processes in total. In our demo script, each process amounts to one trainer (i.e., one GPU), and we use `-np=8` for 2 nodes with 8 GPUs in total. This must be used along with `--oversubscribe`, and the reasons are as follows.

`mpirun` assigns job processes to nodes using [`slot`](https://www.open-mpi.org/doc/v4.0/man1/mpirun.1.php#sect3) scheduling, which was originally intended for CPU-only tasks due to historical reasons (one process amounts to one CPU core). However, such slot assignment may go wrong in the age of GPU training, as now we need to view one GPU as one process. For example, old Sockeye's PBS would not distribute 8 tasks equal to the 2 nodes and instead would raise an error indicating the number of available slots is insufficient. Therefore, we need to use the `--oversubscribe` option to enforce that `mpirun` does distribute tasks equally to each node and ignores the possible false alarm errors.

**`srun + torchrun` method explained**

Sample commands:

```bash
srun --ntasks-per-node=1 --ntasks=2 torchrun --nnodes=2 --nproc_per_node=4 \
--rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$(hostname):$MASTER_PORT \
main.py --batch_size=6144 --ddp -m=cc_demo_multiple_node_srun_ddp
```

The `SLURM_NTASKS` variable tells the script how many processes are available for this execution. `srun` executes the script `<tasks-per-node * nodes>` times. For `torchrun` launch method, we only need to **run it once per node**, and in our example, we are running `torchrun` commands twice on two nodes. Note that this is different than `mpirun + python`, where we *run it once for all nodes*.

For error-free srun execution, we need to pay attention to the `#SBATCH` options set in the very beginning or enforcing these parameters by using `--ntasks=2 --ntasks-per-node=1` explicitly. The nuance is `--ntasks=8 --ntasks-per-node=4` works for `mpirun + python` method, while `--ntasks=2 --ntasks-per-node=1` works for `srun + torchrun`.

## Adapt your code to distributed training
If you are okay with the PyTorch's built-in distributed training utilities, the plugin at `utils/dist_training.py` could be helpful. To change the code minimally for adaptation, please refer to the lines in `main.py` where `dist_helper` is called. 

Other third-party plugins like [horovod](https://horovod.ai/) and [pytorch lightning](https://www.pytorchlightning.ai/) can also possibly do the same things.



## Reference
#### Tutorial
* [Multi Node PyTorch Distributed Training Guide For People In A Hurry](https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide)
* [PyTorch with Multiple GPUs](https://docs.alliancecan.ca/wiki/PyTorch#PyTorch_with_Multiple_GPUs)
* [Multi-node-training on slurm with PyTorch](https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904)

#### Helpful documentations
* [pytorch torchrun](https://pytorch.org/docs/stable/elastic/run.html)
* [mpirun man page](https://www.open-mpi.org/doc/v4.0/man1/mpirun.1.php)
* [SLURM srun page](https://slurm.schedmd.com/srun.html)
* [SLURM sbatch environment variables](https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES)
* [PBS qsub environment variables](https://opus.nci.org.au/display/Help/Useful+PBS+Environment+Variables)

#### Wiki
* [UBC Sockeye](https://confluence.it.ubc.ca/display/UARC/About+Sockeye)
* [Vector](https://support.vectorinstitute.ai/FrontPage)
* [CC](https://docs.alliancecan.ca/wiki/Technical_documentation)