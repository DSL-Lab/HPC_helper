#!/bin/bash
#SBATCH --account=def-rjliao
#SBATCH --gres=gpu:v100l:4        # Number of GPUs per node (specifying v100l gpu)
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=8                # Number of MPI process
#SBATCH --ntasks-per-node=4       # Number of distributed process per compute node
#SBATCH --cpus-per-task=8         # CPU cores per MPI process
#SBATCH --mem=64G                 # memory per node
#SBATCH --time=00-00:20            # time (DD-HH:MM)
#SBATCH --mail-user=yanq@student.ubc.ca # send email regarding task status
#SBATCH --mail-type=ALL
 
################################################################################

# in this demo, we take 1 nodes and each node has 4 V100-32GB GPUs

## set up environment variables for apptainer
## this script is intended for the narval cluster, please adjust the path accordingly for other clusters
module load apptainer-suid/1.1
cd /lustre07/scratch/${USER}/venv
export TMPDIR=/tmp/${USER}tmp
mkdir -p ${TMPDIR}
export APPTAINER_CACHEDIR=${TMPDIR}
export APPTAINER_TMPDIR=${TMPDIR}

# !!!please change the USER_NAME to your own username before running the script!!!

# single GPU: use 1 GPU on 1 node
apptainer exec -C -B /project -B /scratch -B /home -W ${TMPDIR} --nv venvhpc.sandbox bash -c '
export USER_NAME='YOUR_USER_NAME'
source /opt/conda/etc/profile.d/conda.sh
conda activate /lustre07/scratch/${USER_NAME}/venv/condaenvs/venvhpc
cd /lustre06/project/6068146/${USER_NAME}/HPC_helper
CUDA_VISIBLE_DEVICES=0 python main.py --batch_size=768 -m=cc_demo_single_gpu
'

# single GPU: use 1 GPU on 1 node
apptainer exec -C -B /project -B /scratch -B /home -W ${TMPDIR} --nv venvhpc.sandbox bash -c '
export USER_NAME='YOUR_USER_NAME'
source /opt/conda/etc/profile.d/conda.sh
conda activate /lustre07/scratch/${USER_NAME}/venv/condaenvs/venvhpc
cd /lustre06/project/6068146/${USER_NAME}/HPC_helper
CUDA_VISIBLE_DEVICES=0 python main.py --batch_size=768 -m=cc_apptainer_demo_single_gpu
'

# DP: use multiple GPUs on 1 node
apptainer exec -C -B /project -B /scratch -B /home -W ${TMPDIR} --nv venvhpc.sandbox bash -c '
export USER_NAME='YOUR_USER_NAME'
source /opt/conda/etc/profile.d/conda.sh
conda activate /lustre07/scratch/${USER_NAME}/venv/condaenvs/venvhpc
cd /lustre06/project/6068146/${USER_NAME}/HPC_helper
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --batch_size=3072 --dp -m=cc_apptainer_demo_single_node_dp
'

# DDP: use multiple GPUs on 1 node
apptainer exec -C -B /project -B /scratch -B /home -W ${TMPDIR} --nv venvhpc.sandbox bash -c '
export USER_NAME='YOUR_USER_NAME'
export MASTER_PORT=29400
source /opt/conda/etc/profile.d/conda.sh
conda activate /lustre07/scratch/${USER_NAME}/venv/condaenvs/venvhpc
cd /lustre06/project/6068146/${USER_NAME}/HPC_helper
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT main.py --batch_size=3072 --ddp -m=cc_apptainer_demo_single_node_ddp
'

# note: we haven't tested the multi-node DDP with apptainer yet, but the MPI option may work
