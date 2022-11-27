#!/bin/bash
#SBATCH --account=def-rjliao
#SBATCH --gres=gpu:v100l:4        # Number of GPUs per node (specifying v100l gpu)
#SBATCH --nodes=2                 # Number of nodes
#SBATCH --ntasks=1                # Number of MPI process
#SBATCH --ntasks-per-node=1       # Number of distributed process per compute node
#SBATCH --cpus-per-task=8         # CPU cores per MPI process
#SBATCH --mem=32G                 # memory per node
#SBATCH --time=00-00:20            # time (DD-HH:MM)
#SBATCH --mail-user=yanq@student.ubc.ca # send email regarding task status
#SBATCH --mail-type=ALL
 
################################################################################

# in this demo, we take 2 nodes and each node has 4 V100-32GB GPUs
MASTER_PORT=29400

module load gcc
module load cuda
module load nccl
module load openmpi

# you should submit job from the cloned repo's directory
cd ${SLURM_SUBMIT_DIR}

source venvhpc/bin/activate

# note: at Sockeye, it's better to specify CUDA_VISIBLE_DEVICES explicitly for distributed training,
# otherwise methods in torch.cuda may lead to an error, e.g., torch.cuda.device_count()

# single GPU: use 1 GPU on 1 node
CUDA_VISIBLE_DEVICES=0 python main.py --batch_size=768 -m=cc_demo_single_gpu

# DP: use multiple GPUs on 1 node
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --batch_size=3072 --dp -m=cc_demo_single_node_dp

# DDP: use multiple GPUs on 1 node
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT main.py --batch_size=3072 --ddp -m=cc_demo_single_node_ddp

# DDP: use multiple GPUs on multiple nodes

# mpirun method
#mpirun -np 8 \
#-x MASTER_ADDR=$(hostname) \
#-x MASTER_PORT=$MASTER_PORT \
#-x PATH \
#-bind-to none -map-by :OVERSUBSCRIBE \
#-mca pml ob1 -mca btl ^openib \
#python main.py --batch_size=6144 --ddp -m=cc_demo_multiple_node_ddp

# srun method
# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
srun CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=2 --nproc_per_node=4 \
--rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$(hostname) \
--master_port=$MASTER_PORT main.py --batch_size=3072 --ddp -m=cc_demo_single_node_ddp