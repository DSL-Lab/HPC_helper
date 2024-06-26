#!/bin/bash
#SBATCH --job-name=demo_sockeye
#SBATCH --account=st-rjliao-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm-%j_out.txt
#SBATCH --error=slurm-%j_err.txt
#SBATCH --mail-user=yanq@student.ubc.ca
#SBATCH --mail-type=ALL
################################################################################

# in this demo, we take 2 nodes and each node has 4 V100-32GB GPUs
MASTER_PORT=29400

module load gcc
module load cuda
module load nccl
module load openmpi

# you should submit job from the cloned repo's directory
cd ${PBS_O_WORKDIR}
source venvhpc/bin/activate
export OMP_NUM_THREADS=6

# note: at Sockeye, it's better to specify CUDA_VISIBLE_DEVICES explicitly for distributed training,
# otherwise methods in torch.cuda may lead to an error, e.g., torch.cuda.device_count()

# single GPU: use 1 GPU on 1 node
CUDA_VISIBLE_DEVICES=0 python main.py --batch_size=768 -m=sockeye_demo_single_gpu

# DP: use multiple GPUs on 1 node
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --batch_size=3072 --dp -m=sockeye_demo_single_node_dp

# DDP: use multiple GPUs on 1 node
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT main.py --batch_size=3072 --ddp -m=sockeye_demo_single_node_ddp

# DDP: use multiple GPUs on multiple nodes
mpirun -np 8 \
--hostfile $PBS_NODEFILE --oversubscribe \
-x MASTER_ADDR=$(hostname) \
-x MASTER_PORT=$MASTER_PORT \
-x CUDA_VISIBLE_DEVICES=0,1,2,3 \
-x PATH \
-bind-to none -map-by :OVERSUBSCRIBE \
-mca pml ob1 -mca btl ^openib \
python main.py --batch_size=6144 --ddp -m=sockeye_demo_multiple_node_mpi_ddp

