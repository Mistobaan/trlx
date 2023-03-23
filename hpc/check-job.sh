#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=nccl-tests
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --mem=64GB
#SBATCH --output=%x_%j.out
#SBATCH –-comment=trlx-mistobaan
module load openmpi
module load cuda/11.3 # match the cuda in the container image

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_DEBUG=warn
export NCCL_PROTO=simple
export NCCL_TREE_THRESHOLD=0
export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl="^openib"
export SINGULARITY_OMPI_DIR=/opt/amazon/openmpi
export SINGULARITYENV_APPEND_PATH=/opt/amazon/openmpi/bin
export SINGULAIRTYENV_APPEND_LD_LIBRARY_PATH=/opt/amazon/openmpi/lib

mpirun -n 16 \
	singularity exec --nv \
		docker://public.ecr.aws/w6p6i9i7/aws-efa-nccl-rdma:base-cudnn8-cuda11.3-ubuntu20.04 \

 /opt/nccl-tests/build/all_reduce_perf \
	-b 128M \
	-e 8G \
	-f 2 \
	-g 1 \
	-c 1 \
	-n 20