#!/bin/bash

IMAGE=public.ecr.aws#w6p6i9i7/aws-efa-nccl-rdma
TAG=base-cudnn8-cuda11-ubuntu20.04
CONTAINER=${IMAGE}:${TAG}

srun --comment $PROJECT \
    --partition=gpu \
    --nodes=1 \
    --gpus=8 \
    --cpus-per-gpu=6 \
    --container-image=${CONTAINER} \
        nvidia-smi
