#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=simsiam
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --output=/home/nberardo/simsiam/train.log
#SBATCH --error=/home/nberardo/simsiam/train.err
###### 1 Load the module
module load nvidia/cudasdk
module load intel/python/3
# 2 Activate the virtual environment
source /home/nberardo/myenv/bin/activate
# 3 Run the python script
python extract_backbone.py

#python main_lincls.py \
#  -a resnet50 \
#  --batch-size 512 \
#  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
#  --pretrained /home/nberardo/simsiam/downloaded/checkpoint_0099.pth.tar \
#  --resume checkpoint.pth.tar \
#  --lars \
#  /home/eaiello/ImageNet