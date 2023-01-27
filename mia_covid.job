#!/bin/bash

#SBATCH --partition=clara
#SBATCH --mail-type=FAIL,END
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
##SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --job-name=mia_covid
#SBATCH --output=./output/%A-mia_covid-%a.out
#SBATCH --time=2-00:00:00

# create './output' dir beforehand
# use: sbatch -a 1-16 mia_covid.job

# get correct run for array id
case $SLURM_ARRAY_TASK_ID in
  1)
    ds='covid'
    model='resnet18'
    eps='None'
    ;;
  2)
    ds='covid'
    model='resnet18'
    eps=10
    ;;
  3)
    ds='covid'
    model='resnet18'
    eps=1
    ;;
  4)
    ds='covid'
    model='resnet18'
    eps=0.1
    ;;
  5)
    ds='covid'
    model='resnet50'
    eps='None'
    ;;
  6)
    ds='covid'
    model='resnet50'
    eps=10
    ;;
  7)
    ds='covid'
    model='resnet50'
    eps=1
    ;;
  8)
    ds='covid'
    model='resnet50'
    eps=0.1
    ;;
  9)
    ds='mnist'
    model='resnet18'
    eps='None'
    ;;
  10)
    ds='mnist'
    model='resnet18'
    eps=10
    ;;
  11)
    ds='mnist'
    model='resnet18'
    eps=1
    ;;
  12)
    ds='mnist'
    model='resnet18'
    eps=0.1
    ;;
  13)
    ds='mnist'
    model='resnet50'
    eps='None'
    ;;
  14)
    ds='mnist'
    model='resnet50'
    eps=10
    ;;
  15)
    ds='mnist'
    model='resnet50'
    eps=1
    ;;
  16)
    ds='mnist'
    model='resnet50'
    eps=0.1
    ;;
  *) echo "This setting is not available."
    ;; 
esac

echo $ds $model $eps

# load packages
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1
source venv/mia-covid/bin/activate

# set environment variable for pythonpath to prefer venv packages and wandb settings
export $(grep -v '^#' .env | xargs)

# start run
srun python -m mia_covid -d $ds -m $model -e $eps -w 'True'
