#!/bin/bash
#SBATCH -J test
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH --time=48:00:00
#SBATCH --partition=t4v1,p100,t4v2,rtx6000
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=%x.%j.log
#SBATCH --gres=gpu:4


. /etc/profile.d/lmod.sh

echo Activating baiso_env
module load vector_nlp_project
source baiso_env/bin/activate

echo Starting model train
python train.py
