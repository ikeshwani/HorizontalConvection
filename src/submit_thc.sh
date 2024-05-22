#!/bin/bash

#SBATCH --job-name=HC             ## Name of the job.
#SBATCH -p gpu
#SBATCH -A HFDRAKE_LAB_GPU
#SBATCH --gres=gpu:V100:1
#SBATCH --nodes=1                 ## (-N) number of nodes to use
#SBATCH --ntasks=1                ## (-n) number of tasks to launch
#SBATCH --mem=12G
#SBATCH --error=slurm-%J.err      ## error log file
#SBATCH --output=slurm-%J.out     ## output log file

# Run the julia script and save julia's update messages to the file out.txt
module load cuda/11.7.1
module load julia/1.9.3

cd /pub/hfdrake/code/HorizontalConvection/src
julia --project=../ run_thc.jl
