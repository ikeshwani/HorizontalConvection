#!/bin/bash

#SBATCH --job-name="horizontal-convection"             ## Name of the job.
#SBATCH --output="output_message/HC.%j.out"             ##output file
#SBATCH --partition=gpuA40x4
#SBATCH --mem=36G
#SBATCH --nodes=1                 ## (-N) number of nodes to use
#SBATCH --ntasks-per-node=1                ## (-n) number of tasks to launch
#SBATCH --cpus-per-task=4
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bfxn-delta-gpu         ## my account name
#SBATCH --mail-user=ikeshwan@uci.edu
#SBATCH --mail-type="BEGIN,END"
#SBATCH -t 48:00:00

# Run the julia script and save julia's update messages to the file out.txt
module purge
module load cudatoolkit
# module load julia/1.10.10

cd /work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts
julia --project=../ run_thc.jl