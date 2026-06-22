#!/bin/bash

#SBATCH --job-name="GRC-4xRa8"             ## Name of the job.
#SBATCH --output="/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/RA1e8/4x_stretch/output_message.%j.out"             ##output file
#SBATCH --partition=gpuA40x4
#SBATCH --mem=36G
#SBATCH --nodes=1                 ## (-N) number of nodes to use
#SBATCH --ntasks-per-node=1                ## (-n) number of tasks to launch
#SBATCH --cpus-per-task=4
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1                  #this is 0 if cpu run
#SBATCH --gpu-bind=closest
#SBATCH --account=bfxn-delta-gpu         ## my account name #change to cpu or gpu depending 
#SBATCH --mail-user=ikeshwan@uci.edu
#SBATCH --mail-type="BEGIN,END"
#SBATCH -t 48:00:00
#SBATCH --signal=USR1@120

# Run the julia script and save julia's update messages to the file out.txt
module purge
module load cudatoolkit #remove for cpu run
# module load julia/1.10.10

cd /work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts
julia --project=../ job_scripts/run_3hill_GRC.jl