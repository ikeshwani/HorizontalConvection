#!/bin/bash
#SBATCH --job-name="gmixsortregion"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1                  #this is 0 if cpu run
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=4
#SBATCH --account=bfxn-delta-gpu
#SBATCH --mem=140G
#SBATCH --time=12:00:00
#SBATCH --output="analysis_scripts/output_message/gmix_sort_control%j.out"
#SBATCH --mail-user=ikeshwan@uci.edu
#SBATCH --mail-type="BEGIN,END"

module purge

cd /work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts
julia --project=../ analysis_scripts/G_mix_sort.jl
