#!/bin/bash
#SBATCH --job-name="psib_noGmix"
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfxn-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output="analysis_scripts/output_message/psib_noGmix_%j.out"
#SBATCH --mail-user=ikeshwan@uci.edu
#SBATCH --mail-type="BEGIN,END"

module purge

cd /work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts
julia --project=../ analysis_scripts/psi_b_regions_noGmix.jl
