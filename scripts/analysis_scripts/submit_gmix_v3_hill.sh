#!/bin/bash
#SBATCH --job-name="gmix_hill_CODF"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1                  #this is 0 if cpu run
#SBATCH --account=bfxn-delta-gpu
#SBATCH --mem=140G
#SBATCH --time=12:00:00
#SBATCH --output="/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_threehill_baseforcing_zerostart/logs/gmix_hill_CODF_%j.out"
#SBATCH --mail-user=ikeshwan@uci.edu
#SBATCH --mail-type="BEGIN,END"


module purge

# export GMIX_EXPERIMENT=hill
# export GMIX_RA=1e8

cd /work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts
julia --project=../ analysis_scripts/gmix_CODF.jl
