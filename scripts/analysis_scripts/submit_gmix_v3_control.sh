#!/bin/bash
#SBATCH --job-name="gmix_8fl_quantile_CODF"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=bfxn-delta-cpu
#SBATCH --mem=140G
#SBATCH --time=12:00:00
#SBATCH --output="/work/hdd/bfxn/ikeshwani/HorizontalConvection/output/GPU/GRC/ra1e8_4xstretch_flat_baseforcing_zerostart/logs/gmix_control_CODF_%j.out"
#SBATCH --mail-user=ikeshwan@uci.edu
#SBATCH --mail-type="BEGIN,END"

module purge

# export GMIX_EXPERIMENT=control
# export GMIX_RA=1e8

cd /work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts
julia --project=../ analysis_scripts/gmix_CODF.jl
