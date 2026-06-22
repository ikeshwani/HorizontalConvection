#!/bin/bash
#SBATCH --job-name=psi_b_region_flux
#SBATCH --account=bfxn-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=output_message/psi_b_region_flux_%j.out
#SBATCH --error=output_message/psi_b_region_flux_%j.err

cd /work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts
julia --project=../ analysis_scripts/psi_b_region_flux.jl
