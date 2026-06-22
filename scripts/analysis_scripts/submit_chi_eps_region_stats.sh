#!/bin/bash
#SBATCH --job-name="chi_eps_stats"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=bfxn-delta-cpu
#SBATCH --mem=140G
#SBATCH --time=02:00:00
#SBATCH --output="analysis_scripts/output_message/chi_eps_stats_%j.out"
#SBATCH --mail-user=ikeshwan@uci.edu
#SBATCH --mail-type="BEGIN,END"

module purge

cd /work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts
julia --project=../ analysis_scripts/chi_eps_region_stats.jl
