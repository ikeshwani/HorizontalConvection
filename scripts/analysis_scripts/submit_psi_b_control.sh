#!/bin/bash
#SBATCH --job-name="psi_b_ctrl"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=bfxn-delta-cpu
#SBATCH --mem=140G
#SBATCH --time=04:00:00
#SBATCH --output="scripts/analysis_scripts/output_message/psi_b_control_%j.out"
#SBATCH --mail-user=ikeshwan@uci.edu
#SBATCH --mail-type="BEGIN,END"

module purge

cd /work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts
julia --project=../ analysis_scripts/psi_of_b_sort.jl
