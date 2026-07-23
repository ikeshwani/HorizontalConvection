#!/bin/bash
#SBATCH --job-name="ke-hill"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=bfxn-delta-cpu
#SBATCH --mem=140G
#SBATCH --time=08:00:00
#SBATCH --output="analysis_scripts/output_message/ke_control_%j.out"
#SBATCH --mail-user=ikeshwan@uci.edu
#SBATCH --mail-type="BEGIN,END"

# KE analysis (MKE/TKE/KE) for the flat (control) GRC run, segments 1:15.
# kinetic_energetics.jl is set to experiment = "control" at the top of the file.
# Wall time is dominated by reading the large velocity_seg*.nc files from disk.

module purge

cd /work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts
julia --project=../ analysis_scripts/kinetic_energetics.jl
