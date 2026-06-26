#!/usr/bin/env bash
# migrate_output_layout.sh
#
# Migrate ONE simulation run from the old nested output layout
#   output/GPU/GRC/<...>/<Ra>/<stretch>/<Nx>_<Nz>/
# to the new self-contained flat layout
#   output/GPU/GRC/<run_name>/
# This:
#   1. rename-moves the leaf dir to the new run-name folder (same-fs mv = instant)
#   2. renames the checkpoint file(s) from the old generate_filename prefix to the
#      new one, so segment pickups under the new code still resume correctly
#   3. creates an empty figures/ subfolder
# The per-run config.toml is written separately (Julia write_run_config).
#
# IMPORTANT: never run this on a directory a SLURM job is actively writing to.
#
# Usage:
#   migrate_output_layout.sh <old_leaf_dir> <new_run_dir> <old_ckpt_prefix> <new_ckpt_prefix>

set -euo pipefail

old_dir="$1"; new_dir="$2"; old_prefix="$3"; new_prefix="$4"

[ -d "$old_dir" ] || { echo "ERROR: source '$old_dir' not found"; exit 1; }
[ -e "$new_dir" ] && { echo "ERROR: destination '$new_dir' already exists"; exit 1; }

# 1. same-filesystem rename-move (no data copy)
mkdir -p "$(dirname "$new_dir")"
mv "$old_dir" "$new_dir"
echo "moved:      $old_dir  ->  $new_dir"

# 2. rename checkpoint(s): old prefix -> new prefix
shopt -s nullglob
found_ckpt=0
for f in "$new_dir/${old_prefix}_checkpoint"*; do
    found_ckpt=1
    base="$(basename "$f")"
    newbase="${base/${old_prefix}/${new_prefix}}"
    mv "$f" "$new_dir/$newbase"
    echo "checkpoint: $base  ->  $newbase"
done
[ "$found_ckpt" -eq 0 ] && echo "WARN: no checkpoint matching '${old_prefix}_checkpoint*' found"

# 3. empty figures/ subfolder
mkdir -p "$new_dir/figures"
echo "created:    $new_dir/figures"

echo "DONE: $new_dir  (now write config.toml)"
