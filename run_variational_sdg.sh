#!/bin/bash

# Example usage:
# bash run_sdg.sh data/usdb.nat 8 2 600 0.01 200 10 300 1000

FILE_NAME=${1:-data/usdb.nat}
NPARTICLES_A=${2:-8}
NPARTICLES_B=${3:-2}
NUM_STEPS=${4:-600}
LR=${5:-0.01}
SAMPLES=${6:-200}
TRAIN_STEPS=${7:-10}
NUM_PARAMETERS=${8:-300}
BATCHES=${9:-1000}
SAVE_FILE_NAME=${1:-data/sdg/usdb_8_2_200_samples_10_steps_300_parameters_1000_batches}
# Run with nohup so process survives even after logout
nohup python variational_sdg_nsm.py \
    --file_name "$FILE_NAME" \
    --nparticles_a "$NPARTICLES_A" \
    --nparticles_b "$NPARTICLES_B" \
    --num_steps "$NUM_STEPS" \
    --lr "$LR" \
    --samples "$SAMPLES" \
    --train_steps "$TRAIN_STEPS" \
    --num_parameters "$NUM_PARAMETERS" \
    --batches "$BATCHES" \
    --save_file_name "$SAVE_FILE_NAME" \
    > sdg_run.log 2>&1 &
    
echo "Simulation started with PID $!"
echo "Logs are being written to sdg_run.log"