#!/bin/bash

#==============================================================================
# Configuration
#==============================================================================

# Batch processing settings
BATCH_SIZE=20
TOTAL_SAMPLES=4500
START_INDEX=0

# Model configuration
MODEL_NAME="gpt-4o-mini"
RESULTS_DIR="results/"
APPROACH="distractors"

#==============================================================================
# Main Processing Loop
#==============================================================================

echo "Starting batch processing of $TOTAL_SAMPLES samples..."
echo "Using model: $MODEL_NAME"

for ((START_INDEX=START_INDEX; START_INDEX<TOTAL_SAMPLES; START_INDEX+=BATCH_SIZE)); do
    # Calculate end index for current batch
    END_INDEX=$((START_INDEX + BATCH_SIZE))
    
    # Ensure END_INDEX doesn't exceed total samples
    if [ "$END_INDEX" -gt "$TOTAL_SAMPLES" ]; then
        END_INDEX=$TOTAL_SAMPLES
    fi

    echo "Processing batch: $START_INDEX to $END_INDEX"

    # Execute processing for current batch
    python main.py \
        --model_name "$MODEL_NAME" \
        --start_index "$START_INDEX" \
        --end_index "$END_INDEX" \
        --results_dir "$RESULTS_DIR" \
        --approach "$APPROACH"
    
    echo "Batch complete. Waiting for next iteration..."
    sleep 10
done

echo "Processing complete!"
