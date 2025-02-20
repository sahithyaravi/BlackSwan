#!/bin/bash

SPLIT="mmbench_dev_20230712"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=$(( ${#GPULIST[@]} / 2 )) # Calculate chunks for 2 GPUs per chunk

MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi


for IDX in $(seq 0 $((CHUNKS-1))); do
    GPU_IDX1=$((IDX * 2))  # First GPU index
    GPU_IDX2=$((GPU_IDX1 + 1))  # Second GPU index
    CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX1]},${GPULIST[$GPU_IDX2]} python -m llava.eval.model_vqa_mmbench \
        --model-path $MODEL_PATH \
        --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --answers-file runs/eval/$CKPT/mmbench/${CHUNKS}_${IDX}.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode $CONV_MODE  &
done

wait

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT


output_file=runs/eval/$CKPT/mmbench/$SPLIT.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat runs/eval/$CKPT/mmbench/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir runs/eval/$CKPT/mmbench \
    --upload-dir runs/eval/$CKPT/mmbench \
    --experiment $SPLIT
