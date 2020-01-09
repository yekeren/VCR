#!/bin/bash

set -o errexit
set -o nounset
set -x

num_val_shards=5
num_train_shards=10

# for ((i=0;i<${num_val_shards};i+=1)); do
#   python "dataset-tools/create_vcr_tfrecord.py" \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --output_tfrecord_path="/own_files/yekeren/VCR2/val.record" \
#     > "log/val_${i}.log" 2>&1 &
# done
# 
# for ((i=0;i<${num_train_shards};i+=1)); do
#   python "dataset-tools/create_vcr_tfrecord.py" \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --shard_id="$i" \
#     --output_tfrecord_path="/own_files/yekeren/VCR2/train.record" \
#     > "log/train_${i}.log" 2>&1 &

for ((i=0;i<${num_val_shards};i+=1)); do
  python "dataset-tools/create_vcr_tfrecord.py" \
    --noencode_jpeg \
    --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
    --num_shards="${num_val_shards}" \
    --shard_id="$i" \
    --output_tfrecord_path="/own_files/yekeren/VCR-no-img/val.record" \
    > "log/val_${i}.log" 2>&1 &
done

for ((i=0;i<${num_train_shards};i+=1)); do
  python "dataset-tools/create_vcr_tfrecord.py" \
    --noencode_jpeg \
    --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
    --num_shards="${num_train_shards}" \
    --shard_id="$i" \
    --output_tfrecord_path="/own_files/yekeren/VCR-no-img/train.record" \
    > "log/train_${i}.log" 2>&1 &
done
