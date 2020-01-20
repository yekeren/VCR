#!/bin/bash

set -o errexit
set -o nounset
set -x

num_val_shards=5
num_train_shards=10

######################################################
# FRCNN features
######################################################
# for ((i=0;i<${num_val_shards};i+=1)); do
#   export CUDA_VISIBLE_DEVICES=$((i%5))
#   python "dataset-tools/create_vcr_frcnn.py" \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --output_frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_imagenet" \
#     > "log/val_${i}.log" 2>&1 &
# done
# 
# for ((i=0;i<${num_train_shards};i+=1)); do
#   export CUDA_VISIBLE_DEVICES=$((i%6))
#   python "dataset-tools/create_vcr_frcnn.py" \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --shard_id="$i" \
#     --output_frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_imagenet" \
#     > "log/train_${i}.log" 2>&1 &
# done

######################################################
# RCNN features
######################################################
# for ((i=0;i<${num_val_shards};i+=1)); do
#   export CUDA_VISIBLE_DEVICES=$((i%5))
#   python "dataset-tools/create_vcr_rcnn.py" \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --output_rcnn_feature_dir="/own_files/yekeren/rcnn/inception_v4_imagenet" \
#     > "log/val_${i}.log" 2>&1 &
# done

num_train_shards=6
for ((i=0;i<${num_train_shards};i+=1)); do
  export CUDA_VISIBLE_DEVICES=$((i%6))
  python "dataset-tools/create_vcr_rcnn.py" \
    --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
    --num_shards="${num_train_shards}" \
    --shard_id="$i" \
    --output_rcnn_feature_dir="/own_files/yekeren/rcnn/inception_v4_imagenet" \
    > "log/train_${i}.log" 2>&1 &
done
exit 0

######################################################
# Bert features
######################################################
# for ((i=0;i<${num_val_shards};i+=1)); do
#   export CUDA_VISIBLE_DEVICES=$((i%5))
#   python "dataset-tools/create_vcr_bert.py" \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --output_bert_feature_dir="output/bert/cased_L-12_H-768_A-12" \
#     > "log/val_${i}.log" 2>&1 &
# done
# 
# for ((i=0;i<${num_train_shards};i+=1)); do
#   export CUDA_VISIBLE_DEVICES=$((i%6))
#   python "dataset-tools/create_vcr_bert.py" \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --shard_id="$i" \
#     --output_bert_feature_dir="output/bert/cased_L-12_H-768_A-12" \
#     > "log/train_${i}.log" 2>&1 &
# done

######################################################
# Tf record file without jpeg data
######################################################
# num_val_shards=5
# num_train_shards=10
# for ((i=0;i<${num_val_shards};i+=1)); do
#   python "dataset-tools/create_vcr_tfrecord.py" \
#     --noencode_jpeg \
#     --only_use_relevant_dets \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --frcnn_feature_dir="/own_files/yekeren/fast_rcnn/inception_resnet_v2_imagenet" \
#     --output_tfrecord_path="/own_files/yekeren/VCR-relevant-only/val.record" \
#     > "log/val_${i}.log" 2>&1 &
# done
# 
# for ((i=0;i<${num_train_shards};i+=1)); do
#   python "dataset-tools/create_vcr_tfrecord.py" \
#     --noencode_jpeg \
#     --only_use_relevant_dets \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --shard_id="$i" \
#     --frcnn_feature_dir="/own_files/yekeren/fast_rcnn/inception_resnet_v2_imagenet" \
#     --output_tfrecord_path="/own_files/yekeren/VCR-relevant-only/train.record" \
#     > "log/train_${i}.log" 2>&1 &
# done

######################################################
# Tf record file with jpeg data
######################################################
# for ((i=0;i<${num_val_shards};i+=1)); do
#   python "dataset-tools/create_vcr_tfrecord.py" \
#     --encode_jpeg \
#     --only_use_relevant_dets \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --frcnn_feature_dir="/own_files/yekeren/fast_rcnn/inception_resnet_v2_imagenet" \
#     --output_tfrecord_path="/own_files/yekeren/VCR-with-image/val.record" \
#     > "log/val_${i}.log" 2>&1 &
# done
# 
# for ((i=0;i<${num_train_shards};i+=1)); do
#   python "dataset-tools/create_vcr_tfrecord.py" \
#     --encode_jpeg \
#     --only_use_relevant_dets \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --shard_id="$i" \
#     --frcnn_feature_dir="/own_files/yekeren/fast_rcnn/inception_resnet_v2_imagenet" \
#     --output_tfrecord_path="/own_files/yekeren/VCR-with-image/train.record" \
#     > "log/train_${i}.log" 2>&1 &
# done
