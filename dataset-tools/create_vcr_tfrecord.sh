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

# for ((i=0;i<${num_val_shards};i+=1)); do
#   export CUDA_VISIBLE_DEVICES=$((i%5))
#   python "dataset-tools/create_vcr_frcnn.py" \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --fast_rcnn_config="configs/fast_rcnn/inception_resnet_v2_imagenet_2stages.pbtxt" \
#     --output_frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_imagenet_2stages" \
#     > "log/val_${i}.log" 2>&1 &
# done

# for ((i=0;i<${num_train_shards};i+=1)); do
#   export CUDA_VISIBLE_DEVICES=$((i%6))
#   python "dataset-tools/create_vcr_frcnn.py" \
#     --image_zip_file="data/vcr1images.zip" \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --shard_id="$i" \
#     --fast_rcnn_config="configs/fast_rcnn/inception_resnet_v2_imagenet_2stages.pbtxt" \
#     --output_frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_imagenet_2stages" \
#     > "log/train_${i}.log" 2>&1 &
# done

# for ((i=0;i<${num_val_shards};i+=1)); do
#   export CUDA_VISIBLE_DEVICES=$((i%5))
#   python "dataset-tools/create_vcr_frcnn.py" \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --fast_rcnn_config="configs/fast_rcnn/inception_resnet_v2_oid.pbtxt" \
#     --output_frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_oid" \
#     > "log/val_${i}.log" 2>&1 &
# done

# for ((i=0;i<${num_train_shards};i+=1)); do
#   export CUDA_VISIBLE_DEVICES=$((i%6))
#   python "dataset-tools/create_vcr_frcnn.py" \
#     --image_zip_file="data/vcr1images.zip" \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --shard_id="$i" \
#     --fast_rcnn_config="configs/fast_rcnn/inception_resnet_v2_oid.pbtxt" \
#     --output_frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_oid" \
#     > "log/train_${i}.log" 2>&1 &
# done

######################################################
# Bert features
######################################################
# for ((i=0;i<${num_val_shards};i+=1)); do
#   export CUDA_VISIBLE_DEVICES=$((i%5))
#   python "dataset-tools/create_vcr_bert.py" \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --bert_checkpoint_file="logs/bert_next_sentence_lr0.00003/model.ckpt-30005" \
#     --output_bert_feature_dir="output/bert-ft/cased_L-12_H-768_A-12" \
#     > "log/val_${i}.log" 2>&1 &
# done
# 
# for ((i=0;i<${num_train_shards};i+=1)); do
#   export CUDA_VISIBLE_DEVICES=$((i%5))
#   python "dataset-tools/create_vcr_bert.py" \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --shard_id="$i" \
#     --bert_checkpoint_file="logs/bert_next_sentence_lr0.00003/model.ckpt-30005" \
#     --output_bert_feature_dir="output/bert-ft/cased_L-12_H-768_A-12" \
#     > "log/train_${i}.log" 2>&1 &
# done

######################################################
# Tf record file without jpeg data - 1stage
######################################################
# for ((i=0;i<${num_val_shards};i+=1)); do
#   python "dataset-tools/create_vcr_tfrecord.py" \
#     --noencode_jpeg \
#     --only_use_relevant_dets \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --bert_feature_dir="output/bert-ft/cased_L-12_H-768_A-12" \
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
#     --bert_feature_dir="output/bert-ft/cased_L-12_H-768_A-12" \
#     --frcnn_feature_dir="/own_files/yekeren/fast_rcnn/inception_resnet_v2_imagenet" \
#     --output_tfrecord_path="/own_files/yekeren/VCR-relevant-only/train.record" \
#     > "log/train_${i}.log" 2>&1 &
# done

######################################################
# Tf record file without jpeg data - 2stage
######################################################
# for ((i=0;i<${num_val_shards};i+=1)); do
#   python "dataset-tools/create_vcr_tfrecord.py" \
#     --noencode_jpeg \
#     --noonly_use_relevant_dets \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --bert_feature_dir="output/bert-ft/cased_L-12_H-768_A-12" \
#     --frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_imagenet_2stages" \
#     --output_tfrecord_path="/own_files/yekeren/VCR-2stages-allboxes/val.record" \
#     > "log/val_${i}.log" 2>&1 &
# done
# 
# for ((i=0;i<${num_train_shards};i+=1)); do
#   python "dataset-tools/create_vcr_tfrecord.py" \
#     --noencode_jpeg \
#     --noonly_use_relevant_dets \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --shard_id="$i" \
#     --bert_feature_dir="output/bert-ft/cased_L-12_H-768_A-12" \
#     --frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_imagenet_2stages" \
#     --output_tfrecord_path="/own_files/yekeren/VCR-2stages-allboxes/train.record" \
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
#     --bert_feature_dir="output/bert-ft/cased_L-12_H-768_A-12" \
#     --frcnn_feature_dir="/own_files/yekeren/fast_rcnn/inception_resnet_v2_imagenet" \
#     --output_tfrecord_path="/own_files/yekeren/VCR-final/val.record" \
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
#     --bert_feature_dir="output/bert-ft/cased_L-12_H-768_A-12" \
#     --frcnn_feature_dir="/own_files/yekeren/fast_rcnn/inception_resnet_v2_imagenet" \
#     --output_tfrecord_path="/own_files/yekeren/VCR-final/train.record" \
#     > "log/train_${i}.log" 2>&1 &
# done


# ######################################################
# # Tf record file with only text annotations
# ######################################################
# for ((i=0;i<${num_val_shards};i+=1)); do
#   python "dataset-tools/create_vcr_text_only_tfrecord.py" \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --output_tfrecord_path="output/val.record" \
#     > "log/val_${i}.log" 2>&1 &
# done
# 
# for ((i=0;i<${num_train_shards};i+=1)); do
#   python "dataset-tools/create_vcr_text_only_tfrecord.py" \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --shard_id="$i" \
#     --output_tfrecord_path="output/train.record" \
#     > "log/train_${i}.log" 2>&1 &
# done

# ######################################################
# # Tf record file with both text annotations and the image
# ######################################################
# for ((i=0;i<${num_val_shards};i+=1)); do
#   python "dataset-tools/create_vcr_text_image_tfrecord.py" \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --output_tfrecord_path="/own_files/yekeren/VCR-text_and_image/val.record" \
#     > "log/val_${i}.log" 2>&1 &
# done
# 
# for ((i=0;i<${num_train_shards};i+=1)); do
#   python "dataset-tools/create_vcr_text_image_tfrecord.py" \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --shard_id="$i" \
#     --output_tfrecord_path="/own_files/yekeren/VCR-text_and_image/train.record" \
#     > "log/train_${i}.log" 2>&1 &
# done

# ######################################################
# # Tf record file with text annotations and fast rcnn features.
# ######################################################
# for ((i=0;i<${num_val_shards};i+=1)); do
#   python "dataset-tools/create_vcr_text_frcnn_tfrecord.py" \
#     --noonly_use_relevant_dets \
#     --annotations_jsonl_file="data/vcr1annots/val.jsonl" \
#     --num_shards="${num_val_shards}" \
#     --shard_id="$i" \
#     --frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_imagenet_2stages" \
#     --output_tfrecord_path="/own_files/yekeren/VCR-text_and_frcnn/val.record" \
#     > "log/val_${i}.log" 2>&1 &
# done
# 
# for ((i=0;i<${num_train_shards};i+=1)); do
#   python "dataset-tools/create_vcr_text_frcnn_tfrecord.py" \
#     --noonly_use_relevant_dets \
#     --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
#     --num_shards="${num_train_shards}" \
#     --shard_id="$i" \
#     --frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_imagenet_2stages" \
#     --output_tfrecord_path="/own_files/yekeren/VCR-text_and_frcnn/train.record" \
#     > "log/train_${i}.log" 2>&1 &
# done

# ######################################################
# # Adversarial training. Tf record file with text annotations and fast rcnn features.
# ######################################################
for ((i=0;i<${num_train_shards};i+=1)); do
  python "dataset-tools/create_vcr_text_frcnn_tfrecord.py" \
    --noonly_use_relevant_dets \
    --annotations_jsonl_file="data/vcr1annots/train.jsonl" \
    --num_shards="${num_train_shards}" \
    --shard_id="$i" \
    --adversarial_annotations_jsonl_file="data/adv_train.jsonl" \
    --frcnn_feature_dir="output/fast_rcnn/inception_resnet_v2_imagenet_2stages" \
    --output_tfrecord_path="/own_files/yekeren/VCR-adv-text_and_frcnn/train.record" \
    > "log/train_${i}.log" 2>&1 &
done
