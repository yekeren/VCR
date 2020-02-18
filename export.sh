

for ((i=0;i<10;i+=1)); do
  export CUDA_VISIBLE_DEVICES=$((i%5+1))
  python modeling/shortcut_detector_main.py \
    --model_dir "logs/next_bce_lr0.00003/" \
    --pipeline_proto "configs/bak/next_bce_lr0.00003_export${i}.pbtxt" \
    --output_jsonl_file "data/adversarial_train${i}.json" \
    >> "log/export${i}.log" 2>&1 &
done
