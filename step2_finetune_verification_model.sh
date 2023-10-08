CUDA_VISIBLE_DEVICES="0" python finetuning.py \
    --output_dir=checkpoints/verification_model \
    --model_name_or_path=checkpoints/pretrained_hybrib_model \
    --model_type=2 \
    --config_name=configs/verification-model-config.json \
    --verif_conf="verification/verif_conf.json" \
    --do_train \
    --data_dir=data/wikitables_v2 \
    --evaluate_during_training \
    --per_gpu_train_batch_size=20 \
    --per_gpu_eval_batch_size=20 \
    --gradient_accumulation_steps=2 \
    --learning_rate=5e-5 \
    --num_train_epochs=10 \
    --save_total_limit=10 \
    --seed=1 \
    --cache_dir=cache \
    --overwrite_output_dir \
    --save_steps=-1\
    --logging_steps=1500 \
    --warmup_steps=5000
