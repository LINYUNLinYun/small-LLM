export CUDA_VISIBLE_DEVICES=0
# conda activate huanhuan
python lora_finetune.py \
    --model_name_or_path models/Qwen2.5-1.5B \
    --train_files /root/nfs/code/small-LLM/myLLaMA2/dataset/seq-monkey/BelleGroup/train_3.5M_CN.json \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --do_train \
    --output_dir /output/lora_sft \
    --evaluation_strategy  no \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --warmup_steps 200 \
    --logging_dir /output/lora_sft/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 2048 \
    --bf16 True\
    --gradient_checkpointing \
    --report_to none     
    # --report_to swanlab     \
    # --deepspeed ./ds_config_zero.json \
    
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \