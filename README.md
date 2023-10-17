# Hermes

- accelerate config를 사용해서 accelerate 설정을 진행한다면 속도는 확실히 빨라질 수 있을 것 같음.
- FSDP & DeepSpeed 활용할 수 있도록 configuration 구성
- accelerate config 구성 후 configuration.yaml 파일 저장해두기
- deepspeed installation이 Runpod에서 밖에 되지 않으니 참고할 것



#### Supervised Fine-Tuning Hyperparameters

The following hyperparameters were used during training:

- learning_rate: 5e-07
- train_batch_size: 2
- eval_batch_size: 4
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2 x A100 80G
- total_train_batch_size: 4
- total_eval_batch_size: 8
- grdient_accumulation_steps: 1
- lr_scheduler_type: cosine
- warmup_ratio: 0.1
- weight_decay: 0
- num_epochs: 1

The following hyperparameters were used for LoRA training:

- lora_r: 8
- lora_alpha: 16
- lora_dropout: 0.05

**SFT Trainer(do_sample=True)**
```
accelerate launch --config_file=accelerate_configs/desired_configuration --num_processes GPU_NUMBER SFT/SFTTrainer.py \
    --hf_token YOUR_HUGGINGFACE_TOKEN \
    --hf_hub_path PATH_TO_UPLOAD_MODEL \
    --save_strategy epoch \
    --num_workers GPU_NUMS \
    --sample_size 500000
```

**SFT Trainer(do_sample=False)**
```
accelerate launch --config_file=accelerate_configs/desired_configuration --num_processes GPU_NUMBER SFT/SFTTrainer.py \
    --hf_token YOUR_HUGGINGFACE_TOKEN \
    --hf_hub_path PATH_TO_UPLOAD_MODEL \
    --save_strategy epoch \
    --num_workers GPU_NUMS \
    --do_sample False
```

**DPO Trainer**
```
accelerate launch --config_file=accelerate_configs/desired_configuration --num_processes GPU_NUMBER DPO/DPOTrainer.py \
    --hf_token YOUR_HUGGINGFACE_TOKEN \
    --hf_hub_path PATH_TO_UPLOAD_MODEL \
    --save_strategy epoch \
    --num_workers GPU_NUMS
```

!accelerate launch --config_file=accelerate_configs/multi_gpu.yaml --num_processes 2 SFT/SFTTrainer_lora.py \
    --hf_token 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 \
    --hf_hub_path Cartinoe5930/Hermes_SFT \
    --save_strategy epoch \
    --num_workers 2 \
    --sample_size 500000 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 