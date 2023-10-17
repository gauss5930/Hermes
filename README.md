# Hermes

- accelerate config를 사용해서 accelerate 설정을 진행한다면 속도는 확실히 빨라질 수 있을 것 같음.
- FSDP & DeepSpeed 활용할 수 있도록 configuration 구성
- accelerate config 구성 후 configuration.yaml 파일 저장해두기
- deepspeed installation이 Runpod에서 밖에 되지 않으니 참고할 것

**accelerate config**
```
accelerate config
```

**SFT Trainer**
```
accelerate launch --config_file=accelerate_configs/desired_configuration --num_processes GPU_NUMBER SFT/SFTTrainer.py \
    --hf_token YOUR_HUGGINGFACE_TOKEN \
    --hf_hub_path PATH_TO_UPLOAD_MODEL \
    --save_strategy epoch
```

**DPO Trainer**
```
accelerate launch --config_file=accelerate_configs/desired_configuration --num_processes GPU_NUMBER DPO/DPOTrainer.py \
    --hf_token YOUR_HUGGINGFACE_TOKEN \
    --hf_hub_path PATH_TO_UPLOAD_MODEL \
    --save_strategy epoch
```