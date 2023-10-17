# Hermes

- accelerate config를 사용해서 accelerate 설정을 진행한다면 속도는 확실히 빨라질 수 있을 것 같음.
- FSDP & DeepSpeed 활용할 수 있도록 configuration 구성
- accelerate config 구성 후 configuration.yaml 파일 저장해두기
- deepspeed installation이 Runpod에서 밖에 되지 않으니 참고할 것

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

```
!accelerate launch --config_file=accelerate_configs/multi_gpu.yaml --num_processes 2 SFT/SFTTrainer_lora.py \
    --hf_token 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 \
    --hf_hub_path Cartinoe5930/Hermes_SFT \
    --save_strategy epoch \
    --sample_size 500000 \
    --seq_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_train_batch_size 2
```
- seq_length 줄여야 할 것 같고
- multi_gpu.yaml 사용
- batch_size를 늘릴 수 있는 방법 알아보기

일단 대부분의 상황에서 OOM이 나오는 것을 확인함.
- num_proc으로 데이터를 불러오니 OOM이 잘 나오는 것을 확인. 대신 training speed는 확실히 빨라졌음.
- seq_length를 2048로 맞추니 대부분의 경우에서 OOM이 나옴. 따라서 성능에는 저하가 생기겠지만, seq_len=1024로 설정하는 것이 좋을 것 같음.
- 