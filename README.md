# Hermes

We introduce the **Hermes** which is a powerful chat model motivated from [zephyr-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha)!

Hermes is a fine-tuned version of [Mistral-7b](https://huggingface.co/mistralai/Mistral-7B-v0.1) that targets the powerful chat model. 
To achieve this goal, we utilized several chat dataset and preference dataset for training dataset, and used [DPO](https://arxiv.org/abs/2305.18290) that is the alternative of RLHF's reward modeling and PPO.
We also utilized [NEFTune](https://arxiv.org/abs/2310.05914) and LoRA to more improve the performance and efficiency of Hermes!
We found that the used method are very effective through the experiments with [MT-Bench](https://arxiv.org/abs/2306.05685)!

We hope that this repository will provide people with insight into chat model training using DPO, and that it will be helpful for future researches.
The code of SFT and DPO were referred to [TRL example codes](https://github.com/huggingface/trl/tree/main/examples/research_projects/stack_llama_2/scripts)

All models and dataset are available via HuggingFace: [Caritinoe5930](https://huggingface.co/Cartinoe5930)

## Fine-tuning w/ LoRA

If you want to sample the dataset to reduce the training time, please run the `do_sample=True` code, if you don't want to, just run the `do_sample=False` code.
There are several individual parameters in the code, the description of these parameters are as follows:

- desired_configuration: The configuration of `accelerate`. You can choose several options such as `multi-gpu`

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