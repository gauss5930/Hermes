#### Supervised Fine-Tuning Hyperparameters

We utilized NEFTune to improve the Hermes' conversation ability. 
Thanks to the authors of [NEFTune](https://arxiv.org/abs/2310.05914)!

The following hyperparameters were used during training:

- learning_rate: 5e-07
- neftune_noise_alpha: 5
- train_batch_size: 1
- eval_batch_size: 2
- seed: 42
- distributed_type: deepspeed-zero2
- num_devices: 2 x A100 80G
- total_train_batch_size: 2
- total_eval_batch_size: 4
- grdient_accumulation_steps: 1
- lr_scheduler_type: cosine
- warmup_ratio: 0.1
- weight_decay: 0
- num_epochs: 1

The following hyperparameters were used for LoRA training:

- lora_r: 8
- lora_alpha: 16
- lora_dropout: 0.05