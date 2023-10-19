#### DPO Training Hyperparameters

The following hyperparameters were used for DPO:

|Hyperparameters|Value|
|---|---|
|learning rate|5e-7|
|epoch|1|
|train_batch_size|1|
|eval_batch_size|2|
|gradient_accumulation_steps|1|
|weight_decay|0.|
|warmup_ratio|0.1|
|lr_scheduler_type|cosine|
|num_devices|2 x A100 80G|

The following hyperparameters were used for LoRA training:

|Hyperparameter|Value|
|---|---|
|lora_r|8|
|lora_alpha|16|
|lora_dropout|0.05|