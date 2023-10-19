# Hermes

<p align="center"><img src="/assets/Hermes.png", width='250', height='250'></p>

We introduce the **Hermes** which is a powerful chat model motivated from [zephyr-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha)!

Hermes is a fine-tuned version of [Mistral-7b](https://huggingface.co/mistralai/Mistral-7B-v0.1) that targets the powerful chat model. 
To achieve this goal, we utilized several chat dataset and preference dataset for training dataset, and used [DPO](https://arxiv.org/abs/2305.18290) that is the alternative of RLHF's reward modeling and PPO.
We also utilized [NEFTune](https://arxiv.org/abs/2310.05914) and LoRA to more improve the performance and efficiency of Hermes!
We found that the used method are very effective through the experiments with [MT-Bench](https://arxiv.org/abs/2306.05685)!

We hope that this repository will provide people with insight into chat model training using DPO, and that it will be helpful for future researches.
The code of SFT and DPO were referred to [TRL example codes](https://github.com/huggingface/trl/tree/main/examples/research_projects/stack_llama_2/scripts)

All models and dataset are available via HuggingFace: [Caritinoe5930](https://huggingface.co/Cartinoe5930)

## Dataset

The dataset used for Hermes is divided into two categories: one is chat data for SFT, and the other is preference data for DPO.
The following datasets were used for SFT and DPO:

- SFT: [Ultrachat](https://huggingface.co/datasets/stingning/ultrachat) - The conversation data between two ChatGPT
- DPO: [Hermes_preference](https://huggingface.co/datasets/Cartinoe5930/Hermes_preference) - The mixture of several preference dataset([UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback), [hh-rlhf](https://huggingface.co/datasets/openbmb/UltraFeedback), [rlhf-reward-datasets](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets))

We utilized randomly sampled Ultrachat data(1.4M -> 500K) for Supervised Fine-tuning due to the lack of computing resources.
However, we utilized all Hermes_preference dataset for Direct Preference Optimization since it could be sufficiently run with the given computing resources.

## Chat Prompt Format

We utilized the same prompt format with [Zephyr-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha).
The prompt format is implemented in `chat_tempalte` of `tokenizer_config.json`. 
Therefore you can use Hermes' prompt format using `tokenizer.apply_chat_template()`.
The following format is the prompt format of Hermes:

```
<|system|>
The instruction to the model.
<|user|>
The question to the model.
<|assistant|>
The response of the model to the instruction and question that are given by the user.
```

## Training Procedure

The overall training procedure of Hermes consists of major two steps.

- SFT(NEFTune)
- DPO

You can check the hyperparameters of SFT and DPO in README file of each folder! - `SFT` & `DPO`

### Setup

Please install the all dependencies that are required to run the SFT and DPO codes.

```
git clone https://github.com/gauss5930/Hermes.git
cd Hermes
pip install -r requirements.txt
```

Additionally, you can choose the configuration of `accelerate` when running SFT and DPO codes, so hope you can choose an appropriate configuration considering your computing environment.

### Fine-tuning

#### SFT

If you want to sample the dataset to reduce the training time, please run the `do_sample=True` code, if you don't want to, just run the `do_sample=False` code.
If you use sampling, you can adjust the sampling size by tuning sample_size parameter.
There are several individual parameters in the code, the description of these parameters are as follows:

- config_file(desired_configuration): The configuration of `accelerate`. You can choose several options such as `multi-gpu` or `deepspeed_zeron` depend on condition. THe configuration files are in the `accelerate_configs` folder.
- num_processes(GPU_NUMS): The number of GPUs you will be using.
- RUN_FILE: For full fine-tuning, use `SFT/SFTTrainer.py` and for LoRA fine-tuning, use `SFT/SFTTrainer_lora.py`.
- hf_token(YOUR_HUGGINGFACE_TOKEN): Your HuggingFace Access token. It will be used for uploading fine-tuned model to hub.
- hf_hub_path(PATH_TO_UPLOAD_MODEL): The path where to upload the fine-tuned model.

**SFT Trainer(do_sample=True)**
```
accelerate launch --config_file=accelerate_configs/desired_configuration --num_processes GPU_NUMS RUN_FILE \
    --hf_token YOUR_HUGGINGFACE_TOKEN \
    --hf_hub_path PATH_TO_UPLOAD_MODEL \
    --save_strategy epoch \
    --num_workers GPU_NUMS \
    --sample_size 500000
```

**SFT Trainer(do_sample=False)**
```
accelerate launch --config_file=accelerate_configs/desired_configuration --num_processes GPU_NUMS RUN_FILE \
    --hf_token YOUR_HUGGINGFACE_TOKEN \
    --hf_hub_path PATH_TO_UPLOAD_MODEL \
    --save_strategy epoch \
    --num_workers GPU_NUMS \
    --do_sample False
```

#### DPO

After the Supervised Fine-tuning, it's time to do DPO.
Just like did with SFT, all we have to do is run the following code!
The description of parameters are as follows:

- config_file(desired_configuration): The configuration of `accelerate`. You can choose several options such as `multi-gpu` or `deepspeed_zeron`. THe configuration files are in the `accelerate_configs` folder.
- num_processes(GPU_NUMS): The number of GPUs you will be using.
- RUN_FILE: For full fine-tuning, use `DPO/DPOTrainer.py` and for LoRA fine-tuning, use `DPO/DPOTrainer_lora.py`.
- hf_token(YOUR_HUGGINGFACE_TOKEN): Your HuggingFace Access token. It will be used for uploading fine-tuned model to hub.
- hf_hub_path(PATH_TO_UPLOAD_MODEL): The path where to upload the fine-tuned model.

**DPO Trainer**
```
accelerate launch --config_file=accelerate_configs/desired_configuration --num_processes GPU_NUMBER DPO/DPOTrainer.py \
    --hf_token YOUR_HUGGINGFACE_TOKEN \
    --hf_hub_path PATH_TO_UPLOAD_MODEL \
    --save_strategy epoch 
```

### Inference

The inference with Hermes can be achieved on HuggingFace's `pipeline()` function!
In addition, you can utilize the tokenizer's chat template when inferencing with Hermes.

```python
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="Cartinoe5930/Hermes-7b", torch_dtype=torch.bfloat16, device_map="auto")

messages = [
    {
        "role": "system",
        "content": the instruction to the model
    },
    {
        "role": "user",
        "content": the question you want to ask
    }
]

prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_template=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
```

## Citation

- [Zephyr-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha)
- [TRL example](https://github.com/huggingface/trl/tree/main/examples/research_projects/stack_llama_2/scripts)

```
@misc{jiang2023mistral,
      title={Mistral 7B}, 
      author={Albert Q. Jiang and Alexandre Sablayrolles and Arthur Mensch and Chris Bamford and Devendra Singh Chaplot and Diego de las Casas and Florian Bressand and Gianna Lengyel and Guillaume Lample and Lucile Saulnier and Lélio Renard Lavaud and Marie-Anne Lachaux and Pierre Stock and Teven Le Scao and Thibaut Lavril and Thomas Wang and Timothée Lacroix and William El Sayed},
      year={2023},
      eprint={2310.06825},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```