import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
import random

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from trl.trainer import ConstantLengthDataset
from accelerate import Accelerator

import argparse
import os

def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_token", type=str, help="Required to upload models to hub.")

    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset_name", type=str, default="stingning/ultrachat")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--do_sample", type=bool, default=True, help="Sample the dataset.")
    parser.add_argument("--sample_size", type=int, default=None)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, help="You can choose the strategy of saving model.")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_strategy", type=str, help="You can choose the strategy of evaluating model.")
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--group_by_length", type=bool, default=False)
    parser.add_argument("--packing", type=bool, default=False)

    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_r", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="SFT/final_checkpoint"
    )
    parser.add_argument(
        "--hf_hub_path",
        type=str,
        help="The hub path to upload the model"
    )

    return parser.parse_args()

def dataset_sampling(dataset, sample_size):
    random_indices = random.sample(range(len(dataset)), sample_size)

    random_samples = dataset.select(random_indices)

    return random_samples

def process_dataset(example):

    def prompt_formatting(dataset):
        result = ""
        for data in dataset:
            if data["role"] == "user":
                result += f"<|user|>\n{data['content']}</s>"
            elif data["role"] == "assistant":
                result += f"<|assistant|>\n{data['content']}</s>"
            elif data["role"] == "system":
                result += f"<|system|>\n{data['content']}</s>"

        return result
    
    result_data = []
    for n in range(len(example["data"])):
        instruction_prompt = []
        for i in range(len(example["data"][n])):
            if (i + 1) % 2 != 0:
                instruction_prompt.append({"role": "user", "content": example["data"][n][i]})
            else:
                instruction_prompt.append({"role": "assistant", "content": example["data"][n][i]})

        result_data.append(prompt_formatting(instruction_prompt))

    return result_data

def create_datasets(args):
    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        num_proc=args.num_workers if args.num_workers else None,
    )

    if args.do_sample:
        dataset = dataset_sampling(dataset, args.sample_size)

    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_data = dataset["train"]
    valid_data = dataset["test"]

    return train_data, valid_data

if __name__ == "__main__":
    args = args_parse()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map={"": Accelerator().process_index},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    special_tokens_dict = {"additional_special_tokens": ["<unk>", "<s>", "</s>"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.eval_strategy,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        eval_steps=args.save_steps if args.save_strategy == "steps" else None,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        remove_unused_columns=False,
    )

    train_dataset, eval_dataset = create_datasets(args)

    response_template = "<|assistant|>\n"
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=process_dataset,
        data_collator=collator,
        peft_config=peft_config,
        packing=args.packing,
        max_seq_length=args.seq_length,
        tokenizer=tokenizer,
        neftune_noise_alpha=5,
        args=training_args
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    # Free memory for merging weights
    del model
    torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(args.output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    model.push_to_hub(
        args.hf_hub_path,
        use_temp_dir=True,
        use_auth_token=args.hf_token,
    )
    tokenizer.push_to_hub(
        args.hf_hub_path,
        use_temp_dir=True,
        use_auth_token=args.hf_token,
    )