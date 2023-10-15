import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, TrainingArguments
from peft import AutoPeftModelForCausalLM, LoraConfig
import os

from trl import DPOTrainer

import argparse

def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--beta", type=float, default=0.1)

    parser.add_argument("--model_path", type=str, default="Cartinoe5930/Hermes_SFT_adapter")
    parser.add_argument("--dataset_path", type=str, default="Cartinoe5930/Hermes_preference")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)

    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_r", type=int, default=8)

    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)

    parser.add_argument("--output_dir", type=str, default="DPO/")

    return parser.parse_args()

def get_hermes_dataset(dataset, tokenizer, num_proc=42):

    original_columns = dataset.column_names

    def dataset_process(data):
        response_list = []
        if data["source"] == "hh-rlhf":
            human_splitted = "|split|".join(data["prompt"].split("Human: ")[1:])
            assistant_splitted = "|split|".join(human_splitted.split("Assistant:"))
            all_splitted = assistant_splitted.split("|split|")[1:-1]
            for i in range(len(all_splitted)):
                if (i + 1) % 2 != 0:
                    response_list.append({"role": "user", "content": all_splitted[i]})
                else:
                    response_list.append({"role": "assistant", "content": all_splitted[i]})

        elif data["source"] == "rlhf-reward":
            human_splitted = "|split|".join(data["prompt"].split("Human: ")[1:])
            assistant_splitted = "|split|".join(human_splitted.split("Assistant: "))
            all_splitted = assistant_splitted.split("|split|")[1:]
            for i in range(len(all_splitted)):
                if (i + 1) % 2 != 0:
                    response_list.append({"role": "user", "content": all_splitted[i]})
                else:
                    response_list.append({"role": "assistant", "content": all_splitted[i]})

        else:
            response_list.append({"role": "user", "content": data["prompt"]})

        return tokenizer.apply_chat_template(response_list) + "\n<|assistant|>\n"

    def return_prompt_and_responses(samples):            
        return {
            "prompt": [dataset_process(instruction) for instruction in samples],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"]
        }
    
    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

if __name__ == "__main__":
    args = args_parse()

    gradient_accumulation_steps = args.batch_size // args.per_device_train_batch_size

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    model.config.use_cache = False

    model_ref = AutoPeftModelForCausalLM.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        args.dataset_path,
        split="train"
    )

    train_dataset, eval_dataset = dataset.train_test_split(test_size=0.25, seed=42)

    train_dataset = get_hermes_dataset(dataset=train_dataset, tokenizer=tokenizer)
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= args.max_length
    )

    eval_dataset = get_hermes_dataset(dataset=eval_dataset, tokenizer=tokenizer)
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= args.max_length
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        remove_unused_columns=False,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_trainer = DPOTrainer(
        model, 
        model_ref,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
    )

    dpo_trainer.train()
    dpo_trainer.save_model(args.output_dir)

    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)