import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import os

from trl import DPOTrainer

import argparse

def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_token", type=str, help="Required to upload models to hub.")

    parser.add_argument("--beta", type=float, default=0.1)

    parser.add_argument("--model_path", type=str, default="Cartinoe5930/Hermes_SFT")
    parser.add_argument("--dataset_path", type=str, default="Cartinoe5930/Hermes_preference")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)

    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, help="You can choose the strategy of saving model.")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)

    parser.add_argument("--output_dir", type=str, default="DPO/final_checkpoint")
    parser.add_argument("--hf_hub_path", type=str, default="The hub path to upload the model")

    return parser.parse_args()

def get_hermes_dataset(dataset):

    original_columns = dataset.column_names

    def prompt_formatting(dataset):
        result = ""
        for data in dataset:
            if data["role"] == "user":
                result += f"<|user|>\n{data['content']}</s>"
            elif data["role"] == "assistant":
                result += f"<|assitant|>\n{data['content']}</s>"
            elif data["role"] == "system":
                result += f"<|system|>\n{data['content']}</s>"

        return result

    def dataset_process(source, data):
        response_list = []
        if source == "hh-rlhf":
            human_splitted = "|split|".join(data.split("Human: "))
            assistant_splitted = "|split|".join(human_splitted.split("Assistant:"))
            all_splitted = assistant_splitted.split("|split|")[1:-1]
            for i in range(len(all_splitted)):
                if (i + 1) % 2 != 0:
                    response_list.append({"role": "user", "content": all_splitted[i]})
                else:
                    response_list.append({"role": "assistant", "content": all_splitted[i]})

        elif source == "rlhf-reward":
            human_splitted = "|split|".join(data.split("Human: "))
            assistant_splitted = "|split|".join(human_splitted.split("Assistant: "))
            all_splitted = assistant_splitted.split("|split|")[1:]
            for i in range(len(all_splitted)):
                if (i + 1) % 2 != 0:
                    response_list.append({"role": "user", "content": all_splitted[i]})
                else:
                    response_list.append({"role": "assistant", "content": all_splitted[i]})

        else:
            response_list.append({"role": "user", "content": data})

        return prompt_formatting(response_list) + "\n<|assistant|>\n"

    def return_prompt_and_responses(samples):            
        return {
            "prompt": [dataset_process(source, instruction) for source, instruction in zip(samples["source"], samples["prompt"])],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"]
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns,
    )

if __name__ == "__main__":
    args = args_parse()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    model.config.use_cache = False

    model_ref = AutoModelForCausalLM.from_pretrained(
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

    dataset = dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= args.max_length
    )

    train_dataset, eval_dataset = dataset.train_test_split(test_size=0.25, seed=42)

    train_dataset = get_hermes_dataset(dataset=train_dataset)
    eval_dataset = get_hermes_dataset(dataset=eval_dataset)

    training_args = TrainingArguments(
        num_train_epochs=args.num_epoch,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
    )

    dpo_trainer.train()
    dpo_trainer.save_model(args.output_dir)
    
    model.push_to_hub(
        args.hf_hub_path,
        use_temp_dir=True,
        use_auth_token=args.hf_token,
    )
    tokenizer.push_to_hub(
        args.hf_hup_path,
        use_temp_dir=True,
        use_auth_token=args.hf_token,
    )