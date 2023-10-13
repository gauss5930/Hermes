import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from accelerate import Accelerator

import argparse

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset_name", type=str, default="Cartinoe5930/Hermes_preference")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=None)

    parser.add_arugment("--num_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--group_by_length", type=bool, default=False)
    parser.add_argument("--packing", type=bool, default=True)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="SFTTrainer/"
    )

    return parser.parse_args()

def chars_token_ratio(dataset, tokenizer, nb_examples=500):
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example["data"])
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(example["data"]).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(example["data"]))

    return total_characters / total_tokens

def process_dataset(example, tokenizer):
    instruction_prompt = []
    for i in range(len(example)):
        if (i + 1) % 2 != 0:
            instruction_prompt.append({"role": "user", "content": example[i]})
        else:
            instruction_prompt.append({"role": "asisstant", "content": example[i]})

    result_data = tokenizer.apply_chat_template(instruction_prompt, tokenize=False)

    return result_data

def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        num_proc=args.num_workers if args.num_workers else None,
    )

    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_data = dataset["train"]
    valid_data = dataset["test"]

    chars_per_token = chars_token_ratio(train_data, tokenizer)

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=process_dataset,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )

    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=process_dataset,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )

    return train_dataset, valid_dataset

if __name__ == "__main__":
    args = args_parse()

    gradient_accumulation_steps = args.batch_size // args.per_device_train_batch_size

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map={"": Accelerator().process_index},
        trust_remote_code=True,
    )
    model.config.use_cache = False

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
        num_train_epochs=args.num_epochs
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.num_warmup_ratio,
        bf16=True,
        remove_unused_columns=False,
    )

    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=args.packing,
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    output_dir = args.output_dir + "final_checkpoint"
    trainer.model.save_pretrained(output_dir)