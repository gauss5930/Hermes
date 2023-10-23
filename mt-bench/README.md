# MT-Bench

The evaluation codes were brought from LMSYS's [FastChat](https://github.com/lm-sys/FastChat).
Therefore, for conducting MT-Bench evaluation, you should run these codes on the seperate notebook.

## Install

```
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker, llm_judge]"
```

## MT-Bench

### Evaluate a Hermes on MT-Bench

**Step 1. Add a new model(Hermes) to the FastChat**

To support a Hermes in FastChat, we need to correctly handle its prompt template and model loading.
Please run the following code to run the model evaluation with correct prompts.

```
python3 -m fastchat.serve.cli --model Cartinoe5930/Hermes-7b
```

You can add --debug to see the actual prompt sent to the model.

**Step 2. Generate model answers to MT-Bench questions**

```
python fastchat/llm_judge/gen_model_answer.py \
    --model-path Cartinoe5930/Hermes-7b \
    --model-id hermes-7b
```

The answers will be saved to `data/mt_bench/model_answer/hermes-7b.jsonl`.

You can also specify `--num-gpus-per-model` for model parallelism (needed for large 65B models) and `--num-gpus-total` to parallelize answer generation with multiple GPUs.

**Step 3. Generate GPT-4 judgments**

There are several options to use GPT-4 as a judge, however we use single-answer grading following the suggestion of MT-bench paper.
For each turn, GPT-4 will give a score on a scale of 10.
We then compute the average score on all turns.

```
export OPENAI_API_KEY=XXXXX   # set the openai api key
python fastchat/llm_judge/gen_judgment.py --model-list hermes-7b
```

**Step 4. Show MT-Bench scores**

```
python fastchat/llm_judge/show_result.py --model-list hermes-7b
```