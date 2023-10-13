from datasets import load_dataset
import json

dataset_list = ["ultrafeedback", "hh-rlhf", "rlhf-reward"]

dataset_dict = {
    "ultrafeedback": "openbmb/UltraFeedback",
    "hh-rlhf": "Anthropic/hh-rlhf",
    "rlhf-reward": "yitingxie/rlhf-reward-datasets"
}

def ultrafeedback_process(dataset):
    ultrafeedback_prompt = []
    for data in dataset:
        prompt = data["instruction"]
        chosen = None
        rejected = None

        cur_1st = ""
        cur_2nd = ""
        first_score = 0
        second_score = 0

        for response in data["completions"]:
            if not response["response"]:
                continue
            cur_score = float(response["overall_score"])
            if cur_score > first_score:
                cur_2nd = copy.copy(cur_1st)
                second_score = first_score
                cur_1st = response["response"]
                first_score = cur_score
            elif cur_score > second_score:
                cur_2nd = response["response"]
                second_score = cur_score

        chosen = cur_1st
        rejected = cur_2nd

        response_dict = {
            "source": "ultrafeedback",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }

        ultrafeedback_prompt.append(response_dict)

    return ultrafeedback_prompt

def hh_rlhf_process(dataset):
    # data random sampling
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select([i for i in range(50000)])

    hh_rlhf_prompt = []

    for data in dataset:
        prompt = ""
        splitted_prompt = data["chosen"].split("Assistant: ")
        range_num = len(splitted_prompt) - 2 if len(splitted_prompt) - 2 != 0 else 1
        for i in range(range_num):
            prompt += splitted_prompt[i] + "Assistant: "

        chosen = data["chosen"].split("Assistant: ")[-1]
        rejected = data["rejected"].split("Assistant: ")[-1]

        response_dict = {
            "source": "hh-rlhf",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }

        hh_rlhf_prompt.append(response_dict)

    return hh_rlhf_prompt

def rlhf_reward_process(dataset):
    rlhf_reward_prompt = []
    for data in dataset:
        response_dict = {
            "source": "rlhf-reward",
            "prompt": data["prompt"],
            "chosen": data["chosen"].split("Assistant: ")[-1],
            "rejected": data["rejected"].split("Assistant: ")[-1]
        }

        rlhf_reward_prompt.append(response_dict)

    return rlhf_reward_prompt

if __name__ == "__main__":
    for dataset_name in dataset_list:
        dataset = load_dataset(dataset_dict[dataset_name], split='train')
        if dataset_name == "ultrafeedback":
            ultrafeedback_result = ultrafeedback_process(dataset)
        elif dataset_name == "hh-rlhf":
            hh_rlhf_result = hh_rlhf_process(dataset)
        elif dataset_name == "rlhf-reward":
            rlhf_reward_result = rlhf_reward_process(dataset)

    ares_dataset = ultrafeedback_result + hh_rlhf_result + rlhf_reward_result

    with open("data_pipeline/ares_dataset.json", "x") as f:
        json.dump(ares_dataset, f, indent=4)