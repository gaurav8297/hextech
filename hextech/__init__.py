import os

import matplotlib.pyplot as plt

from datasets import load_dataset
from vllm import LLM, SamplingParams

# vllm settings
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"
LLM_MODEL = "TinyLlama/TinyLlama_v1.1"
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model=LLM_MODEL, tensor_parallel_size=2)

# prompt settings
MAX_PROMPT_LEN = 2048
NUM_PROMPTS = 10


def get_share_gpt_prompts(num_prompts=10000, max_prompt_len=8192):
    share_gpt_dataset = load_dataset("theblackcat102/sharegpt-english")
    filter_dataset = share_gpt_dataset.data["train"]['conversations']
    prompts = [str(item[0]["text"]) for item in filter_dataset]
    prompts = [prompt for prompt in prompts if len(prompt) <= max_prompt_len]
    prompts = prompts[:num_prompts]
    return prompts


def print_prompt_len_distribution(prompts):
    if not prompts:
        print("No prompts provided.")
        return

    prompt_lens = [len(prompt) for prompt in prompts]
    mean_length = sum(prompt_lens) / len(prompt_lens)
    max_length = max(prompt_lens)
    min_length = min(prompt_lens)

    print(f"Mean prompt length: {mean_length:.2f}")
    print(f"Max prompt length: {max_length}")
    print(f"Min prompt length: {min_length}")

    plt.hist(prompt_lens, bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribution of Prompt Lengths")
    plt.xlabel("Prompt Length")
    plt.ylabel("Frequency")
    plt.axvline(mean_length, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {mean_length:.2f}")
    plt.legend()
    plt.show()


def generate_responses(llm, sampling_params, prompts):
    llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()
    return outputs


if __name__ == "__main__":
    prompts = get_share_gpt_prompts(num_prompts=NUM_PROMPTS, max_prompt_len=MAX_PROMPT_LEN)
    print_prompt_len_distribution(prompts)
    responses = generate_responses(llm, sampling_params, prompts)
    print(responses[:5])
