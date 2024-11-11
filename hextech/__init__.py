import os
import argparse

import matplotlib.pyplot as plt

from datasets import load_dataset
from vllm import LLM, SamplingParams

# vllm settings
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"
LLM_MODEL = "TinyLlama/TinyLlama_v1.1"
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# prompt settings
MAX_PROMPT_LEN = 2048


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


def generate_responses(llm, sampling_params, prompts, skip_profile=False):
    if not skip_profile:
        llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    ttft = 0.0
    schedule_delay = 0.0
    avg_tpot = 0.0
    for output in outputs:
        time_to_first_token = output.metrics.first_token_time - output.metrics.arrival_time
        schedule_delay += output.metrics.scheduler_time
        ttft += time_to_first_token
        total_tokens = 0
        for completion_output in output.outputs:
            total_tokens += len(completion_output.token_ids)
        avg_tpot += (output.metrics.finished_time - output.metrics.first_token_time) / total_tokens
    print(outputs[:3])
    print(f"Average time to first token: {ttft / len(outputs):.4f} seconds")
    print(f"Average time per output token: {avg_tpot / len(outputs):.4f} seconds")
    print(f"Average schedule delay: {schedule_delay / len(outputs):.4f} seconds")
    if not skip_profile:
        print(f"Generating profile")
        llm.stop_profile()
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--num_prompts", type=int, default=500)
    parser.add_argument("--enable_chunked_prefill", action="store_true")
    parser.add_argument("--skip_profile", action="store_true")
    args = parser.parse_args()
    llm = LLM(
        model=LLM_MODEL, 
        tensor_parallel_size=args.tensor_parallel_size, 
        enable_chunked_prefill=args.enable_chunked_prefill, 
        max_num_seqs=args.max_num_seqs, 
        scheduling_policy="priority"
    )
    prompts = get_share_gpt_prompts(num_prompts=args.num_prompts, max_prompt_len=MAX_PROMPT_LEN)
    print_prompt_len_distribution(prompts)
    responses = generate_responses(llm, sampling_params, prompts, skip_profile=args.skip_profile)
