import os
import argparse
import torch
import time
import asyncio
import math

import matplotlib.pyplot as plt
import numpy as np

from datasets import load_dataset
from multiprocessing import Process, Queue
from vllm import LLM, AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from vllm.transformers_utils.config import get_config, get_hf_text_config

# vllm settings
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"
# prompt settings
MAX_PROMPT_LEN = 512


def get_share_gpt_prompts(num_prompts=10000, max_prompt_len=8192):
    share_gpt_dataset = load_dataset("theblackcat102/sharegpt-english")
    filter_dataset = share_gpt_dataset.data["train"]['conversations']
    prompts = [str(item[0]["text"]) for item in filter_dataset]
    prompts = [prompt for prompt in prompts if len(prompt) <= max_prompt_len]
    prompts = prompts[:num_prompts]
    return prompts


def print_prompt_len_distribution(prompts):
    if len(prompts) == 0:
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


def generate_responses(args, sampling_params, prompts, profile=False):
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        num_scheduler_steps=args.num_scheduler_steps,
        multi_step_stream_outputs=args.multi_step_stream_outputs,
        block_size=args.block_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype
    )

    if profile:
        llm.start_profile()

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    torch.cuda.synchronize()

    e2e_time = end_time - start_time
    print(f"Total time: {e2e_time:.2f} seconds")

    if profile:
        print(f"Generating profile")
        llm.stop_profile()

    return outputs


async def set_partitions(args):
    if args.pp_partition_ratio is None:
        return

    pp_partitions = [float(layer) for layer in args.pp_partition_ratio.split(",")]
    hf_text_config = get_hf_text_config(get_config(args.model_name, True))
    total_num_hidden_layers = getattr(hf_text_config, "num_hidden_layers", 0)
    print(f"Total number of hidden layers: {total_num_hidden_layers}")

    scaled_partitions = [int(round(total_num_hidden_layers * partition)) for partition in pp_partitions]
    scaled_partitions[-1] = total_num_hidden_layers - sum(scaled_partitions[:-1])

    os.environ["VLLM_PP_LAYER_PARTITION"] = ",".join([str(partition) for partition in scaled_partitions])

    print(f"Set partitions: {os.environ['VLLM_PP_LAYER_PARTITION']}")


async def get_async_llm_engine(args, sampling_params, prompts, profile=False):
    """Generate responses using asynchronous LLM."""
    engine_args = AsyncEngineArgs(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        num_scheduler_steps=args.num_scheduler_steps,
        multi_step_stream_outputs=args.multi_step_stream_outputs,
        block_size=args.block_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        distributed_executor_backend=args.distributed_executor_backend,
        dtype=args.dtype
    )
    await set_partitions(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    requests = [{"prompt": prompt, "stream": False, "request_id": i + 1} for i, prompt in enumerate(prompts)]

    if profile:
        await engine.start_profile()

    async def process_request(request):
        results = []
        async for output in engine.generate(request["prompt"], sampling_params, str(request["request_id"])):
            results.append(output)
        # only return last result, which is the final output
        return results[-1]

    start_time = time.time()
    # Gather all results asynchronously
    final_outputs = await asyncio.gather(*(process_request(req) for req in requests))
    end_time = time.time()
    time_taken = end_time - start_time
    # print throughput
    print(f"Throughput: {len(prompts) / time_taken:.4f} requests/second")

    if profile:
        await engine.stop_profile()

    return final_outputs

def print_metrics(args, outputs):
    start_time = math.inf
    end_time = 0
    ttft = 0.0
    schedule_delay = 0.0
    tpot = 0.0
    responses = 0
    input_lens = []
    output_lens = []
    for output in outputs:
        time_to_first_token = output.metrics.first_token_time - output.metrics.arrival_time
        schedule_delay += output.metrics.scheduler_time
        ttft += time_to_first_token
        total_tokens = 0
        input_lens.append(len(output.prompt_token_ids))
        for completion_output in output.outputs:
            total_tokens += len(completion_output.token_ids)
            if args.print_outputs:
                print(f"*** Prompt {responses}: {output.prompt}")
                print(f"- Reponse {responses}: {completion_output.text}")
                print(
                    f"- Prompt Tokens: {len(output.prompt_token_ids)}, Response Tokens: {len(completion_output.token_ids)}")
            # compute mean median min max of input and outputs lens
            output_lens.append(len(completion_output.token_ids))
            responses += 1
        tpot += (output.metrics.finished_time - output.metrics.first_token_time) / total_tokens
        start_time = min(start_time, output.metrics.arrival_time)
        end_time = max(end_time, output.metrics.finished_time)
    input_lens = np.array(input_lens)
    output_lens = np.array(output_lens)

    avg_ttft = ttft / len(outputs)
    avg_tpot = tpot / len(outputs)
    avg_schedule_delay = schedule_delay / len(outputs)
    e2e_time = end_time - start_time
    throughput = len(prompts) / e2e_time
    print(
        f"Input length stats: mean={np.mean(input_lens):.2f}, median={np.median(input_lens):.2f}, min={np.min(input_lens)}, max={np.max(input_lens)}")
    print(
        f"Output length stats: mean={np.mean(output_lens):.2f}, median={np.median(output_lens):.2f}, min={np.min(output_lens)}, max={np.max(output_lens)}")
    print(f"Average time to first token: {avg_ttft:.4f} seconds")
    print(f"Average time per output token: {avg_tpot:.4f} seconds")
    print(f"Average schedule delay: {avg_schedule_delay:.4f} seconds")
    print(f"End-to-end time: {e2e_time:.4f} seconds")
    print(f"Throughput: {throughput:.4f} requests/second")
    print("=============================")
    print(f"Requests,MaxSeqs,SchedulerSteps,MaxTokens,ChunkedPrefill,GPUUtil,BlockSize,TTFT,TTPOT,Schedule_Delay,E2E_Time")
    print(
        f"{len(prompts)},{args.max_num_seqs},{args.num_scheduler_steps},{args.max_tokens},{args.enable_chunked_prefill},{args.gpu_memory_utilization},{args.block_size},{avg_ttft:.4f},{avg_tpot:.4f},{avg_schedule_delay:.4f},{e2e_time:.4f}")
    print("=============================")

def vllm_instance(worker_id, args, prompts, output_queue):
    num_gpus = args.tensor_parallel_size * args.pipeline_parallel_size
    gpu_ids = ",".join([f"{worker_id * num_gpus + i}" for i in range(num_gpus)])
    print(f"Worker {worker_id} started on GPUs: {gpu_ids}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    run_async = False
    if args.pipeline_parallel_size > 1:
        run_async = True
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=args.max_tokens)
    print_prompt_len_distribution(prompts)
    # TODO: separate LLM engine instantiation logic from prompt serving
    if run_async:
        responses = asyncio.run(get_async_llm_engine(args, sampling_params, prompts, profile=args.profile))
    else:
        responses = generate_responses(args, sampling_params, prompts, profile=args.profile)
    output_queue.put((worker_id, responses))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B",
                        help="meta-llama/Llama-3.1-8B, TinyLlama/TinyLlama_v1.1")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--num_prompts", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--enable_chunked_prefill", action="store_true")
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--num_scheduler_steps", type=int, default=1)
    parser.add_argument("--multi_step_stream_outputs", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_prompt_len", type=int, default=MAX_PROMPT_LEN)
    parser.add_argument("--distributed_executor_backend", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--pp_partition_ratio", type=str, default=None)
    parser.add_argument("--print_outputs", action="store_true")
    args = parser.parse_args()
    print(f" *** Args: {args}")

    output_queue = Queue()
    prompts = get_share_gpt_prompts(num_prompts=args.num_prompts, max_prompt_len=args.max_prompt_len)
    prompt_splits = np.array_split(prompts, args.data_parallel_size)
    processes = []
    # run separate process for each vllm instance
    for i in range(args.data_parallel_size):
        p = Process(target=vllm_instance, args=(i, args, prompt_splits[i], output_queue))
        processes.append(p)
        p.start()

    # Collect results
    outputs = []
    for _ in range(args.data_parallel_size):
        process_id, responses = output_queue.get()
        outputs.append((process_id, responses))

    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    all_responses = [output for _, proc_output in outputs for output in proc_output]
    print_metrics(args, all_responses)