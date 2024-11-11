# -*- coding: utf-8 -*-
import argparse
import functools
import torch
import contextlib
import os
import glob
import torch.distributed as dist
import json
from comm_patterns import AllReduce, AllGather, ReduceScatter, Broadcast

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile")
PATTERN_CLASSES = [
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast,
]

def get_world_size():
    world_size = int(os.environ["WORLD_SIZE"])
    return world_size

def get_local_rank():
    return int(os.environ["LOCAL_RANK"])

def get_global_rank():
    return int(os.environ["RANK"])

def dist_init():
    rank = get_global_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank} local rank {local_rank} world size {world_size}")

    return world_size


def is_leader():
    """Return True if the current process is the leader."""
    return dist.get_rank() == 0


def get_profiler_trace_path():
    log_dir = os.path.join(PROFILE_DIR, "*")
    files = glob.glob(log_dir)
    # assume we are fetching most recent trace
    profiler_trace = max(files, key=os.path.getmtime)

    return profiler_trace

def extract_kernel_runtime(num_iterations=1, trace_dir=None):
    """
    Extracts the average duration of kernel operations from a given trace file.
    """
    file_path = get_profiler_trace_path()
    with open(file_path, "r") as file:
        profiling_data = json.load(file)

    # Filter out "Memcpy DtoH (Device -> Pinned)" events and calculate durations
    event_count = {}
    total_duration = 0
    for event in profiling_data["traceEvents"]:
        if event.get("cat") != "kernel":
            continue
        event_name = event.get("name")
        if event_name not in event_count:
            event_count[event_name] = 0
        event_count[event_name] += 1
        total_duration += (
            event.get("dur", 0) / 1000
        )  # Convert from microseconds to milliseconds

    # Calculate average duration if there are any durations collected
    avg_duration = total_duration / num_iterations

    return avg_duration

def get_profiler_context():
    out_dir = PROFILE_DIR
    os.makedirs(out_dir, exist_ok=True)
    handler = torch.profiler.tensorboard_trace_handler(out_dir)
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    return torch.profiler.profile(
        activities=activities,
        on_trace_ready=handler,
    )


def main():
    args = parse_args()
    world_size = dist_init()
    args.world_size = world_size
    for _ in range(args.num_outer_trials):
        measure_communication_times(args)


def parse_args():
    parser = argparse.ArgumentParser()
    get_dtype = functools.partial(getattr, torch)
    parser.add_argument(
        "--dtype",
        type=get_dtype,
        choices=[torch.bfloat16, torch.float16, torch.float32],
        default=torch.float16,
    )
    parser.add_argument("--num_warmup_trials", type=int, default=8)
    parser.add_argument("--num_inner_trials", type=int, default=32)
    parser.add_argument("--num_outer_trials", type=int, default=1)
    return parser.parse_args()


def measure_communication_times(args):
    MESSAGE_SIZES_MB = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    measurements = {}
    for size_mb in MESSAGE_SIZES_MB:
        if is_leader():
            print("========================================")
            print(f"World size: {args.world_size}")
            print(f"Message size (MiB): {size_mb}")
            print(f"Dtype: {args.dtype}")
            print(f"Num inner trials: {args.num_inner_trials}")
            print(f"Num outer trials: {args.num_outer_trials}")

        times = {}
        for pattern_cls in PATTERN_CLASSES:
            pattern = pattern_cls(size_mb, dtype=args.dtype)
            if is_leader():
                print(f"Benchmarking {pattern.name}")
            for _ in range(args.num_warmup_trials):
                pattern.execute()
            torch.cuda.synchronize()
            if is_leader():
                profiler_ctx = get_profiler_context()
            else:
                profiler_ctx = contextlib.nullcontext()
            with profiler_ctx:
                for _ in range(args.num_inner_trials):
                    pattern.execute()
            torch.cuda.synchronize()
            if not is_leader():
                continue
            avg_comm_time = extract_kernel_runtime(num_iterations=args.num_inner_trials)
            times[pattern.name] = avg_comm_time

        for key, measured_time in times.items():
            if key not in measurements:
                measurements[key] = []
            measurements[key].append(measured_time)

        if is_leader():
            print("Message Sizes (MB)", end="")
            for size in MESSAGE_SIZES_MB:
                print(f", {size}", end="")
            print("")
            for key in measurements.keys():
                time_str = f"{key}(ms)"
                for measured_time in measurements[key]:
                    time_str += f", {measured_time:.2f}"
                print(time_str)


if __name__ == "__main__":
    main()
