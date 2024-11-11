# -*- coding: utf-8 -*-
import abc
import torch
import os


def get_world_size():
    world_size = int(os.environ["WORLD_SIZE"])
    return world_size

def make_tensor(size_mb, dtype=torch.bfloat16):
    element_size = torch.tensor(0, dtype=dtype).element_size()
    size = round(1024**2 * size_mb / element_size)
    return torch.zeros(size, dtype=dtype, device="cuda")

class CommunicationPattern(abc.ABC):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        self.message_size_mb = message_size_mb
        self.dtype = dtype

    @property
    @abc.abstractmethod
    def name(self):
        """The name of the pattern."""

    @abc.abstractmethod
    def execute(self):
        """Run the communication pattern once."""

class AllReduce(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        
        super().__init__(message_size_mb, dtype=dtype)
        self.tensor = make_tensor(message_size_mb, dtype=dtype)

    @property
    def name(self):
        return "global_all_reduce"

    def execute(self):
        torch.distributed.all_reduce(self.tensor)


class AllGather(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        input_size = message_size_mb / get_world_size()
        self.input_tensor = make_tensor(input_size, dtype=dtype)
        output_size = self.input_tensor.numel() * get_world_size()
        self.output_tensor = torch.empty(output_size, dtype=dtype, device="cuda")

    @property
    def name(self):
        return "global_all_gather"

    def execute(self):
        torch.distributed.all_gather_into_tensor(self.output_tensor, self.input_tensor)


class ReduceScatter(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):

        super().__init__(message_size_mb, dtype=dtype)
        output_size = message_size_mb / get_world_size()
        self.output_tensor = make_tensor(output_size, dtype=dtype)
        input_size = self.output_tensor.numel() * get_world_size()
        self.input_tensor = torch.zeros(input_size, dtype=dtype, device="cuda")

    @property
    def name(self):
        return "global_reduce_scatter"

    def execute(self):
        torch.distributed.reduce_scatter_tensor(self.output_tensor, self.input_tensor)

class Broadcast(CommunicationPattern):
    def __init__(self, message_size_mb, dtype=torch.bfloat16):
        super().__init__(message_size_mb, dtype=dtype)
        self.input = torch.zeros(message_size_mb * 1000000, dtype=dtype, device="cuda")

    @property
    def name(self):
        return "broadcast"

    def execute(self):
        torch.distributed.broadcast(self.input, 0)
