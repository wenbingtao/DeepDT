import warnings
from itertools import chain

import torch

# modified from
# https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/data_parallel.py


class DeepDTParallel(torch.nn.DataParallel):
    r"""
    Args:
        module (Module): Module to be parallelized.
        device_ids (list of int or torch.device): CUDA devices.
            (default: all devices)
        output_device (int or torch.device): Device location of output.
            (default: :obj:`device_ids[0]`)
    """

    def __init__(self, module, device_ids=None, output_device=None):
        super(DeepDTParallel, self).__init__(module, device_ids, output_device)
        self.src_device = torch.device("cuda:{}".format(self.device_ids[0]))

    def forward(self, data_list):
        """"""
        if len(data_list) == 0:
            warnings.warn('DataParallel received an empty data list, which '
                          'may result in unexpected behaviour.')
            return None

        if len(data_list) > len(self.device_ids):
            raise RuntimeError("The length of data_list must no more than device_ids"
                            "in this implementation.")

        if not self.device_ids or len(self.device_ids) == 1:  # Fallback
            data = data_list[0].to(self.src_device)
            return self.module(data)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device:
                raise RuntimeError(
                    ('Module must have its parameters and buffers on device '
                     '{} but found one of them on device {}.').format(
                         self.src_device, t.device))

        inputs = self.scatter(data_list, self.device_ids)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, None)
        return self.gather(outputs, self.output_device)

    def scatter(self, data_list, device_ids):
        if len(data_list) > len(self.device_ids):
            raise RuntimeError("The length of data_list must no more than device_ids"
                               "in this implementation.")
        return [data_list[i].to(torch.device('cuda:{}'.format(device_ids[i]))) for i in range(len(data_list))]
