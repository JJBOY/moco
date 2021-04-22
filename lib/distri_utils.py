import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def can_log():
    return is_main_process()


def dist_print(*args, **kwargs):
    if can_log():
        print(*args, **kwargs)


class DistSummaryWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        if can_log():
            super(DistSummaryWriter, self).__init__(*args, **kwargs)

    def add_text(self, *args, **kwargs):
        if can_log():
            super(DistSummaryWriter, self).add_text(*args, **kwargs)

    def close(self):
        if can_log():
            super(DistSummaryWriter, self).close()

    # def add_scalar(self, *args, **kwargs):
    #     if can_log():
    #         super(DistSummaryWriter, self).add_scalar(*args, **kwargs)
    #
    # def add_figure(self, *args, **kwargs):
    #     if can_log():
    #         super(DistSummaryWriter, self).add_figure(*args, **kwargs)
    #
    # def add_graph(self, *args, **kwargs):
    #     if can_log():
    #         super(DistSummaryWriter, self).add_graph(*args, **kwargs)
    #
    # def add_histogram(self, *args, **kwargs):
    #     if can_log():
    #         super(DistSummaryWriter, self).add_histogram(*args, **kwargs)
    #
    # def add_image(self, *args, **kwargs):
    #     if can_log():
    #         super(DistSummaryWriter, self).add_image(*args, **kwargs)




import tqdm


def dist_tqdm(obj, *args, **kwargs):
    if can_log():
        return tqdm.tqdm(obj, *args, **kwargs)
    else:
        return obj
