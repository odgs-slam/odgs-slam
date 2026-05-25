import copy
from queue import Empty

import torch
import torch.multiprocessing as mp


class FakeQueue:
    def put(self, arg):
        del arg

    def get_nowait(self):
        raise Empty
    
    def get(self):
        raise Empty

    def qsize(self):
        return 0

    def empty(self):
        return True


def clone_obj(obj):
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj
