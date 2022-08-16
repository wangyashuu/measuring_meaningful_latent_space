import imp
from typing import List
from abc import abstractmethod

from torch import nn, Tensor


class BaseAE(nn.Module):

    def __init__(self) -> None:
        super(BaseAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, *inputs: Tensor, **kwargs) -> Tensor:
        pass
