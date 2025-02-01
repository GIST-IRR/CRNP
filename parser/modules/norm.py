import numbers
from typing import Tuple, Union, List

import torch
from torch import Size, Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter
import torch.nn as nn

_shape_t = Union[int, List[int], Size]


class MeanOnlyLayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        idx = tuple(range(-1, -(len(self.normalized_shape) + 1), -1))
        res = input - input.mean(idx, keepdim=True)
        if self.elementwise_affine:
            res = res * self.weight + self.bias
        return res

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )
