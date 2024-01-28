import torch
import torch.nn as nn
from torch._custom_op import impl as custom_op


class FastSwish(nn.Module):
    """
    Improved version of HardSwish that does not use transcendental functions
    and has no section with zero gradient (avoids dying neurons).

    NOTE: Jit compiling only the function, as well as torch.compile
    seem to not bring any benefit for the speed of this custom activation.
    It seems they can't handle such custom stuff and then don't generate proper
    kernels at all. The only way to make this fast would be to write an explicit
    Trition or CUDA kernel, but even then layer fusion and such optimisations of
    TensorRT or Dynamo would not work probably. Because the internal graph of
    this activation has many nodes, running the network with it also requires 50%
    more VRAM. This can be addressed by implementing a custom backward function.
    """

    def __init__(self) -> None:
        super().__init__()
        self._alpha = 3.230769230769231
        self._beta = self._alpha + 3.
    
    def forward(self, x: torch.Tensor):
        # fix amp issues
        if x.dtype == torch.float16:
           x = x.type(torch.float32)

        y = x * (x + self._alpha)/self._beta
        y[x > 3.] = x[x > 3.]
        y[x < -3.] = -1./x[x < -3.]**2
        return y

