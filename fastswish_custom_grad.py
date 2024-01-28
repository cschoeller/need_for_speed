import torch
import torch.nn as nn


class fastswish_grad(torch.autograd.Function):

    _ALPHA = 3.230769230769231
    _BETA = _ALPHA + 3.

    def __init__(self) -> None:
        super().__init__()
        self._ALPHA = 3.230769230769231
        self._BETA = self._ALPHA + 3.

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x * (x + fastswish_grad._ALPHA)/ fastswish_grad._BETA
        y[x > 3.] = x[x > 3.]
        y[x < -3.] = -1./x[x < -3.]**2
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (2.*x + fastswish_grad._ALPHA)/fastswish_grad._BETA
        y[x > 3.] = 1.
        y[x < -3.] = 2./x[x < -3.]**3
        grad_input = grad_output * y
        return grad_input


class FastSwishCustomGrad(nn.Module):
    """
    Custom grad computation reduces inflated memory footprint back to
    20% over autograd version.
    But it is incompatible with torch dynamo, i.e., torch.compile doesn't result
    in a static graph. And as this version also isn't composed of basic graphable
    operations, it cannot be easily converted to ONNX without a symbolic op. The
    conversion results in an error.
    """

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        # fix amp issues
        if x.dtype == torch.float16:
           x = x.type(torch.float32)
        
        return fastswish_grad.apply(x)
