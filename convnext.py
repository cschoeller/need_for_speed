import torch
import torch.nn as nn

#from torch.onnx import register_custom_op_symbolic
#import onnxscript

def to_pair(x):
    if type(x) == tuple:
        return x
    return (x,x)








# custom_opset = onnxscript.values.Opset(domain="onnx-script", version=1)

# @onnxscript.script(custom_opset)
# def Selu(X):
#     alpha = 1.67326  # auto wrapped as Constants
#     gamma = 1.0507
#     alphaX = op.CastLike(alpha, X)
#     gammaX = op.CastLike(gamma, X)
#     neg = gammaX * (alphaX * op.Exp(X) - alphaX)
#     pos = gammaX * X
#     zero = op.CastLike(0, X)
#     return op.Where(X <= zero, neg, pos)

# # setType API provides shape/type to ONNX shape/type inference
# def fastswishop(g: jit_utils.GraphContext, X):
#     return g.onnxscript_op(Selu, X).setType(X.type())

# # Register custom symbolic function
# # There are three opset version needed to be aligned
# # This is (2) the opset version in registry
# torch.onnx.register_custom_op_symbolic(
#     symbolic_name="aten::fastswish",
#     symbolic_fn=fastswishop,
#     opset_version=opset_version,
# )







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
        self.alpha = 3.230769230769231
        self.beta = self.alpha + 3.
    
    def forward(self, x: torch.Tensor):
        # fix amp issues
        if x.dtype == torch.float16:
           x = x.type(torch.float32)

        y = x * (x + self.alpha)/ self.beta
        y[x > 3.] = x[x > 3.]
        y[x < -3.] = -1./x[x < -3.]**2
        return y
    

class CXBlock(nn.Module):
    """ Inverted bottleneck module of the ConvNext architecture. """
    def __init__(self, dim, h, w):
        super().__init__()
        layers = [
            nn.Conv2d(dim, 4 * dim, kernel_size=7, groups=dim, padding='same'),
            nn.LayerNorm([4 * dim, h, w]),
            nn.Conv2d(4 * dim, 4 * dim, kernel_size=1, padding='same'),
            #nn.GELU(approximate='tanh'),
            nn.ReLU(),
            #nn.Hardswish(),
            #FastSwishCustomGrad(),
            #FastSwish(),
            #nn.SiLU(),
            nn.Conv2d(4 * dim, dim, kernel_size=1, padding='same'),
        ]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.net(x)

class ConvNext(nn.Module):
    """"
    ConvNext from the paper 'A ConvNet for the 2020s', Liu et al., 2022.

    NOTE: Stochastic depth, EMA parameter averaging, and layer scale not implemented.
    """

    def __init__(self, img_size, num_classes, img_ch=3, blocks=(3,3,9,3), channels=(96, 192, 384, 768)):
        super().__init__()
        assert(len(blocks) == len(channels))

        H, W = to_pair(img_size)
        H, W = H//4, W//4 # internal after stem
        assert(H%4 == 0 and W%4 == 0)

        # input stem layer
        self.stem = nn.Sequential(*[
            nn.Conv2d(img_ch, channels[0], kernel_size=4, stride=4),
            nn.LayerNorm([channels[0], H, W])
            ])
        
        # intermediate inverted bottleneck blocks and downsampling
        cx_blocks = []
        for i, (b, ch) in enumerate(zip(blocks, channels)):
             cx_blocks.extend([CXBlock(ch, H//(2**i), W//(2**i)) for j in range(b)])
             if i < len(channels) - 1 : # downsampling
                cx_blocks.append(nn.LayerNorm([ch, H//(2**i), W//(2**i)]))
                cx_blocks.append(nn.Conv2d(ch, channels[i + 1], kernel_size=2, stride=2))
        self.cx_blocks = nn.Sequential(*cx_blocks)

        # output layer
        self.out = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], num_classes)
            ])

    def forward(self, x):
        x = self.stem(x)
        x = self.cx_blocks(x)
        x = self.out(x)
        return x
