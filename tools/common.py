import numpy as np

import paddle
import paddle.nn as nn

from paddle.nn.initializer import TruncatedNormal, KaimingNormal, Constant, Assign


# Common initializations
ones_ = Constant(value=1.)
zeros_ = Constant(value=0.)
kaiming_normal_ = KaimingNormal()
trunc_normal_ = TruncatedNormal(std=.02)


def orthogonal_(tensor, gain=1):
    r"""Fills the input `Tensor` with a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.orthogonal_(w)
    """
    if tensor.ndimension() < 2:
        raise ValueError(
            "Only tensors with 2 or more dimensions are supported")

    if  paddle.fluid.data_feeder.convert_dtype(tensor.dtype) != 'float32':
        raise ValueError(
            "Only tensors in float32 dtype are supported")

    rows = tensor.shape[0]
    cols = tensor.numel() // rows
    flattened = np.random.randn(rows, cols).astype('float32')

    if rows < cols:
        flattened = flattened.T

    # Compute the qr factorization
    q, r = np.linalg.qr(flattened)

    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = np.diag(r, 0)
    ph = paddle.sign(paddle.to_tensor(d))
    q = paddle.to_tensor(q) * ph

    if rows < cols:
        q.t()

    with paddle.no_grad():
        tensor.reshape(q.shape).set_value(q * gain)

    return tensor


# Common Functions
def load_model(model, url):
    path = paddle.utils.download.get_weights_path_from_url(url)
    model.set_state_dict(paddle.load(path))
    return model


def to_2tuple(x):
    return tuple([x] * 2)


def add_parameter(layer, datas, name=None):
    parameter = layer.create_parameter(
        shape=(datas.shape),
        default_initializer=Assign(datas)
    )
    if name:
        layer.add_parameter(name, parameter)
    return parameter


# Common Layers
def drop_path(x, drop_prob=0., training=False):
    """
        Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # B= paddle.shape(x)[0]
        N, C = x.shape[1:]
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Unfold(nn.Layer):
    '''
    Fix the bug of nn.Unfold
    Will be updated sonn.
    '''
    def __init__(self,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 name=None):
        super(Unfold, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.name = name

    def forward(self, input):
        return nn.functional.unfold(
            input, self.kernel_size, self.stride, 
            self.padding, self.dilation, self.name
        )
