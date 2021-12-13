import math
import numpy as np

import paddle
import paddle.nn as nn
import os
import cv2
import paddle.vision.transforms as T
import paddle.distributed as dist
from paddle.io import DistributedBatchSampler, DataLoader
from scheduler import WarmupCosineScheduler
from PIL import Image
from paddle.vision import transforms

from common import Attention as Attention_Pure 
from common import Unfold  
from common import add_parameter  
from common import DropPath, Identity, Mlp  
from common import orthogonal_, trunc_normal_, zeros_, ones_ 

from dataset import get_dataset

def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoid position encoding table"""

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return sinusoid_table[None, ...].astype("float32")


class Token_performer(nn.Layer):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2=0.1):
        super().__init__()
        self.emb = in_dim * head_cnt  # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        self.w = paddle.randn((self.m, self.emb))

        self.w = add_parameter(self, orthogonal_(self.w) * math.sqrt(self.m))

    def prm_exp(self, x):
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(axis=-1, keepdim=True)).tile([1, 1, self.m]) / 2
        wtx = paddle.mm(x, self.w.transpose((1, 0)))

        return paddle.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        x = self.kqv(x)
        k, q, v = paddle.split(x, x.shape[-1] // self.emb, axis=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)

        # (B, T, m) * (B, m) -> (B, T, 1)
        D = paddle.bmm(qp, kp.sum(axis=1).unsqueeze(axis=-1))
        kptv = paddle.bmm(v.astype("float32").transpose((0, 2, 1)), kp)  # (B, emb, m)
        y = paddle.bmm(qp, kptv.transpose((0, 2, 1))) / (
            D.tile([1, 1, self.emb]) + self.epsilon
        )  # (B, T, emb) / Diag

        # skip connection
        # same as token_transformer in T2T layer, use v as skip connection
        y = v + self.dp(self.proj(y))

        return y

    def forward(self, x):
        x = self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        in_dim=None,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape((B, N, 3, self.num_heads, self.in_dim))
            .transpose((2, 0, 3, 1, 4))
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, self.in_dim))
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        # because the original x has different size with current x, use v to do skip connection
        x = v.squeeze(1) + x

        return x


class Token_transformer(nn.Layer):
    def __init__(
        self,
        dim,
        in_dim,
        num_heads,
        mlp_ratio=1.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim,
            in_dim=in_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

        self.norm2 = norm_layer(in_dim)

        self.mlp = Mlp(
            in_features=in_dim,
            hidden_features=int(in_dim * mlp_ratio),
            out_features=in_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Block(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention_Pure(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class T2T_Layer(nn.Layer):
    """
    Tokens-to-Token encoding module
    """

    def __init__(
        self,
        img_size=224,
        tokens_type="performer",
        in_chans=3,
        embed_dim=768,
        token_dim=64,
    ):
        super().__init__()

        if tokens_type == "transformer":
            self.soft_split0 = Unfold(kernel_size=[7, 7], stride=[4, 4], padding=[2, 2])
            self.soft_split1 = Unfold(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])
            self.soft_split2 = Unfold(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])

            self.attention1 = Token_transformer(
                dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0
            )
            self.attention2 = Token_transformer(
                dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0
            )
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == "performer":
            self.soft_split0 = Unfold(kernel_size=[7, 7], stride=[4, 4], padding=[2, 2])
            self.soft_split1 = Unfold(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])
            self.soft_split2 = Unfold(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1])

            self.attention1 = Token_performer(
                dim=in_chans * 7 * 7, in_dim=token_dim, kernel_ratio=0.5
            )
            self.attention2 = Token_performer(
                dim=token_dim * 3 * 3, in_dim=token_dim, kernel_ratio=0.5
            )
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif (
            tokens_type == "convolution"
        ):  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            self.soft_split0 = nn.Conv2D(
                3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2)
            )  # the 1st convolution
            self.soft_split1 = nn.Conv2D(
                token_dim, token_dim
                , kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )  # the 2nd convolution
            self.project = nn.Conv2D(
                token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )  # the 3rd convolution

        self.num_patches = (img_size // (4 * 2 * 2)) * (
            img_size // (4 * 2 * 2)
        )  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose((0, 2, 1))

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose((0, 2, 1)).reshape(
            (B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        )

        # iteration1: soft split
        x = self.soft_split1(x).transpose((0, 2, 1))

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose((0, 2, 1)).reshape(
            (B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        )

        # iteration2: soft split
        x = self.soft_split2(x).transpose((0, 2, 1))

        # final tokens
        x = self.project(x)

        return x


class T2T_ViT(nn.Layer):
    def __init__(
        self,
        img_size=224,
        tokens_type="performer",
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        token_dim=64,
        class_dim=1000,
    ):
        super().__init__()
        self.class_dim = class_dim
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_Layer(
            img_size=img_size,
            tokens_type=tokens_type,
            in_chans=in_chans,
            embed_dim=embed_dim,
            token_dim=token_dim,
        )

        num_patches = self.tokens_to_token.num_patches

        self.cls_token = add_parameter(self, paddle.zeros((1, 1, embed_dim)))
        self.pos_embed = add_parameter(
            self, get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)  # stochastic depth decay rule
        self.blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        if class_dim > 0:
            self.head = nn.Linear(embed_dim, class_dim)

        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)

        if self.class_dim > 0:
            x = self.head(x)

        return x


def t2t_vit_7(pretrained=False, **kwargs):
    model = T2T_ViT(
        tokens_type='performer', embed_dim=256, depth=7, 
        num_heads=4, mlp_ratio=2., **kwargs
    )
    if pretrained:
        params = paddle.load('./output/t2t_vit_7.pdparams')
        model.set_dict(params)
    return model