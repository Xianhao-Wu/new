"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
from einops import rearrange
import numbers

from functools import partial
from timm.models.layers import DropPath, trunc_normal_
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from selective_scan import selective_scan_fn as selective_scan_fn_v1
from selective_scan import selective_scan_ref as selective_scan_ref_v1
import copy
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True)
    return flops

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class conv3(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(conv3, self).__init__()
        self.conv1 = conv(n_feat, 3, kernel_size, bias=bias)

    def forward(self, x, x_img):

        img = self.conv1(x) + x_img

        return x,img

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.act = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = attn.softmax(dim=-1)
        attn = self.act(attn)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.skip_scale = nn.Parameter(torch.ones(dim))
        self.skip_scale2 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x1 = x.permute(0, 3, 2, 1)
        x2 = x1 * self.skip_scale
        x = x2.permute(0, 3, 2, 1) + self.attn(self.norm1(x))
        x3 = x.permute(0, 3, 2, 1)
        x4 = x3 * self.skip_scale2
        x = x4.permute(0, 3, 2, 1) + self.ffn(self.norm2(x))

        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

########################################################################## MambaBlock
class SS2D_1(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            ssm_ratio=2,
            dt_rank="auto",
            # ======================
            dropout=0.,
            conv_bias=True,
            bias=False,
            dtype=None,
            # ======================
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            shared_ssm=False,
            softmax_version=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": dtype}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 4 if not shared_ssm else 1

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K * inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)  # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)  # (K * D)

        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # channel scan settings
        self.KC = 2
        self.K2C = self.KC

        self.cforward_core = self.cforward_corev1
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.channel_norm = nn.LayerNorm(self.d_inner)

        # cccc
        dc_inner = 4
        self.dtc_rank = 6  # 6
        self.dc_state = 16  # 16
        self.conv_cin = nn.Conv2d(in_channels=1, out_channels=dc_inner, kernel_size=1, stride=1, padding=0)
        self.conv_cout = nn.Conv2d(in_channels=dc_inner, out_channels=1, kernel_size=1, stride=1, padding=0)

        # xc proj ============================
        self.xc_proj = [
            nn.Linear(dc_inner, (self.dtc_rank + self.dc_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.KC)
        ]
        self.xc_proj_weight = nn.Parameter(torch.stack([tc.weight for tc in self.xc_proj], dim=0))  # (K, N, inner)
        del self.xc_proj
        # simple init
        self.Dsc = nn.Parameter(torch.ones((self.K2C * dc_inner)))
        self.Ac_logs = nn.Parameter(
            torch.randn((self.K2C * dc_inner, self.dc_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dtc_projs_weight = nn.Parameter(torch.randn((self.KC, dc_inner, self.dtc_rank)).contiguous())
        self.dtc_projs_bias = nn.Parameter(torch.randn((self.KC, dc_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y

    def forward_corev0_seq(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float()  # (b, k, d, l)
        dts = dts.contiguous().float()  # (b, k, d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
        Ds = self.Ds.float().view(K, -1)  # (k, d)
        dt_projs_bias = self.dt_projs_bias.float().view(K, -1)  # (k, d)

        # assert len(xs.shape) == 4 and len(dts.shape) == 4 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 3 and len(Ds.shape) == 2 and len(dt_projs_bias.shape) == 2

        out_y = []
        for i in range(4):
            yi = self.selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)],
                                 dim=1)  # 一个h,w展开，一个w,h展开，然后堆在一起
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l) #把上面那俩再翻译下，然后堆在一起
            return xs

            # 四个方向

        if self.K == 4:
            # K = 4
            xs = cross_scan_2d(x)  # (b, k, d, l) #[batch_size, 4, channels, height * width]

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L)  # (b, k * d, l)
            dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
            As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
            Ds = self.Ds  # (k * d)
            dt_projs_bias = self.dt_projs_bias.view(-1)  # (k * d)

            # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
            # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
            # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

            out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)
            # assert out_y.dtype == torch.float16

        # if this shows potential, we can raise the speed later by modifying selective_scan
        elif self.K == 1:
            x_dbl = torch.einsum("b d l, c d -> b c l", x.view(B, -1, L), self.x_proj_weight[0])
            # x_dbl = x_dbl + self.x_proj_bias.view(1, -1, 1)
            dt, BC = torch.split(x_dbl, [self.dt_rank, 2 * self.d_state], dim=1)
            dt = torch.einsum("b r l, d r -> b d l", dt, self.dt_projs_weight[0])
            x_dt_BC = torch.cat([x, dt.view(B, -1, H, W), BC.view(B, -1, H, W)], dim=1)  # (b, -1, h, w)

            x_dt_BCs = cross_scan_2d(x_dt_BC)  # (b, k, d, l)
            xs, dts, Bs, Cs = torch.split(x_dt_BCs, [self.d_inner, self.d_inner, self.d_state, self.d_state], dim=2)

            xs = xs.contiguous().view(B, -1, L)  # (b, k * d, l)
            dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
            As = -torch.exp(self.A_logs.float()).repeat(4, 1)  # (k * d, d_state)
            Ds = self.Ds.repeat(4)  # (k * d)
            dt_projs_bias = self.dt_projs_bias.view(-1).repeat(4)  # (k * d)

            # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
            # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

            out_y = self.selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)
            # assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()

        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)

        return y

    forward_core = forward_corev1

    # forward_core = forward_corev0

    def cforward_corev1(self, xc: torch.Tensor):
        self.selective_scanC = selective_scan_fn_v1

        xc = xc.permute(0, 3, 1, 2).contiguous()

        b, d, h, w = xc.shape

        # xc = self.pooling(xc).squeeze(-1).permute(0,2,1).contiguous() #b,1,d, >1!
        # print("xc shape", xc.shape) # 8,1,96
        xc = self.pooling(xc)  # b,d,1,1
        xc = xc.permute(0, 2, 1, 3).contiguous()  # b,1,d,1
        xc = self.conv_cin(xc)  # b,4,d,1
        xc = xc.squeeze(-1)  # b,4,d

        B, D, L = xc.shape  # b,1,c
        D, N = self.Ac_logs.shape  # 2,16
        K, D, R = self.dtc_projs_weight.shape  # 2,1,6

        xsc = torch.stack([xc, torch.flip(xc, dims=[-1])], dim=1)  # input:b,d,l output:b,2,d,l
        # print("xsc shape", xsc.shape) # 8,2,1,96

        xc_dbl = torch.einsum("b k d l, k c d -> b k c l", xsc, self.xc_proj_weight)  # 8,2,1,96; 2,38,1 ->8,2,38,96

        dts, Bs, Cs = torch.split(xc_dbl, [self.dtc_rank, self.dc_state, self.dc_state], dim=2)  # 8,2,38,96-> 6,16,16
        # dts:8,2,6,96 bs,cs:8,2,16,96
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dtc_projs_weight).contiguous()

        xsc = xsc.view(B, -1, L)  # (b, k * d, l) 8,2,96
        dts = dts.contiguous().view(B, -1, L).contiguous()  # (b, k * d, l) 8,2,96
        As = -torch.exp(self.Ac_logs.float())  # (k * d, d_state) 2,16
        Ds = self.Dsc  # (k * d) 2
        dt_projs_bias = self.dtc_projs_bias.view(-1)  # (k * d)2

        out_y = self.selective_scanC(
            xsc, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, 2, -1, L)

        y = out_y[:, 0].float() + torch.flip(out_y[:, 1], dims=[-1]).float()

        # y: b,4,d
        y = y.unsqueeze(-1)  # b,4,d,1
        y = self.conv_cout(y)  # b,1,d,1

        y = y.transpose(dim0=2, dim1=3).contiguous()  # b,1,1,d
        y = self.channel_norm(y)
        # before norm: input 222 shape torch.Size([8, 1, 1, 48])  ; output:333 shape torch.Size([8, 48, 1, 48])
        y = y.to(xc.dtype)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        # input x:b,c,h,w
        x = x.permute(0, 2, 3, 1).contiguous()  # b,h,w,c
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, c)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y = self.forward_core(x)
        y = y * F.silu(z)  # b,h,w,c

        # channel attention
        c = self.cforward_core(y)  # x:b,h,w,d; output:b,1,1,d
        y2 = y * c
        y = y + y2

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out

class MamberBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(MamberBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        #self.attn = Attention(dim, num_heads, bias)
        self.attn = SS2D_1(d_model=dim,ssm_ratio=1)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################

class model(nn.Module):
    def __init__(self,
                 in_c=3,
                 out_c=3,
                 n_feat=40,
                 scale_unetfeats=20,
                 scale_orsnetfeats=16,
                 num_cab=8,
                 kernel_size=3,
                 reduction=4,
                 bias=False,
                 inp_channels=3,
                 out_channels=3,
                 dim=40,
                 num_blocks=[2, 2, 2],
                 heads=[1, 2, 4],
                 x_21_channels=6,
                 reduced_channels = 3,
                 x_3_channels=6,
                 reduced1_channels=3,
                 ffn_expansion_factor=2.66,
                 LayerNorm_type='WithBias',):
        super(model, self).__init__()

        #self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)

        self.conv3_12 = conv3(n_feat, kernel_size=3, bias=bias)
        self.conv3_23 = conv3(n_feat, kernel_size=3, bias=bias)
        ##################################################################################### #ADD  START ZHL
        self.conv3_34 = conv3(3, kernel_size=3, bias=bias)
        self.conv3_45 = conv3(3, kernel_size=3, bias=bias)
        self.conv3_56 = conv3(3, kernel_size=3, bias=bias)
        #######################################################################################ADD  END  ZHL

        self.concat12  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.concat23  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.tail     = conv(n_feat+scale_orsnetfeats, out_c, kernel_size, bias=bias)
        ######################################################################################################\
        ##stage1 ZHL
        self.patch_embed_stage1= OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1_stage1 = nn.Sequential(*[MamberBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2_stage1 = Downsample(dim)
        self.encoder_level2_stage1 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3_stage1 = Downsample(int(dim * 2 ** 1))
        self.latent_stage1 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2_stage1 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2_stage1 = nn.Conv2d(int(160), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2_stage1 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1_stage1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1_stage1 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.reduce_chan_level2_stage1_2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1 ** 1), kernel_size=1, bias=bias)

        ##stage2 ZHL
        self.patch_embed_stage2 = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1_stage2 = nn.Sequential(*[MamberBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2_stage2 = Downsample(dim)
        self.encoder_level2_stage2 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3_stage2 = Downsample(int(dim * 2 ** 1))
        self.latent_stage2 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2_stage2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2_stage2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2_stage2 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1_stage2 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1_stage2 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.reduce_chan_level2_stage2_2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 1 ** 1), kernel_size=1, bias=bias)

        ##stage3 ZHL
        self.patch_embed_stage3 = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1_stage3 = nn.Sequential(*[MamberBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2_stage3 = Downsample(dim)
        self.encoder_level2_stage3 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3_stage3 = Downsample(int(dim * 2 ** 1))
        self.latent_stage3 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2_stage3 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2_stage3 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2_stage3 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1_stage3 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1_stage3 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.reduce_chan_level2_stage3_2 = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=1, bias=bias)
        ######################################################################################################
        ##stage4 ZHL
        self.patch_embed_stage4 = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1_stage4 = nn.Sequential(*[MamberBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2_stage4 = Downsample(dim)
        self.encoder_level2_stage4 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3_stage4 = Downsample(int(dim * 2 ** 1))
        self.latent_stage4 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2_stage4 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2_stage4 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2_stage4 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1_stage4 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1_stage4 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.reduce_chan_level2_stage4_2 = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=1, bias=bias)
        ######################################################################################################
        ##stage5 ZHL
        self.patch_embed_stage5 = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1_stage5 = nn.Sequential(*[MamberBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2_stage5 = Downsample(dim)
        self.encoder_level2_stage5 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3_stage5 = Downsample(int(dim * 2 ** 1))
        self.latent_stage5 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2_stage5 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2_stage5 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2_stage5 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1_stage5 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1_stage5 = nn.Sequential(*[MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.reduce_chan_level2_stage5_2 = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=1, bias=bias)
        ######################################################################################################
        self.reduction_conv = nn.Conv2d(x_21_channels, reduced_channels, kernel_size=1)
        self.reduction_conv1 = nn.Conv2d(x_3_channels, reduced1_channels, kernel_size=1)
    def forward(self, x3_img):

        H = x3_img.size(2)
        W = x3_img.size(3)


        x2top_img  = x3_img[:,:,0:int(H/2),:]
        x2bot_img  = x3_img[:,:,int(H/2):H,:]

        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:,:,:,0:int(W/2)]
        x1rtop_img = x2top_img[:,:,:,int(W/2):W]
        x1lbot_img = x2bot_img[:,:,:,0:int(W/2)]
        x1rbot_img = x2bot_img[:,:,:,int(W/2):W]


        x_2 = F.interpolate(x3_img, scale_factor=0.5)

        x_4 = F.interpolate(x_2, scale_factor=0.5)

        x_41 = self.patch_embed_stage5(x_4)

        x_41_1 = self.encoder_level1_stage5(x_41)  # Encoder
        x_41 = self.down1_2_stage5(x_41_1)
        x_41_2 = self.encoder_level2_stage5(x_41)
        x_41 = self.down2_3_stage5(x_41_2)
        x_41 = self.latent_stage5(x_41)
        x_41 = self.up3_2_stage5(x_41)           # Decoder
        x_41 = torch.cat([x_41, x_41_2], 1)
        x_41 = self.reduce_chan_level2_stage5(x_41)
        x_41 = self.decoder_level2_stage5(x_41)
        x_41 = self.up2_1_stage5(x_41)
        x_41 = torch.cat([x_41, x_41_1], 1)
        x_41= self.decoder_level1_stage5(x_41)
        x_41 = self.reduce_chan_level2_stage5_2(x_41)

        feat41 = x_41

        _,Stage5_img = self.conv3_34(feat41,x_4)

        x_2_up = F.interpolate(Stage5_img, scale_factor=2)

        x_21_add = torch.add(x_2_up, x_2)
        x_21 = self.patch_embed_stage4(x_21_add)

        x_21_1 = self.encoder_level1_stage4(x_21)
        x_21 = self.down1_2_stage4(x_21_1)
        x_21_2 = self.encoder_level2_stage4(x_21)
        x_21 = self.down2_3_stage4(x_21_2)
        x_21 = self.latent_stage4(x_21)
        x_21 = self.up3_2_stage4(x_21)  # Decoder
        x_21 = torch.cat([x_21, x_21_2], 1)
        x_21 = self.reduce_chan_level2_stage4(x_21)
        x_21 = self.decoder_level2_stage4(x_21)
        x_21 = self.up2_1_stage4(x_21)
        x_21 = torch.cat([x_21, x_21_1], 1)
        x_21 = self.decoder_level1_stage4(x_21)
        x_21 = self.reduce_chan_level2_stage4_2(x_21)
        ###### print("end decoder")######
        feat21 = x_21

        _,Stage4_img = self.conv3_45(feat21, x_2)

        x_3_up = F.interpolate(Stage4_img, scale_factor=2)


        x1ltop = self.patch_embed_stage1(x1ltop_img)
        x1rtop = self.patch_embed_stage1(x1rtop_img)
        x1lbot = self.patch_embed_stage1(x1lbot_img)
        x1rbot = self.patch_embed_stage1(x1rbot_img)

        feat1_ltop_1 = self.encoder_level1_stage1(x1ltop)
        feat1_ltop = self.down1_2_stage1(feat1_ltop_1)
        feat1_ltop_2 = self.encoder_level2_stage1(feat1_ltop)
        feat1_ltop = self.down2_3_stage1(feat1_ltop_2)
        feat1_ltop = self.latent_stage1(feat1_ltop)
        feat1_ltop = self.up3_2_stage1(feat1_ltop)
        feat1_ltop = torch.cat([feat1_ltop, feat1_ltop_2 ], 1)

        feat1_ltop = self.reduce_chan_level2_stage1(feat1_ltop)
        feat1_ltop = self.decoder_level2_stage1(feat1_ltop)
        feat1_ltop = self.up2_1_stage1(feat1_ltop)
        feat1_ltop = torch.cat([feat1_ltop, feat1_ltop_1], 1)
        feat1_ltop = self.decoder_level1_stage1(feat1_ltop)
        feat1_ltop = self.reduce_chan_level2_stage1_2(feat1_ltop)

        feat1_rtop_1 = self.encoder_level1_stage1(x1rtop)
        feat1_rtop= self.down1_2_stage1(feat1_rtop_1)
        feat1_rtop_2 = self.encoder_level2_stage1(feat1_rtop)
        feat1_rtop = self.down2_3_stage1(feat1_rtop_2)
        feat1_rtop = self.latent_stage1(feat1_rtop)
        feat1_rtop = self.up3_2_stage1(feat1_rtop)
        feat1_rtop = torch.cat([feat1_rtop, feat1_rtop_2], 1)
        feat1_rtop= self.reduce_chan_level2_stage1(feat1_rtop)
        feat1_rtop = self.decoder_level2_stage1(feat1_rtop)
        feat1_rtop = self.up2_1_stage1(feat1_rtop)
        feat1_rtop = torch.cat([feat1_rtop, feat1_rtop_1], 1)
        feat1_rtop = self.decoder_level1_stage1(feat1_rtop)
        feat1_rtop = self.reduce_chan_level2_stage1_2(feat1_rtop)

        feat1_lbot_1 = self.encoder_level1_stage1(x1lbot)
        feat1_lbot = self.down1_2_stage1(feat1_lbot_1)
        feat1_lbot_2 = self.encoder_level2_stage1(feat1_lbot)
        feat1_lbot = self.down2_3_stage1(feat1_lbot_2)
        feat1_lbot = self.latent_stage1(feat1_lbot)
        feat1_lbot = self.up3_2_stage1(feat1_lbot)
        feat1_lbot = torch.cat([feat1_lbot, feat1_lbot_2], 1)
        feat1_lbot = self.reduce_chan_level2_stage1(feat1_lbot)
        feat1_lbot = self.decoder_level2_stage1(feat1_lbot)
        feat1_lbot = self.up2_1_stage1(feat1_lbot)
        feat1_lbot = torch.cat([feat1_lbot, feat1_lbot_1], 1)
        feat1_lbot = self.decoder_level1_stage1(feat1_lbot)
        feat1_lbot = self.reduce_chan_level2_stage1_2(feat1_lbot)

        feat1_rbot_1 = self.encoder_level1_stage1(x1rbot)
        feat1_rbot = self.down1_2_stage1(feat1_rbot_1)
        feat1_rbot_2 = self.encoder_level2_stage1(feat1_rbot)
        feat1_rbot = self.down2_3_stage1(feat1_rbot_2)
        feat1_rbot = self.latent_stage1(feat1_rbot)
        feat1_rbot = self.up3_2_stage1(feat1_rbot)
        feat1_rbot = torch.cat([feat1_rbot, feat1_rbot_2], 1)
        feat1_rbot = self.reduce_chan_level2_stage1(feat1_rbot)
        feat1_rbot = self.decoder_level2_stage1(feat1_rbot)
        feat1_rbot = self.up2_1_stage1(feat1_rbot)
        feat1_rbot = torch.cat([feat1_rbot, feat1_rbot_1], 1)
        feat1_rbot = self.decoder_level1_stage1(feat1_rbot)
        feat1_rbot = self.reduce_chan_level2_stage1_2(feat1_rbot)


        feat1_top = torch.cat((feat1_ltop,feat1_rtop), 3)
        feat1_bot = torch.cat((feat1_lbot,feat1_rbot), 3)


        res1_top = feat1_top
        res1_bot = feat1_bot

        x2top_samfeats, stage1_img_top = self.conv3_12(res1_top, x2top_img)
        x2bot_samfeats, stage1_img_bot = self.conv3_12(res1_bot, x2bot_img)


        stage1_img = torch.cat([stage1_img_top, stage1_img_bot],2)

        x2top  = self.patch_embed_stage2(x2top_img)
        x2bot  = self.patch_embed_stage2(x2bot_img)

        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        feat2_top_1 = self.encoder_level1_stage2(x2top_cat)
        feat2_top = self.down1_2_stage2(feat2_top_1)
        feat2_top_2 = self.encoder_level2_stage2(feat2_top)
        feat2_top = self.down2_3_stage2(feat2_top_2)
        feat2_top = self.latent_stage2(feat2_top)
        feat2_top = self.up3_2_stage2(feat2_top)
        feat2_top = torch.cat([feat2_top, feat2_top_2], 1)
        feat2_top = self.reduce_chan_level2_stage2(feat2_top)
        feat2_top = self.decoder_level2_stage2(feat2_top)
        feat2_top = self.up2_1_stage2(feat2_top)
        feat2_top = torch.cat([feat2_top, feat2_top_1], 1)
        feat2_top = self.decoder_level1_stage2(feat2_top)
        feat2_top = self.reduce_chan_level2_stage2_2(feat2_top)

        feat2_bot_1 = self.encoder_level1_stage2(x2bot_cat)
        feat2_bot  = self.down1_2_stage2(feat2_bot_1)
        feat2_bot_2 = self.encoder_level2_stage2(feat2_bot)
        feat2_bot  = self.down2_3_stage2(feat2_bot_2)
        feat2_bot  = self.latent_stage2(feat2_bot)
        feat2_bot  = self.up3_2_stage2(feat2_bot)
        feat2_bot  = torch.cat([feat2_bot, feat2_bot_2], 1)
        feat2_bot  = self.reduce_chan_level2_stage2(feat2_bot)
        feat2_bot  = self.decoder_level2_stage2(feat2_bot)
        feat2_bot  = self.up2_1_stage2(feat2_bot)
        feat2_bot  = torch.cat([feat2_bot, feat2_bot_1], 1)
        feat2_bot  = self.decoder_level1_stage2(feat2_bot)
        feat2_bot  = self.reduce_chan_level2_stage2_2(feat2_bot)

        feat2 = torch.cat((feat2_top,feat2_bot), 2)

        res2 = feat2

        x3_samfeats, stage2_img = self.conv3_23(res2, x3_img)

        x_3_add = torch.add(x_3_up, x3_img)
        x3 = self.patch_embed_stage3(x_3_add)

        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))

        x3_cat_1 = self.encoder_level1_stage3(x3_cat)
        x3_cat = self.down1_2_stage3(x3_cat_1)
        x3_cat_2 = self.encoder_level2_stage3(x3_cat)
        x3_cat = self.down2_3_stage3(x3_cat_2)
        x3_cat = self.latent_stage3(x3_cat)
        x3_cat = self.up3_2_stage3(x3_cat)
        x3_cat = torch.cat([x3_cat, x3_cat_2], 1)
        x3_cat = self.reduce_chan_level2_stage3(x3_cat)
        x3_cat = self.decoder_level2_stage3(x3_cat)
        x3_cat = self.up2_1_stage3(x3_cat)
        x3_cat = torch.cat([x3_cat, x3_cat_1], 1)
        x3_cat = self.decoder_level1_stage3(x3_cat)
        x3_cat = self.reduce_chan_level2_stage3_2(x3_cat)

        feat3 = x3_cat
        _,Stage3_img = self.conv3_56(feat3, x3_img)

        return [Stage3_img, stage2_img, stage1_img,Stage4_img,Stage5_img]


if __name__ == '__main__':
    input = torch.rand(1, 3, 256, 256)
    model = model()
    # output = model(input)

    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    flops = FlopCountAnalysis(model, input)

    print("FLOPs: ", flops.total())
    print(parameter_count_table(model))