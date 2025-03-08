import torch
import torch.nn as nn

# from functools import reduce, lru_cache
# import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import math
import numpy as np
from operator import mul
from ...builder import BACKBONES


def _to_channel_last(x):
    """
    Args:
        x: (B, C, D, H, W)

    Returns:
        x: (B, D, H, W, C)
    """
    return x.permute(0, 2, 3, 4, 1).contiguous()


def _to_channel_first(x):
    """
    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, C, D, H, W)
    """
    return x.permute(0, 4, 1, 2, 3).contiguous()


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


class CRM(nn.Module):
    """Constructs a channel refinement module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, dim, out_dim, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        B, C, D, H, W = y.shape

        # Two different branches of CRM module
        y = y.reshape(B, C, -1).transpose(-1, -2).contiguous()
        y = self.conv(y)
        y = y.transpose(-1, -2).contiguous().reshape(B, C, D, H, W)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        y = x + x * y.expand_as(x)

        return y


class PatchMerge(nn.Module):
    """
    Down-sampling block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(
        self,
        dim,
        dim_out,
        norm_layer=nn.LayerNorm,
        reduce_temporal_dim=False,
        flag=None,
    ):
        """
        Args:
            dim: input features dimension.
            norm_layer: normalization layer.
            dim_out: output features dimension.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.GELU(),
            CRM(dim, dim),
            nn.Conv3d(dim, dim, 1, 1, 0, bias=False),
        )

        # 2x downsampling
        if reduce_temporal_dim:
            self.reduction = nn.Conv3d(dim, dim_out, 3, 2, 1, bias=False)
        else:
            if flag == 1:
                # B, C, D, H, W -> B, C, D, H//2, W//2
                self.reduction = nn.Conv3d(dim, dim_out, 3, (1, 2, 2), 1, bias=False)
            elif flag == 2 or flag == 3:
                # B, C, D, H, W -> B, C, D-8, H//2, W//2
                self.reduction = nn.Conv3d(
                    dim, dim_out, (9, 3, 3), (1, 2, 2), (0, 1, 1), bias=False
                )

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim_out)

    def forward(self, x):
        B, D, H, W, C = x.shape
        x = x.contiguous()
        x = self.norm1(x)
        x = _to_channel_first(x)
        # x = x + self.conv(x)
        x = self.reduction(x)
        x = _to_channel_last(x)
        x = self.norm2(x)
        return x  # B D H W C


class PosCNN(nn.Module):
    """
    PosCNN  PEG  from https://arxiv.org/abs/2102.10882

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dim):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim),
        )

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).contiguous().reshape(B, C, D, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x  # B N C

    def no_weight_decay(self):
        return ["proj.%d.weight" % i for i in range(4)]


class PyramidPooling(nn.Module):
    def __init__(self, dim, proj_drop=0.0, pool_ratios=[1, 2, 3, 6]):
        super().__init__()

        self.pool_ratios = pool_ratios
        self.convs = nn.ModuleList(
            [
                nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
                for _ in range(len(pool_ratios))
            ]
        )

        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, 2 * dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, D, H, W):
        B, N, C = x.shape

        pools = []
        x = x.permute(0, 2, 1).reshape(B, C, D, H, W).contiguous()
        for pool_ratio, conv in zip(self.pool_ratios, self.convs):
            depth_pool_ratio = min(pool_ratio, D)
            pool = F.adaptive_avg_pool3d(
                x, (depth_pool_ratio, pool_ratio, pool_ratio)
            )  # todo
            pool = pool + conv(pool)  # todo
            pools.append(pool.reshape(B, C, -1))

        x = torch.cat(pools, dim=2)
        x = self.norm(x.permute(0, 2, 1).contiguous())

        x = self.proj(x)
        x = self.proj_drop(x)

        return x  # B N 2C


# SLGT2
class SLGT2LocalAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        topk_div=2,
        wss=None,
        dws=None,
        rel_pos=False,
    ):
        super().__init__()
        self.wss = wss  # [7, 14]
        self.dws = dws  # [2, 4]
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.qk_scale = qk_scale
        self.rel_pos = rel_pos

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # todo
        self.topk_div = topk_div
        self.n_groups = len(self.wss)  # 多少组
        self.gdim = dim // self.n_groups  # 每组维度
        self.gnum_heads = num_heads // self.n_groups  # 每组多少头
        self.ghead_dim = self.gdim // self.gnum_heads  # 每组每个头多少维度
        self.scale = qk_scale or self.ghead_dim**-0.5

        # define a parameter table of relative position bias
        if self.rel_pos:
            self.window_size = [[wss, wss, dsw] for wss, dsw in zip(self.wss, self.dws)]
            self.relative_position_bias_table = []
            self.relative_position_index = []
            for i, window_size in enumerate(self.window_size):
                relative_position_bias_params = nn.Parameter(
                    torch.zeros(
                        (2 * window_size[0] - 1)
                        * (2 * window_size[1] - 1)
                        * (2 * window_size[2] - 1),
                        self.gnum_heads,
                    )
                )
                trunc_normal_(relative_position_bias_params, std=0.02)
                setattr(
                    self,
                    "relative_position_bias_params_{}".format(i),
                    relative_position_bias_params,
                )
                self.relative_position_bias_table.append(
                    getattr(self, "relative_position_bias_params_{}".format(i))
                )

                # get pair-wise relative position index for each token inside the window
                coords_d = torch.arange(window_size[0])
                coords_h = torch.arange(window_size[1])
                coords_w = torch.arange(window_size[2])
                coords = torch.stack(
                    torch.meshgrid(coords_d, coords_h, coords_w)
                )  # 3, Wd, Wh, Ww
                coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
                # 3, Wd*Wh*Ww, Wd*Wh*Ww
                relative_coords = (
                    coords_flatten[:, :, None] - coords_flatten[:, None, :]
                )
                relative_coords = relative_coords.permute(
                    1, 2, 0
                ).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
                relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
                relative_coords[:, :, 1] += window_size[1] - 1
                relative_coords[:, :, 2] += window_size[2] - 1

                relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (
                    2 * window_size[2] - 1
                )
                relative_coords[:, :, 1] *= 2 * window_size[2] - 1
                relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
                # print(relative_position_index.shape)
                self.register_buffer(
                    "relative_position_index_{}".format(i), relative_position_index
                )
                self.relative_position_index.append(
                    getattr(self, "relative_position_index_{}".format(i))
                )

    def forward(self, x, D, H, W):
        # local
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3).contiguous()
        qkv = qkv.reshape(3, B, D, H, W, C)

        qkv_groups = qkv.chunk(self.n_groups, -1)
        x_groups = []
        for i, qkv_group in enumerate(qkv_groups):
            # qkv_group.shape=[3, B, D, H, W, gdim]
            window_s = self.wss[i]
            d_window_s = self.dws[i]
            loc_d, loc_h, loc_w = d_window_s, window_s, window_s
            glb_d, glb_h, glb_w = (
                math.ceil(D / loc_d),
                math.ceil(H / loc_h),
                math.ceil(W / loc_w),
            )
            glb_dhw = glb_d * glb_h * glb_w
            # padding
            pad_d0 = pad_l = pad_t = 0
            pad_d1 = (d_window_s - D % d_window_s) % d_window_s
            pad_r = (window_s - W % window_s) % window_s
            pad_b = (window_s - H % window_s) % window_s
            qkv_group = F.pad(
                qkv_group, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1)
            )
            _, B, Dp, Hp, Wp, gdim = qkv_group.shape

            # local relation
            # window partition, qkv_group.shape=[3, B, D, H, W, gdim]
            qkv_windows = (
                qkv_group.reshape(3, B, glb_d, loc_d, glb_h, loc_h, glb_w, loc_w, gdim)
                .permute(0, 1, 2, 4, 6, 3, 5, 7, 8)
                .contiguous()
            )
            # nW*B, d_window_s*window_s*window_s, gdim
            qkv_windows = qkv_windows.reshape(
                3, -1, d_window_s * window_s * window_s, self.gdim
            )
            _, B_, N, _ = qkv_windows.shape

            # [3, B_, gnum_heads, N, ghead_dim]
            qkv = (
                qkv_windows.reshape(3, B_, N, self.gnum_heads, self.ghead_dim)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )

            head_dim = qkv.shape[-1]
            # B*dhw, gnum_heads, dws*ws*ws, ghead_dim
            [q, k, v] = [x for x in qkv]
            self.scale = self.qk_scale or head_dim**-0.5
            # B*dhw, gnum_heads, dws*ws*ws, dws*ws*ws
            attn = (q @ k.transpose(-2, -1)) * self.scale

            # todo
            if self.rel_pos:
                relative_position_bias = self.relative_position_bias_table[i][
                    self.relative_position_index[i].reshape(-1)
                ].reshape(
                    np.prod(self.window_size[i]), np.prod(self.window_size[i]), -1
                )  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(
                    2, 0, 1
                ).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0)
            # todo

            # todo the core code block https://github.com/damo-cv/KVT
            if self.topk_div > 1:
                mask_ = torch.zeros(
                    B_, self.gnum_heads, N, N, device=x.device, requires_grad=False
                )
                index = torch.topk(attn, k=N // self.topk_div, dim=-1, largest=True)[1]
                # print(f"N={N}, self.topk_div={self.topk_div}, k={N//self.topk_div}")
                mask_.scatter_(-1, index, 1.0)
                attn = torch.where(
                    mask_ > 0, attn, torch.full_like(attn, float("-inf"))
                )
            # todo end of the core code block

            # B*dhw, gnum_heads, dws*ws*ws, dws*ws*ws
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # # todo sparse core code
            # attn = torch.where(attn > self.thr, attn,
            #                    torch.full_like(attn, 0.))  # note that not inf!
            # # todo end of the sparse core code block

            # B*dhw, gnum_heads, dws*ws*ws, ghead_dim
            x = (attn @ v).transpose(1, 2).reshape(B_, N, self.gdim).contiguous()

            # merge windows
            x = x.reshape(B, glb_d, glb_h, glb_w, loc_d, loc_h, loc_w, self.gdim)
            x = (
                x.permute(0, 1, 4, 2, 5, 3, 6, 7)
                .reshape(B, Dp, Hp, Wp, self.gdim)
                .contiguous()
            )  # B D H W gdim

            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :D, :H, :W, :].contiguous()
            x_groups.append(x)

        x = torch.cat(x_groups, -1)  # [B,D,H,W,C]
        # x = self.sknet(x)  # [B,C,D,H,W]
        # x = x.reshape(B, self.dim, D * H * W).permute(0, 2, 1)  # [B,N,C]
        # no sknet
        x = x.reshape(B, -1, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x  # B,N,C


class SLGT2GlobalAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        topk_div=2,
        wss=None,
        dws=None,
        rel_pos=False,
    ):
        super().__init__()
        self.wss = wss  # [7, 14]
        self.dws = dws  # [2, 4]
        self.rel_pos = rel_pos
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.qk_scale = qk_scale

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # todo
        self.topk_div = topk_div
        self.thr = 1e-2

        self.n_groups = len(self.wss)  # 多少组
        self.gdim = dim // self.n_groups  # 每组维度
        self.gnum_heads = num_heads // self.n_groups  # 每组多少头
        self.ghead_dim = self.gdim // self.gnum_heads  # 每组每个头多少维度
        self.scale = qk_scale or self.ghead_dim**-0.5
        # todo
        self.glb_kv = PyramidPooling(dim=dim)

    def forward(self, x, D, H, W):
        # local-global
        B, N, C = x.shape
        q = self.q(x).reshape(B, D, H, W, C)
        kv = self.glb_kv(x, D, H, W)  # B Nk 2C
        kv = kv.reshape(B, -1, 2, C).permute(2, 0, 1, 3).contiguous()  # 2 B Nk C

        q_groups = q.chunk(self.n_groups, -1)
        kv_groups = kv.chunk(self.n_groups, -1)
        x_groups = []
        for i, q_group in enumerate(q_groups):
            # q_group.shape=B, D, H, W, gdim
            kv_group = kv_groups[i]
            window_s = self.wss[i]
            d_window_s = self.dws[i]
            loc_d, loc_h, loc_w = d_window_s, window_s, window_s
            glb_d, glb_h, glb_w = (
                math.ceil(D / loc_d),
                math.ceil(H / loc_h),
                math.ceil(W / loc_w),
            )
            # padding
            pad_d0 = pad_l = pad_t = 0
            pad_d1 = (d_window_s - D % d_window_s) % d_window_s
            pad_r = (window_s - W % window_s) % window_s
            pad_b = (window_s - H % window_s) % window_s
            q_group = F.pad(q_group, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            B, Dp, Hp, Wp, gdim = q_group.shape

            # local relation
            # window partition, q_group.shape=B, D, H, W, gdim
            q_windows = (
                q_group.reshape(B, glb_d, loc_d, glb_h, loc_h, glb_w, loc_w, gdim)
                .permute(0, 1, 3, 5, 2, 4, 6, 7)
                .contiguous()
            )
            # B, nW, d_window_s*window_s*window_s, gdim
            q_windows = q_windows.reshape(
                B, -1, d_window_s * window_s * window_s, self.gdim
            )
            B, nW, Nq, _ = q_windows.shape

            # B, nW, gnum_heads, Nq, ghead_dim
            q = (
                q_windows.reshape(B, nW, Nq, self.gnum_heads, self.ghead_dim)
                .transpose(-2, -3)
                .contiguous()
            )
            kv_group = (
                kv_group.reshape(2, B, 1, -1, self.gnum_heads, self.ghead_dim)
                .transpose(-2, -3)
                .contiguous()
            )
            k, v = kv_group
            _, _, _, Nk, _ = k.shape
            # B, nW, gnum_heads, Nq, Nk
            attn = (q @ k.transpose(-2, -1)) * self.scale

            # todo the core code block https://github.com/damo-cv/KVT
            mask_ = torch.zeros(
                B, nW, self.gnum_heads, Nq, Nk, device=x.device, requires_grad=False
            )
            index = torch.topk(attn, k=Nk // self.topk_div, dim=-1, largest=True)[1]
            # print(f"N={Nk}, self.topk_div={self.topk_div}, k={Nk//self.topk_div}")
            mask_.scatter_(-1, index, 1.0)
            attn = torch.where(mask_ > 0, attn, torch.full_like(attn, float("-inf")))
            # todo end of the core code block

            # B, nW, gnum_heads, Nq, Nk
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # # todo sparse core code
            # attn = torch.where(attn > self.thr, attn,
            #                    torch.full_like(attn, 0.))  # note that not inf!
            # # todo end of the sparse core code block

            # B, nW, gnum_heads, Nq, ghead_dim
            x = (
                (attn @ v).transpose(2, 3).contiguous().reshape(B, nW, Nq, self.gdim)
            )  # B N gdim

            # merge windows
            x = x.reshape(B, glb_d, glb_h, glb_w, loc_d, loc_h, loc_w, self.gdim)
            x = (
                x.permute(0, 1, 4, 2, 5, 3, 6, 7)
                .reshape(B, Dp, Hp, Wp, self.gdim)
                .contiguous()
            )  # B D H W gdim

            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :D, :H, :W, :].contiguous()
            x_groups.append(x)

        x = torch.cat(x_groups, -1)  # [B,D,H,W,C]
        x = x.reshape(B, -1, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x  # B,N,C


class SLGT2Block(nn.Module):
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
        attention=SLGT2LocalAttention,
        layer_scale=None,
        wss=None,
        dws=None,
        topk_div=None,
        rel_pos=False,
    ):
        super().__init__()
        assert wss is not None and dws is not None

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            # ws=window_size,
            # ms_attn=True if dim <= 96 * 2 else False,
            wss=wss,
            dws=dws,
            topk_div=topk_div,
            rel_pos=rel_pos,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x), D, H, W))
        # MLP
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))

        return x  # B, N, C


class SLGT2Layer(nn.Module):
    """
    GCViT layer based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(
        self,
        dim,
        depth,
        input_resolution,  # not used here
        image_resolution,  # not used here
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        wss=None,
        dws=None,
        topk_div=None,
        rel_pos=False,
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            input_resolution: input image resolution.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """

        super().__init__()
        assert wss is not None and dws is not None
        self.blocks = nn.ModuleList(
            [
                SLGT2Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attention=(
                        SLGT2LocalAttention if (i % 2 == 0) else SLGT2GlobalAttention
                    ),
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                    wss=wss,
                    dws=dws,
                    topk_div=topk_div,
                    rel_pos=rel_pos,
                )
                for i in range(depth)
            ]
        )
        self.cpe = PosCNN(dim)  # conditional position embedding

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        for i, blk in enumerate(self.blocks):
            x = blk(x, D, H, W)
            if i == 0:  # todo
                x = self.cpe(x, D, H, W)  # PEG here
        return x  # B N C


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(
        self,
        img_size=[12, 224, 224],
        patch_size=[2, 4, 4],
        in_chans=3,
        embed_dim=96,
        norm_layer=nn.LayerNorm,
        patchmerge=False,
        flag=None,
    ):
        super().__init__()
        self.patchmerge = patchmerge

        if not patchmerge:  # patch embedded
            # downsample x2
            self.proj = nn.Conv3d(
                in_chans, embed_dim, kernel_size=3, stride=2, padding=1
            )
            # downsample x2
            self.conv_down = PatchMerge(
                dim=embed_dim,
                dim_out=embed_dim,
                norm_layer=norm_layer,
                reduce_temporal_dim=False,
                flag=flag,
            )
        else:  # patch merging
            self.proj = nn.Identity()
            # todo downsample x2
            self.conv_down = PatchMerge(
                dim=in_chans,
                dim_out=embed_dim,
                norm_layer=norm_layer,
                reduce_temporal_dim=True,
                flag=flag,
            )

    def forward(self, x):
        B, C, D, H, W = x.shape

        x = self.proj(x)  # B, C, D, H, W
        x = _to_channel_last(x)
        x = self.conv_down(x)  # B, D, H, W, C
        B, D, H, W, C = x.shape
        x = x.reshape(B, -1, C)

        return x, (D, H, W)  # B,N,C


@BACKBONES.register_module()
class SLGT2(nn.Module):

    def __init__(
        self,
        img_size=[12, 224, 224],
        patch_size=[2, 4, 4],
        in_chans=2,
        num_classes=8,
        embed_dims=[96, 192, 384, 768],
        wss_lst=[[7, 14, 21], [7, 14, 21], [7, 14, 14], [7, 7, 7]],
        dss_lst=[[8, 4, 2], [4, 4, 2], [2, 2, 2], [1, 1, 1]],
        topk_div=2,
        num_heads=[3, 6, 12, 24],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,  # todo 0.2
        norm_layer="nn.LayerNorm",
        depths=[2, 2, 6, 2],
        layer_cls="SLGT2Layer",
        layer_scale=None,
        ape=False,
        rel_pos=False,
        flag=1,
    ):
        super().__init__()
        norm_layer = eval(norm_layer)
        layer_cls = eval(layer_cls)
        self.num_classes = num_classes
        self.depths = depths
        self.ape = ape
        self.rel_pos = rel_pos
        self.wss_lst = wss_lst
        self.dss_lst = dss_lst
        self.num_heads = num_heads
        self.flag = flag
        if self.flag == 1:
            temporal_dim_list = [16, 8, 4, 2, 1]  # slovenia
            self.patch_nums = [8 * 56 * 56, 4 * 28 * 28, 2 * 14 * 14, 1 * 7 * 7]  # todo
        elif self.flag == 2:  # for brandenburg
            temporal_dim_list = [48, 16, 8, 4, 2]
            self.patch_nums = [
                16 * 56 * 56,
                8 * 28 * 28,
                4 * 14 * 14,
                2 * 7 * 7,
            ]  # todo
        elif self.flag == 3:  # for pastisr
            temporal_dim_list = [80, 40, 20, 10, 5]
            self.patch_nums = [
                40 * 56 * 56,
                20 * 28 * 28,
                10 * 14 * 14,
                5 * 7 * 7,
            ]  # todo

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # encoder
        self.layers = nn.ModuleList()
        self.patch_embeds = nn.ModuleList()
        self.abs_pos_embeds = nn.ParameterList()
        self.abs_pos_embeds_up = nn.ParameterList()  # for decoder
        self.pos_drops = nn.ModuleList()
        for i in range(len(depths)):  # 0 1 2 3
            input_resolution = int(img_size[-1] // 4 // 2**i)
            _layer = layer_cls(
                dim=int(embed_dims[i]),
                depth=depths[i],
                input_resolution=input_resolution,  # todo
                image_resolution=img_size[-1],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                layer_scale=layer_scale,
                wss=self.wss_lst[i],
                dws=self.dss_lst[i],
                topk_div=topk_div,
                rel_pos=rel_pos,
            )
            self.layers.append(_layer)

            # patch embed
            if i == 0:
                _patch_embed = PatchEmbed(
                    img_size,
                    patch_size,
                    in_chans,
                    embed_dims[i],
                    patchmerge=False,
                    flag=flag,
                )
            else:  # todo
                _img_size = [
                    temporal_dim_list[i],
                    img_size[1] // 4 // 2 ** (i - 1),
                    img_size[2] // 4 // 2 ** (i - 1),
                ]
                _patch_embed = PatchEmbed(
                    _img_size,
                    np.array(patch_size) // 2,
                    embed_dims[i - 1],
                    embed_dims[i],
                    patchmerge=True,
                    flag=flag,
                )

            self.patch_embeds.append(_patch_embed)
            # patch_num = self.patch_embeds[-1].num_patches
            patch_num = self.patch_nums[i]
            self.abs_pos_embeds.append(
                nn.Parameter(torch.zeros(1, patch_num, embed_dims[i]))
            )
            self.abs_pos_embeds_up.append(
                nn.Parameter(torch.zeros(1, patch_num, embed_dims[i]))
            )
            self.pos_drops.append(nn.Dropout(p=drop_rate))
        self.norm = norm_layer(embed_dims[-1])

        # decoder
        self.abs_pos_embeds_up = self.abs_pos_embeds_up[:-1][::-1]
        self.layers_up = nn.ModuleList()
        # self.patch_expands = nn.ModuleList()
        self.transpose_convs = nn.ModuleList()
        self.pos_drops_up = nn.ModuleList()
        self.cat_back_dims = nn.ModuleList()
        for i in range(len(depths) - 1):  # 0 1 2
            input_resolution = int(img_size[-1] // 16 * 2**i)  # 14 28 56
            _layer_up = layer_cls(
                dim=int(embed_dims[-i - 2]),
                depth=depths[-i - 2],
                input_resolution=input_resolution,  # todo
                image_resolution=img_size[-1],
                num_heads=num_heads[-i - 2],
                mlp_ratio=mlp_ratios[-i - 2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[: -i - 2]) : sum(depths[: -i - 2 + 1])],
                norm_layer=norm_layer,
                layer_scale=layer_scale,
                wss=self.wss_lst[-i - 2],  # todo
                dws=self.dss_lst[-i - 2],  # todo
                topk_div=topk_div,
                rel_pos=rel_pos,
            )
            self.layers_up.append(_layer_up)
            self.transpose_convs.append(
                nn.Sequential(
                    nn.ConvTranspose3d(embed_dims[-1 - i], embed_dims[-2 - i], 2, 2, 0),
                    nn.GELU(),
                    nn.Conv3d(
                        embed_dims[-2 - i],
                        embed_dims[-2 - i],
                        1,
                        groups=embed_dims[-2 - i],
                    ),
                    nn.GELU(),
                )
            )
            _cat_back_dim = nn.Sequential(
                nn.Conv3d(2 * embed_dims[-2 - i], embed_dims[-2 - i], 3, 1, 1)
            )
            self.cat_back_dims.append(_cat_back_dim)
            self.pos_drops_up.append(nn.Dropout(p=drop_rate))

        # final upsample
        self.up_x4 = nn.Sequential(
            nn.ConvTranspose3d(
                embed_dims[0], embed_dims[0], (1, 2, 2), (1, 2, 2), (0, 0, 0)
            ),
            nn.GELU(),
            nn.Conv3d(embed_dims[0], embed_dims[0], 1, groups=embed_dims[0]),
            nn.GELU(),
            nn.ConvTranspose3d(
                embed_dims[0], embed_dims[0], (1, 2, 2), (1, 2, 2), (0, 0, 0)
            ),
            nn.GELU(),
            nn.Conv3d(embed_dims[0], embed_dims[0], 1, groups=embed_dims[0]),
            nn.GELU(),
        )
        # segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv3d(embed_dims[0], num_classes, 1),
        )

        # init weights
        for pos_emb in self.abs_pos_embeds:
            trunc_normal_(pos_emb, std=0.02)
        for pos_emb in self.abs_pos_embeds_up:
            trunc_normal_(pos_emb, std=0.02)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            import math

            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv3d):
                fan_out = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            # logger.info(f"load model from: {pretrained}")

            # todo
            weights = None
            self.load_from(weights)

        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward_features(self, x):
        B = x.shape[0]
        down_feats = []  # all b,c,d,h,w
        # patch embed
        x, (D, H, W) = self.patch_embeds[0](x)  # B N C
        if self.ape:
            x = x + self.abs_pos_embeds[0]  # abs pos
            x = self.pos_drops[0](x)
        # merge path
        for i, layer in enumerate(self.layers):
            x = layer(x, D, H, W)  # B N C
            if i < len(self.depths) - 1:  # 0 1 2
                x = (
                    x.transpose(-1, -2).contiguous().reshape(B, -1, D, H, W)
                )  # B C D H W
                down_feats.append(x)
                # patch merge
                x, (D, H, W) = self.patch_embeds[i + 1](x)  # B N C
                if self.ape:
                    x = x + self.abs_pos_embeds[i + 1]
                    x = self.pos_drops[i + 1](x)

        x = self.norm(x)
        x = x.transpose(-1, -2).contiguous().reshape(B, -1, D, H, W)  # B C D H W
        down_feats.append(x)

        return down_feats

    def forward_features_up(self, down_feats):
        x = down_feats[-1]  # B C D H W
        # expand path
        for i, layer_up in enumerate(self.layers_up):
            # upsample x2
            x = self.transpose_convs[i](x)  # B C D H W
            x = torch.cat((x, down_feats[-i - 2]), dim=1)  # B C D H W
            x = self.cat_back_dims[i](x)  # B C D H W
            B, C, D, H, W = x.shape
            x = x.reshape(B, C, -1).contiguous().transpose(-1, -2)  # B N C
            if self.ape:
                x = x + self.abs_pos_embeds_up[i]
                x = self.pos_drops_up[i](x)

            x = layer_up(x, D, H, W)  # B N C
            x = x.transpose(-1, -2).contiguous().reshape(B, -1, D, H, W)  # B C D H W

        return x

    def forward(self, x):
        B, C, D, H, W = x.shape
        # padding for x
        if self.flag == 1:  # no padding for slovenia
            desired_temporal_dim = 16
        elif self.flag == 2:  # brandenburg 41->48 for dim D
            desired_temporal_dim = 48
        elif self.flag == 3:  # pastisr 65/70/71->80 for dim D
            desired_temporal_dim = 80
        padding = (
            0,
            0,
            0,
            0,
            (desired_temporal_dim - D) // 2,
            desired_temporal_dim - D - (desired_temporal_dim - D) // 2,
        )
        x = F.pad(x, padding, "constant", 0)
        down_feats = self.forward_features(x)
        x = self.forward_features_up(down_feats)
        x = self.up_x4(x)
        x = self.seg_head(x)
        out = torch.mean(x, dim=2, keepdim=False)

        return out


if __name__ == "__main__":
    x = torch.randn((1, 2, 16, 224, 224)).cuda()
    net = SLGT2().cuda()
    out = net(x)
    print(out.shape)
    pass
