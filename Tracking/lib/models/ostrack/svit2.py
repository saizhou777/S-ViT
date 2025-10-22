
import math
from functools import partial
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from timm.models.registry import register_model

from collections import OrderedDict


def combine_tokens(z_tokens: torch.Tensor, x_tokens: torch.Tensor, mode: str = 'direct'):
    if mode == 'direct':
        return torch.cat([z_tokens, x_tokens], dim=1)
    else:
        raise ValueError(f"Unsupported cat_mode: {mode}")


def recover_tokens(merged: torch.Tensor, lens_z: int, lens_x: int, mode: str = 'direct'):
    if mode == 'direct':
        z = merged[:, :lens_z]
        x = merged[:, lens_z:lens_z + lens_x]
        return z, x
    else:
        raise ValueError(f"Unsupported cat_mode: {mode}")


class SVITBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'
        self.pos_embed_z: Optional[nn.Parameter] = None
        self.pos_embed_x: Optional[nn.Parameter] = None
        self.template_segment_pos_embed: Optional[nn.Parameter] = None
        self.search_segment_pos_embed: Optional[nn.Parameter] = None

        self.return_inter = False
        self.return_stage: List[int] = []
        self.add_cls_token = False
        self.add_sep_seg = False

    def finetune_track(self, cfg, patch_start_index=0):
        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch = cfg.MODEL.BACKBONE.STRIDE
        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        if new_patch != self.patch_size:
            self.patch_size = new_patch

        stride_total = 16  # PatchEmbed(4x) + stage1(2x) + stage2(2x)
        th, tw = template_size
        sh, sw = search_size
        tph, tpw = th // stride_total, tw // stride_total
        sph, spw = sh // stride_total, sw // stride_total
        tz = tph * tpw
        sx = sph * spw

        if self.embed_dim is None:
            raise ValueError("self.embed_dim  Not correctly configured in subclass construction")
        self.pos_embed_z = nn.Parameter(torch.zeros(1, tz, self.embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, sx, self.embed_dim))
        trunc_normal_(self.pos_embed_z, std=.02)
        trunc_normal_(self.pos_embed_x, std=.02)

        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(self.template_segment_pos_embed, std=.02)
            trunc_normal_(self.search_segment_pos_embed, std=.02)

        if self.return_inter:
            for idx in self.return_stage:
                ln = nn.LayerNorm(self.embed_dim)
                self.add_module(f"norm{idx}", ln)


# ============== LayerNorm2d ==============
class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DepthwiseConvTokenMixer(nn.Module):
    def __init__(self, dim, num_heads, use_rope=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward_single(self, x):
        return self.dwconv(self.bn(x))

    def forward_pair(self, z, x):
        return self.forward_single(z), self.forward_single(x)



class SalienceAttention(nn.Module):
    def __init__(self, dim, num_heads, fixed_n_tokens=98, pool_size=(1, 1)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.scale = self.head_dim ** -0.5
        self.pool_size = pool_size

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.proj_out = nn.Linear(dim, dim)

        self.proj = nn.Conv2d(dim, dim, 1)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.activation_fn = F.gelu
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, dim, 1)

    @staticmethod
    def _select(feat: torch.Tensor, k_target: int, pool_size=(1, 1)):
        B, C, H, W = feat.shape
        if pool_size != (1, 1):
            pooled = F.avg_pool2d(feat, pool_size, pool_size)
        else:
            pooled = feat
        B, C, Hp, Wp = pooled.shape
        Np = Hp * Wp
        flat = pooled.view(B, C, Np)
        scores = flat.square().sum(1)  # (B,Np)
        max_k = max(1, Np - 1)
        k = min(k_target, max_k)
        _, top_idx = scores.topk(k, dim=1)
        top_tokens = flat.gather(2, top_idx.unsqueeze(1).expand(-1, C, -1))
        all_sum = flat.sum(2)
        sel_sum = top_tokens.sum(2)
        denom = (Np - k)
        if denom <= 0:
            bg = torch.zeros_like(all_sum)
        else:
            bg = (all_sum - sel_sum) / denom
        return {
            "feat": feat,
            "pooled": pooled,
            "flat": flat,
            "Hp": Hp, "Wp": Wp, "Np": Np,
            "H": H, "W": W,
            "k": k,
            "top_idx": top_idx,        # (B,k)
            "top_tokens": top_tokens,  # (B,C,k)
            "bg": bg                   # (B,C)
        }

    @staticmethod
    def _restore(info: Dict, sel_tokens: torch.Tensor, bg_token: torch.Tensor):
        B, k, C = sel_tokens.shape
        flat = bg_token.transpose(1, 2).expand(-1, C, info["Np"]).clone()
        flat.scatter_(2, info["top_idx"].unsqueeze(1).expand(-1, C, -1),
                      sel_tokens.transpose(1, 2))
        rec = flat.view(B, C, info["Hp"], info["Wp"])
        if (info["Hp"], info["Wp"]) != (info["H"], info["W"]):
            rec = F.interpolate(rec, size=(info["H"], info["W"]), mode='nearest')
        return rec

    def _region_self_attn(self, tokens: torch.Tensor):
        B, N, C = tokens.shape
        qkv = self.qkv_proj(tokens)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_out(out)
        return out

    def forward_pair(self, z: torch.Tensor, x: torch.Tensor,
                     k_template=32, k_search=98):

        z_info = self._select(z, k_template, self.pool_size)
        x_info = self._select(x, k_search, self.pool_size)


        z_sel = z_info["top_tokens"].transpose(1, 2)  # (B,kz,C)
        z_bg = z_info["bg"].unsqueeze(1)              # (B,1,C)
        z_tokens = torch.cat([z_sel, z_bg], dim=1)

        x_sel = x_info["top_tokens"].transpose(1, 2)  # (B,kx,C)
        x_bg = x_info["bg"].unsqueeze(1)              # (B,1,C)
        x_tokens = torch.cat([x_sel, x_bg], dim=1)

        z_tokens_out = self._region_self_attn(z_tokens)
        x_tokens_out = self._region_self_attn(x_tokens)

        kz = z_info["k"]
        kx = x_info["k"]
        z_restored = self._restore(z_info, z_tokens_out[:, :kz], z_tokens_out[:, kz:kz + 1])
        x_restored = self._restore(x_info, x_tokens_out[:, :kx], x_tokens_out[:, kx:kx + 1])

        def conv_branch(feat):
            y = self.bn(feat)
            y = self.fc1(y)
            y = self.activation_fn(y)
            y = self.conv(y)
            y = self.fc2(y)
            return y

        z_conv = conv_branch(z)
        x_conv = conv_branch(x)

        z_out = self.proj(z_restored + z_conv)
        x_out = self.proj(x_restored + x_conv)
        return z_out, x_out


class VanillaSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.scale = self.head_dim ** -0.5
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward_pair(self, z: torch.Tensor, x: torch.Tensor):
        B, C, Hz, Wz = z.shape
        _, _, Hx, Wx = x.shape

        def extract(feat):
            qkvo = self.qkvo(feat)          # (B,4C,H,W)
            qkv = qkvo[:, :3 * C]           # (B,3C,H,W)
            o = qkvo[:, 3 * C:]             # (B,C,H,W)
            lepe = self.lepe(qkv[:, 2 * C:])
            q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d',
                                m=3, n=self.num_heads, d=C // self.num_heads)
            return q, k, v, o, lepe

        qz, kz, vz, oz, lepez = extract(z)
        qx, kx, vx, ox, lepex = extract(x)

        q_all = torch.cat([qz, qx], dim=2)
        k_all = torch.cat([kz, kx], dim=2)
        v_all = torch.cat([vz, vx], dim=2)

        attn = (self.scale * q_all @ k_all.transpose(-1, -2)).softmax(dim=-1)
        out_all = attn @ v_all

        Nz = Hz * Wz
        out_z = out_all[:, :, :Nz]
        out_x = out_all[:, :, Nz:]

        out_z = rearrange(out_z, 'b h (hw) d -> b (h d) hw', hw=Nz).view(B, C, Hz, Wz)
        out_x = rearrange(out_x, 'b h (hw) d -> b (h d) hw', hw=Hx * Wx).view(B, C, Hx, Wx)

        out_z = self.proj((out_z + lepez) * oz)
        out_x = self.proj((out_x + lepex) * ox)
        return out_z, out_x


# ============== FeedForwardNetwork ==============
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, activation_fn=F.gelu,
                 dropout=0.0, activation_dropout=0.0, subconv=True):
        super().__init__()
        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, 1)
        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, 1)
        self.act = activation_fn
        self.drop1 = nn.Dropout(activation_dropout)
        self.drop2 = nn.Dropout(dropout)
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, 3, 1, 1, groups=ffn_dim) if subconv else None

    def forward_single(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        if self.dwconv is not None:
            r = x
            x = self.dwconv(x)
            x = x + r
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def forward_pair(self, z, x):
        return self.forward_single(z), self.forward_single(x)


# ============== PatchMerging ==============
class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward_single(self, x):
        x = self.reduction(x)
        x = self.norm(x)
        return x

    def forward_pair(self, z, x):
        return self.forward_single(z), self.forward_single(x)


# ============== Block ==============
class Block(nn.Module):
    def __init__(self, flag, embed_dim, num_heads, ffn_dim, drop_path=0.,
                 layerscale=False, layer_init_value=1e-6, pool_size=(1, 1)):
        super().__init__()
        self.flag = flag
        self.layerscale = layerscale
        assert flag in ['l', 's', 'v']

        if flag != 'l':
            self.norm1_z = LayerNorm2d(embed_dim)
            self.norm1_x = LayerNorm2d(embed_dim)

        if flag == 'l':
            self.attn = DepthwiseConvTokenMixer(embed_dim, num_heads)
        elif flag == 's':
            self.attn = SalienceAttention(embed_dim, num_heads, pool_size=pool_size)
        else:
            self.attn = VanillaSelfAttention(embed_dim, num_heads)

        self.norm2_z = LayerNorm2d(embed_dim)
        self.norm2_x = LayerNorm2d(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_value * torch.ones(1, embed_dim, 1, 1))
            self.gamma_2 = nn.Parameter(layer_init_value * torch.ones(1, embed_dim, 1, 1))

    def forward_pair(self, z, x, k_template=32, k_search=98):
        z = z + self.pos(z)
        x = x + self.pos(x)

        if self.layerscale:
            if self.flag == 'l':
                az, ax = self.attn.forward_pair(z, x)
                z = z + self.drop_path(self.gamma_1 * az)
                x = x + self.drop_path(self.gamma_1 * ax)
            elif self.flag == 's':
                az_in, ax_in = self.norm1_z(z), self.norm1_x(x)
                az, ax = self.attn.forward_pair(az_in, ax_in, k_template=k_template, k_search=k_search)
                z = z + self.drop_path(self.gamma_1 * az)
                x = x + self.drop_path(self.gamma_1 * ax)
            else:
                az_in, ax_in = self.norm1_z(z), self.norm1_x(x)
                az, ax = self.attn.forward_pair(az_in, ax_in)
                z = z + self.drop_path(self.gamma_1 * az)
                x = x + self.drop_path(self.gamma_1 * ax)

            fz_in, fx_in = self.norm2_z(z), self.norm2_x(x)
            fz, fx = self.ffn.forward_pair(fz_in, fx_in)
            z = z + self.drop_path(self.gamma_2 * fz)
            x = x + self.drop_path(self.gamma_2 * fx)
        else:
            if self.flag == 'l':
                az, ax = self.attn.forward_pair(z, x)
                z = z + self.drop_path(az)
                x = x + self.drop_path(ax)
            elif self.flag == 's':
                az_in, ax_in = self.norm1_z(z), self.norm1_x(x)
                az, ax = self.attn.forward_pair(az_in, ax_in, k_template=k_template, k_search=k_search)
                z = z + self.drop_path(az)
                x = x + self.drop_path(ax)
            else:
                az_in, ax_in = self.norm1_z(z), self.norm1_x(x)
                az, ax = self.attn.forward_pair(az_in, ax_in)
                z = z + self.drop_path(az)
                x = x + self.drop_path(ax)

            fz_in, fx_in = self.norm2_z(z), self.norm2_x(x)
            fz, fx = self.ffn.forward_pair(fz_in, fx_in)
            z = z + self.drop_path(fz)
            x = x + self.drop_path(fx)
        return z, x


# ============== BasicLayer ==============
class BasicLayer(nn.Module):
    def __init__(self, flags, embed_dim, out_dim, depth, num_heads,
                 ffn_dim=96., drop_path=0.,
                 downsample: PatchMerging = None,
                 layerscale=False, layer_init_value=1e-6,
                 pool_size=(1, 1)):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(flags[i], embed_dim, num_heads, ffn_dim,
                  drop_path[i] if isinstance(drop_path, list) else drop_path,
                  layerscale, layer_init_value,
                  pool_size=pool_size if flags[i] == 's' else (1, 1))
            for i in range(depth)
        ])
        self.downsample = PatchMerging(embed_dim, out_dim) if downsample is not None else None

    def forward_pair(self, z, x, k_template=32, k_search=98):
        for blk in self.blocks:
            z, x = blk.forward_pair(z, x, k_template=k_template, k_search=k_search)
        if self.downsample is not None:
            z, x = self.downsample.forward_pair(z, x)
        return z, x


# ============== PatchEmbed ==============
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x):
        return self.proj(x)

    def forward_pair(self, z, x):
        return self.forward(z), self.forward(x)


class FinalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.ln_z = nn.LayerNorm(dim)
        self.ln_x = nn.LayerNorm(dim)

        # q/k/v 投影
        self.q_z = nn.Linear(dim, dim)
        self.k_z = nn.Linear(dim, dim)
        self.v_z = nn.Linear(dim, dim)

        self.q_x = nn.Linear(dim, dim)
        self.k_x = nn.Linear(dim, dim)
        self.v_x = nn.Linear(dim, dim)

        self.proj_z = nn.Linear(dim, dim)
        self.proj_x = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def _mha(self, q, k, v):
        # q,k,v: (B,N,C)
        B, N, C = q.shape
        h = self.num_heads
        d = self.head_dim
        q = q.view(B, N, h, d).transpose(1, 2)  # (B,h,N,d)
        k = k.view(B, -1, h, d).transpose(1, 2)
        v = v.view(B, -1, h, d).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return out

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        B, C, Hz, Wz = z.shape
        B2, C2, Hx, Wx = x.shape
        assert B == B2 and C == C2

        Nz = Hz * Wz
        Nx = Hx * Wx

        z_tok = z.flatten(2).transpose(1, 2)  # (B,Nz,C)
        x_tok = x.flatten(2).transpose(1, 2)  # (B,Nx,C)

        z_ln = self.ln_z(z_tok)
        x_ln = self.ln_x(x_tok)

        # z <- x
        q_z = self.q_z(z_ln)
        k_x = self.k_x(x_ln)
        v_x = self.v_x(x_ln)
        z_ctx = self._mha(q_z, k_x, v_x)
        z_ctx = self.proj_z(z_ctx)
        z_out = z_tok + self.drop(z_ctx)

        # x <- z
        q_x = self.q_x(x_ln)
        k_z = self.k_z(z_ln)
        v_z = self.v_z(z_ln)
        x_ctx = self._mha(q_x, k_z, v_z)
        x_ctx = self.proj_x(x_ctx)
        x_out = x_tok + self.drop(x_ctx)

        z_out = z_out.transpose(1, 2).view(B, C, Hz, Wz)
        x_out = x_out.transpose(1, 2).view(B, C, Hx, Wx)
        return z_out, x_out


class SVIT(SVITBackbone):
    def __init__(self,
                 in_chans=3,
                 flagss=None,
                 embed_dims=[96, 192, 384],
                 depths=[2, 2, 6],
                 num_heads=[3, 6, 12],
                 mlp_ratios=[3, 3, 3],
                 drop_path_rate=0.1,
                 layerscales=[True, True, True],
                 layer_init_values=[1e-6, 1e-6, 1e-6],
                 k_template=63,
                 k_search=144,
                 final_cross=True,          
                 final_cross_dropout=0.0):
        super().__init__()
        assert len(embed_dims) == len(depths) == len(num_heads) == len(mlp_ratios) == 3, "only first three stages"
        self.num_layers = 3
        self.k_template = k_template
        self.k_search = k_search
        self.embed_dim = embed_dims[-1]
        self.final_cross_enabled = final_cross

        if flagss is None:
            flagss = [
                ['l'] * depths[0],
                ['s'] * depths[1],
                ['s'] * depths[2],
            ]

        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0])

        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(self.num_layers):
            pool_size = (8, 8) if i == 0 else (2, 2) if i == 1 else (1, 1)
            layer = BasicLayer(
                flags=flagss[i],
                embed_dim=embed_dims[i],
                out_dim=embed_dims[i + 1] if i < self.num_layers - 1 else None,
                depth=depths[i],
                num_heads=num_heads[i],
                ffn_dim=int(mlp_ratios[i] * embed_dims[i]),
                drop_path=dpr[cur:cur + depths[i]],
                downsample=PatchMerging if i < self.num_layers - 1 else None,
                layerscale=layerscales[i],
                layer_init_value=layer_init_values[i],
                pool_size=pool_size
            )
            cur += depths[i]
            self.stages.append(layer)

        if self.final_cross_enabled:
            self.final_cross_attn = FinalCrossAttention(embed_dims[-1], num_heads[-1],
                                                        dropout=final_cross_dropout)

        self.final_norm = nn.LayerNorm(embed_dims[-1])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features_pair(self, z: torch.Tensor, x: torch.Tensor):
        z_feat, x_feat = self.patch_embed.forward_pair(z, x)
        stage_feats = []
        for i, stage in enumerate(self.stages):
            z_feat, x_feat = stage.forward_pair(z_feat, x_feat,
                                                k_template=self.k_template,
                                                k_search=self.k_search)
            stage_feats.append((z_feat, x_feat))
        return z_feat, x_feat, stage_feats

    def forward(self, z: torch.Tensor, x: torch.Tensor, **kwargs):
        z_last, x_last, stage_feats = self.forward_features_pair(z, x)

        if self.final_cross_enabled:
            z_last, x_last = self.final_cross_attn(z_last, x_last)

        B, C, Hz, Wz = z_last.shape
        _, _, Hx, Wx = x_last.shape
        assert C == self.embed_dim

        z_tokens = z_last.flatten(2).transpose(1, 2)  # (B,Lz,C)
        x_tokens = x_last.flatten(2).transpose(1, 2)  # (B,Lx,C)
        lens_z = z_tokens.shape[1]
        lens_x = x_tokens.shape[1]

        if self.pos_embed_z is not None and self.pos_embed_z.shape[1] == lens_z:
            z_tokens = z_tokens + self.pos_embed_z
        if self.pos_embed_x is not None and self.pos_embed_x.shape[1] == lens_x:
            x_tokens = x_tokens + self.pos_embed_x
        if self.add_sep_seg and self.template_segment_pos_embed is not None:
            z_tokens = z_tokens + self.template_segment_pos_embed
            x_tokens = x_tokens + self.search_segment_pos_embed

        merged = combine_tokens(z_tokens, x_tokens, mode=self.cat_mode)
        merged = self.final_norm(merged)

        aux = {
            "lens_z": lens_z,
            "lens_x": lens_x,
            "stage_feats": stage_feats
        }
        return merged, aux


def build_svit_backbone(variant='T', pretrained=False, **kwargs):
    v = variant.upper()
    if v == 'T':
        cfg = dict(embed_dims=[48, 96, 144],
                   depths=[2, 2, 8],
                   num_heads=[1, 2, 3],
                   mlp_ratios=[3.5, 3.5, 3.5],
                   drop_path_rate=0.1)
    elif v == 'S':
        cfg = dict(embed_dims=[64, 128, 280],
                   depths=[3, 4, 16],
                   num_heads=[1, 2, 5],
                   mlp_ratios=[3.5, 3.5, 3.5],
                   drop_path_rate=0.15)
    else:
        raise ValueError(f"Unknown variant {variant}")
    model = SVIT(
        embed_dims=cfg["embed_dims"],
        depths=cfg["depths"],
        num_heads=cfg["num_heads"],
        mlp_ratios=cfg["mlp_ratios"],
        drop_path_rate=cfg["drop_path_rate"],
        **kwargs
    )

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


@register_model
def SVIT_T(pretrained=False, args=None, **kwargs):
    return build_svit_backbone('T', pretrained=pretrained, **kwargs)


@register_model
def SVIT_S(pretrained=False, args=None, **kwargs):
    return build_svit_backbone('S', pretrained=pretrained, **kwargs)


def convert_cls_to_track_state_dict(cls_ckpt: dict,
                                    track_model,
                                    backbone_prefix: str = '',
                                    ignore_last_stage: bool = True,
                                    duplicate_norm: bool = True,
                                    verbose: bool = True):
    if 'model' in cls_ckpt and isinstance(cls_ckpt['model'], dict):
        cls_sd = cls_ckpt['model']
    else:
        cls_sd = cls_ckpt

    if any(k.startswith('module.') for k in cls_sd.keys()):
        cls_sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in cls_sd.items()}

    new_sd = OrderedDict()
    skipped = []
    dup_norm1 = dup_norm2 = 0
    track_block_counts = [len(s.blocks) for s in track_model.stages]

    def blk_valid(si, bi):
        if si >= len(track_block_counts):
            return False
        return bi < track_block_counts[si]

    for k, v in cls_sd.items():
        if k.startswith('proj.') or k.startswith('head.') or k.startswith('avgpool') or k.startswith('swish') or k.startswith('norm.'):
            skipped.append(k); continue

        if k.startswith('layers.'):
            parts = k.split('.')
            try:
                si = int(parts[1])
            except:
                skipped.append(k); continue
            if ignore_last_stage and si >= 3:
                skipped.append(k); continue

            bi = None
            if '.blocks.' in k:
                try:
                    bi = int(k.split('.blocks.')[1].split('.')[0])
                except:
                    skipped.append(k); continue
                if not blk_valid(si, bi):
                    skipped.append(k); continue

            k2 = k.replace('layers.', 'stages.', 1)

            if duplicate_norm and '.norm1.' in k2:
                base = k2.replace('.norm1.', '.norm1_z.')
                k_z = base
                k_x = k2.replace('.norm1.', '.norm1_x.')
                if backbone_prefix:
                    k_z = f'{backbone_prefix}.{k_z}'
                    k_x = f'{backbone_prefix}.{k_x}'
                new_sd[k_z] = v
                new_sd[k_x] = v.clone()
                dup_norm1 += 2
                continue
            if duplicate_norm and '.norm2.' in k2:
                base = k2.replace('.norm2.', '.norm2_z.')
                k_z = base
                k_x = k2.replace('.norm2.', '.norm2_x.')
                if backbone_prefix:
                    k_z = f'{backbone_prefix}.{k_z}'
                    k_x = f'{backbone_prefix}.{k_x}'
                new_sd[k_z] = v
                new_sd[k_x] = v.clone()
                dup_norm2 += 2
                continue

            if backbone_prefix:
                k2 = f'{backbone_prefix}.{k2}'
            new_sd[k2] = v
        elif k.startswith('patch_embed.'):
            k2 = f'{backbone_prefix}.{k}' if backbone_prefix else k
            new_sd[k2] = v
        else:
            skipped.append(k)

    if verbose:
        print('=== Convert CLS->TRACK Summary ===')
        print(f'Original cls params: {len(cls_sd)}')
        print(f'Mapped params (after duplication): {len(new_sd)}')
        print(f'Duplicated norm1: {dup_norm1}, norm2: {dup_norm2}')
        print(f'Skipped: {len(skipped)}  (show 8) {skipped[:8]}')
    return new_sd


def load_cls_as_track(backbone,
                      cls_ckpt_path: str,
                      backbone_prefix: str = '',
                      strict: bool = False,
                      **convert_kwargs):
    ckpt = torch.load(cls_ckpt_path, map_location='cpu')
    converted = convert_cls_to_track_state_dict(ckpt, backbone,
                                                backbone_prefix=backbone_prefix,
                                                verbose=True,
                                                **convert_kwargs)
    model_sd = backbone.state_dict()
    loadable = {}
    bad = []
    for k, v in converted.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            loadable[k] = v
        else:
            bad.append(k)
    msg = backbone.load_state_dict(loadable, strict=False)
    print('=== Load Report ===')
    print(f'Loadable (shape matched): {len(loadable)}')
    print(f'Bad (name or shape mismatch): {len(bad)} (show 8) {bad[:8]}')
    print(f'Missing after load: {len(msg.missing_keys)} (show 8) {msg.missing_keys[:8]}')
    print(f'Unexpected: {len(msg.unexpected_keys)}')
    if strict and (msg.missing_keys or msg.unexpected_keys):
        raise RuntimeError('Strict load failed.')
    return msg