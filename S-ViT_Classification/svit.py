import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Tuple


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = x.permute(0, 2, 3, 1)  # .contiguous() #(b h w c)
        x = self.norm(x)  # (b h w c)
        x = x.permute(0, 3, 1, 2)  # .contiguous()
        return x


class DepthwiseConvTokenMixer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.bn = nn.BatchNorm2d(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups= dim)

    def forward(self, x: torch.Tensor):
        # x: (b c h w)

        x = self.bn(x)
        x = self.dwconv(x)

        return x


class SalienceAttention(nn.Module): 
    def __init__(self, dim, num_heads, fixed_n_tokens=98, pool_size=(1, 1)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.scale = self.head_dim ** (-0.5)
        self.qkv_proj = nn.Linear(dim, dim * 3) 
        self.proj = nn.Conv2d(dim, dim, 1) 

        self.fixed_n_tokens = fixed_n_tokens  
        self.pool_size = pool_size  

        self.bn = nn.BatchNorm2d(dim)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)  
        self.activation_fn = F.gelu
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        N = H * W  

        if self.pool_size != (1, 1):
            x_pooled = F.avg_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)  
        else:
            x_pooled = x 

        B, C, H_pooled, W_pooled = x_pooled.shape
        N_pooled = H_pooled * W_pooled 
        x_flat = x_pooled.view(B, C, N_pooled)

        token_scores = x_flat.square().sum(1)  # (B, N_pooled)
        top_scores, top_idx = token_scores.topk(self.fixed_n_tokens, dim=1)

        top_tokens = x_flat.gather(2, top_idx.unsqueeze(1).expand(-1, C, -1))   # (B, C, k)

        all_sum = x_flat.sum(2)  # (B, C)
        top_sum = top_tokens.sum(2)  # (B, C)
        bg_token = (all_sum - top_sum) / (N_pooled - self.fixed_n_tokens)  # (B, C)

        tokens = torch.cat([top_tokens, bg_token.unsqueeze(-1)], dim=2)  # (B, C, k+1)
        tokens = tokens.transpose(1, 2)  # (B, k+1, C)

        qkv = self.qkv_proj(tokens)  
        qkv = rearrange(qkv, 'b n (three heads d) -> three b heads n d', three=3, heads=self.num_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        attn = torch.softmax(self.scale * q @ k.transpose(-1, -2), dim=-1) 
        res = attn @ v  
        attn_out = rearrange(res, 'b heads n d -> b (heads d) n')  

        top_res, bg_res = attn_out[:, :, :self.fixed_n_tokens], attn_out[:, :, self.fixed_n_tokens:]
        restored = bg_res.expand(-1, -1, N_pooled).clone()  # (B, C, Np)
        restored.scatter_(2, top_idx.unsqueeze(1).expand(-1, C, -1), top_res)

        restored_res = restored.view(B, C, H_pooled, W_pooled)
        restored_res = F.interpolate(
            restored_res,
            size=(H, W)
        )

        x = self.bn(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.conv(x)
        conv_out = self.fc2(x)

        combined = restored_res + conv_out  

        return self.proj(combined) 


class VanillaSelfAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor):
        # x: (b c h w)
        # sin: ((h w) d1)
        # cos: ((h w) d1)

        B, C, H, W = x.shape
        qkvo = self.qkvo(x)  # (b 3*c h w)
        qkv = qkvo[:, :3 * self.dim, :, :]
        o = qkvo[:, 3 * self.dim:, :, :]
        lepe = self.lepe(qkv[:, 2 * self.dim:, :, :])  # (b c h w)

        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads)  # (b n (h w) d)

        attn = torch.softmax(self.scale * q @ k.transpose(-1, -2), dim=-1)  # (b n (h w) (h w))
        res = attn @ v  # (b n (h w) d)
        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        res = res + lepe
        return self.proj(res * o)


class FeedForwardNetwork(nn.Module):
    def __init__(
            self,
            embed_dim,
            ffn_dim,
            activation_fn=F.gelu,
            dropout=0.0,
            activation_dropout=0.0,
            subconv=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Conv2d(self.embed_dim, ffn_dim, 1)
        self.fc2 = nn.Conv2d(ffn_dim, self.embed_dim, 1)
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, 3, 1, 1, groups=ffn_dim) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.dwconv.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.dwconv is not None:
            residual = x
            x = self.dwconv(x)
            x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


class Block(nn.Module):

    def __init__(self, flag, embed_dim, num_heads, ffn_dim, drop_path=0., layerscale=False, layer_init_value=1e-6,
                 pool_size=(1, 1)):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.flag = flag  

        if flag != 'l':
            self.norm1 = LayerNorm2d(embed_dim, eps=1e-6)

        assert flag in ['l', 's', 'v']
        if flag == 'l':
            self.attn = DepthwiseConvTokenMixer(embed_dim, num_heads)
        elif flag == 's':
            self.attn = SalienceAttention(embed_dim, num_heads, pool_size=pool_size)
        elif flag == 'v':
            self.attn = VanillaSelfAttention(embed_dim, num_heads)

        self.drop_path = DropPath(drop_path)
        self.norm2 = LayerNorm2d(embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_value * torch.ones(1, embed_dim, 1, 1), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_value * torch.ones(1, embed_dim, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor):
        x = x + self.pos(x)

        if self.layerscale:
            if self.flag == 'l':
                x = x + self.drop_path(self.gamma_1 * self.attn(x))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            if self.flag == 'l':
                x = x + self.drop_path(self.attn(x))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        '''
        x: B C H W
        '''
        x = self.reduction(x)  # (b oc oh ow)
        x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, flags, embed_dim, out_dim, depth, num_heads,
                 ffn_dim=96., drop_path=0.,
                 downsample: PatchMerging = None,
                 layerscale=False, layer_init_value=1e-6,
                 pool_size=(1, 1)):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            Block(flags[i], embed_dim, num_heads, ffn_dim,
                  drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_value,
                  pool_size=pool_size if flags[i] == 's' else (1, 1)) 
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor):
        _, _, h, w = x.size()

        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
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
        B, C, H, W = x.shape
        x = self.proj(x)  # (b c h w)
        return x


class SVIT(nn.Module):

    def __init__(self, in_chans=3, num_classes=1000, flagss=[['l'] * 10, ['s'] * 10, ['s', 's'] * 10, ['v'] * 10],
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1,
                 projection=1024, layerscales=[False, False, False, False], layer_init_values=[1e-6, 1e-6, 1e-6, 1e-6]):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0])

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            pool_size = (8, 8) if i_layer == 0 else (2, 2) if i_layer == 1 else (1, 1) if i_layer == 2 else (
            1, 1) 
            layer = BasicLayer(
                flags=flagss[i_layer],
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                layerscale=layerscales[i_layer],
                layer_init_value=layer_init_values[i_layer],
                pool_size=pool_size 
            )
            self.layers.append(layer)

        self.proj = nn.Conv2d(self.num_features, projection, 1)
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Conv2d(projection, num_classes, 1) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.proj(x)  # (b c h w)
        x = self.norm(x)  # (b c h w)
        x = self.swish(x)

        x = self.avgpool(x)  # B C 1 1
        return x

    def forward(self, x):
        # x = F.interpolate(x, (384, 384), mode='bicubic')
        x = self.forward_features(x)
        x = self.head(x).flatten(1)
        return x


@register_model
def SVIT_T(args=None):
    model = SVIT(
        embed_dims=[48, 96, 144, 240],
        depths=[2, 2, 8, 3],
        num_heads=[1, 2, 3, 5],
        mlp_ratios=[3.5, 3.5, 3.5, 3.5],
        drop_path_rate=0.05,
        projection=1024,
        layerscales=[True, True, True, True],
        layer_init_values=[1, 1, 1, 1]
    )
    model.default_cfg = _cfg()
    return model



@register_model
def SVIT_S(args=None):
    model = SVIT(
        embed_dims=[64, 128, 280, 384],   
        depths=[3, 4, 16, 4],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[3.5, 3.5, 3.5, 3.5],
        drop_path_rate=0.15,
        projection=1024,
        layerscales=[True, True, True, True],
        layer_init_values=[1, 1, 1, 1]
    )
    model.default_cfg = _cfg()
    return model



if __name__ == "__main__":
    device = 'cuda'
    model = SVIT(
        embed_dims=[64, 128, 280, 384],
        depths=[3, 4, 16, 4],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[3.5, 3.5, 3.5, 3.5],
        drop_path_rate=0.15,
        projection=1024,
        layerscales=[True, True, True, True],
        layer_init_values=[1, 1, 1, 1]
    ).to(device)
    model.default_cfg = _cfg()

    model.eval()
    inputs = torch.randn(1, 3, 224, 224).to(device)
    model(inputs)

    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, flop_count_table

    flops = FlopCountAnalysis(model, inputs)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, inputs)

    print(f"total flops : {flops.total()}")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")

    print(flop_count_table(flops, max_depth=1))