# models/backbones/ravlt_backbone.py
import torch
import torch.nn as nn
from typing import Dict, Tuple, Union
from util.lazy_load import LazyCall as L, instantiate
from util.utils import load_checkpoint
from models.backbones.base_backbone import BaseBackbone
#from models.backbones.ravlt import RAVLT  # 根据你工程里的真实相对路径调整
from models.backbones.svit import SVIT

class SVITFeatureExtractor(nn.Module):
    """
    手动展开每个 layer 的 blocks，在 downsample 之前截取特征。
    输出: { "layers.0.blocks": T0, "layers.1.blocks": T1, ... }
    """
    def __init__(self, model: SVIT, return_indices=(0,1,2,3)):
        super().__init__()
        self.model = model
        self.return_indices = set(return_indices)
        # 由于原模型未保存 embed_dims，我们动态推断
        self.stage_dims = [layer.embed_dim for layer in model.layers]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = {}
        x = self.model.patch_embed(x)  # -> (B, C0, H/4, W/4)
        for i, layer in enumerate(self.model.layers):
            # layer: BasicLayer，内部为 blocks (ModuleList) + 可选 downsample
            for blk in layer.blocks:
                x = blk(x)
            if i in self.return_indices:
                feats[f"layers.{i}.blocks"] = x
            if layer.downsample is not None:
                x = layer.downsample(x)
        return feats


class SVITBackbone(BaseBackbone):
    '''model_arch = {
        "svit_t": L(SVIT)(
            embed_dims=[48, 96, 144, 240],
            depths=[2, 2, 8, 3],
            num_heads=[1, 2, 3, 5],
            mlp_ratios=[3.5, 3.5, 3.5, 3.5],
            drop_path_rate=0.0,
            projection=1024,
            layerscales=[True, True, True, True],
            layer_init_values=[1, 1, 1, 1],
            num_classes=1000,
            url="/data1/saizhou777/S-ViT-DETR/checkpoints/svit_tiny_best.pth",
        ),
    }'''
    model_arch = {
        "svit_s": L(SVIT)(
            embed_dims=[64, 128, 280, 384],   
            depths=[3, 4, 16, 4],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[3.5, 3.5, 3.5, 3.5],
            drop_path_rate=0.15,
            projection=1024,
            layerscales=[True, True, True, True],
            layer_init_values=[1, 1, 1, 1],
            num_classes=1000,
            url="/data1/saizhou777/S-ViT-DETR/checkpoints/svit_small_best.pth",
        ),
    }


    def __new__(
        self,
        arch: str = "svit_s",
        weights: Union[str, Dict, None] = None,
        return_indices: Tuple[int] = (0,1,2,3),
        freeze_indices: Tuple[int] = (),
        **kwargs
    ):
        # 1. 构建并实例化
        model_config = self.get_instantiate_config(self, SVIT, arch, kwargs)
        default_weight = model_config.pop("url", None)
        model: SVIT = instantiate(model_config)

        # 2. 加载权重（保持最简，允许不严格）
        '''ckpt_path = weights if weights is not None else default_weight
        state = load_checkpoint(ckpt_path)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            cleaned[nk] = v
        model.load_state_dict(cleaned, strict=False)'''

        ckpt_path = weights if weights is not None else default_weight
        if ckpt_path is not None:
            raw = load_checkpoint(ckpt_path)
            # 1. 解析出真正的 state_dict
            if isinstance(raw, dict):
                if 'model_ema' in raw and isinstance(raw['model_ema'], dict) and len(raw['model_ema']) > 0:
                    state_dict = raw['model_ema']
                    source_tag = 'model_ema'
                if 'model' in raw and isinstance(raw['model'], dict):
                    state_dict = raw['model']
                    source_tag = 'model'
                elif 'state_dict' in raw and isinstance(raw['state_dict'], dict):
                    state_dict = raw['state_dict']
                    source_tag = 'state_dict'
                else:
                    # 可能本身就是纯的 state_dict
                    state_dict = raw
                    source_tag = 'raw_dict'
            else:
                raise TypeError(f"Checkpoint format not supported: {type(raw)}")

            # 2. 处理 DataParallel 前缀
            cleaned = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    nk = k[7:]
                else:
                    nk = k
                cleaned[nk] = v

            # 3. 可选：过滤掉分类头、投影尾部（如果想避免覆盖随机初始化的 Identity）
            drop_prefixes = ["head.", "proj.", "avgpool", "norm", "swish"]
            filtered = {k: v for k, v in cleaned.items()
                 if not any(k.startswith(dp) for dp in drop_prefixes)}

            # 4. 加载
            ret = model.load_state_dict(filtered, strict=False)

            if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                model_keys = set(model.state_dict().keys())
                loaded_keys = set(filtered.keys())
                matched = model_keys & loaded_keys
                missing = model_keys - loaded_keys
                unused = loaded_keys - model_keys
                print(f"[SVIT][PRETRAIN] ckpt: {ckpt_path} (use={source_tag})")
                print(f"[SVIT][PRETRAIN] matched={len(matched)} missing={len(missing)} unused={len(unused)}")
                if ret.missing_keys:
                    print("  missing(example):", ret.missing_keys[:10])
                if ret.unexpected_keys:
                    print("  unexpected(example):", ret.unexpected_keys[:10])
        else:
            if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                print("[SVIT][PRETRAIN] No checkpoint provided, random init.")

        for tail in ["proj", "norm", "head", "avgpool", "swish"]:
            if hasattr(model, tail):
                setattr(model, tail, nn.Identity())  # 或直接 delattr(model, tail)

        # 3. 冻结（按索引：与 Swin/Focal 一致；0 表示第 1 个 stage）
        if len(freeze_indices) > 0:
            SVITBackbone.freeze_module(model.patch_embed)
        for idx in freeze_indices:
            if 0 <= idx < len(model.layers):
                SVITBackbone.freeze_module(model.layers[idx])

        # 4. 包装特征提取
        extractor = SVITFeatureExtractor(model, return_indices=return_indices)

        # 5. 构造最终 backbone（不需要后处理，已是 NCHW）
        backbone = extractor
        # 动态得到各 stage 通道
        stage_dims = extractor.stage_dims
        backbone.num_channels = [stage_dims[i] for i in return_indices]
        backbone.return_layers = [f"layers.{i}.blocks" for i in return_indices]
        return backbone