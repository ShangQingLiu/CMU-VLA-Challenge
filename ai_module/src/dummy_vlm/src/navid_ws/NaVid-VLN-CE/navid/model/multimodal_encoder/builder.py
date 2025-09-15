import os
from pathlib import Path
from .eva_vit import EVAVisionTowerLavis

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = None
for p in _THIS.parents:
    if (p / "model_zoo").exists() and (p / "navid").exists():
        _PROJECT_ROOT = p
        break
if _PROJECT_ROOT is None:
    # 兜底：假设上三级是仓库根（builder.py 位于 navid/model/multimodal_encoder/）
    _PROJECT_ROOT = _THIS.parents[3]

def build_vision_tower(vision_tower_cfg, **kwargs):
    # 读取配置
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower',
                   getattr(vision_tower_cfg, 'vision_tower', None))
    clip_vit_larget_path = _PROJECT_ROOT / "model_zoo" / "clip-vit-large-patch14" \
    / "clip-vit-large-patch14" / "snapshots" / "32bd64288804d66eefd0ccbe215aa642df71cc41"
    image_processor = getattr(vision_tower_cfg, 'image_processor',
                      clip_vit_larget_path)

    # 统一把 image_processor 解析成绝对路径（基于项目根）
    # 如果你固定用自带的处理器目录，就指向 repo 内的路径：
    image_processor = (_PROJECT_ROOT / "navid" / "processor" / "clip-patch14-224").resolve()
    print(f"[DEBUG] image_processor resolved to: {image_processor}")

    # 统一规范 vision_tower 路径
    if vision_tower in (None, "", "./model_zoo/eva_vit_g.pth", "model_zoo/eva_vit_g.pth"):
        print(f"[DEBUG] Using default vision tower:{_PROJECT_ROOT} ")
        vision_tower = _PROJECT_ROOT / "model_zoo" / "eva_vit_g.pth"
    else:
        # 相对 -> 绝对；绝对则保持
        vision_tower = Path(vision_tower)
        print(f"[DEBUG] Original vision_tower path: {vision_tower}")
        if not vision_tower.is_absolute():
            vision_tower = (_PROJECT_ROOT / vision_tower).resolve()

    # 存在性检查
    if not vision_tower.exists():
        raise ValueError(f"Not find vision tower: {vision_tower}")

    vt_str = str(vision_tower)
    if ("lavis" in vt_str.lower()) or ("eva" in vt_str.lower()):
        return EVAVisionTowerLavis(str(vision_tower), str(image_processor),
                                   args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}")

    
