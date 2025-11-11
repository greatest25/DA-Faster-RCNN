# register_cityscapes.py
import os
from typing import Dict, Optional, Tuple

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances, register_pascal_voc

DATA_ROOT = "/mnt/lyh/DA-FasterCNN/DA-Faster-RCNN/datasets/cityscape"
NAME_TO_SPLIT: Dict[str, str] = {
    "city_trainS": "train_s",
    "city_trainT": "train_t",
    "city_testT": "test_t",
}
CLASSES = ['car', 'person', 'rider', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']


def load_pseudo_labels_custom(json_file, image_root):
    """
    加载自定义格式的伪标签JSON ({"results": [...]})
    转换为Detectron2格式
    """
    import json
    from detectron2.structures import BoxMode
    
    with open(json_file) as f:
        data = json.load(f)
    
    # 检查是否为自定义格式
    if "results" in data:
        results = data["results"]
    else:
        raise ValueError(f"Unexpected JSON format in {json_file}")
    
    # 转换为Detectron2格式
    dataset_dicts = []
    for result in results:
        if not result.get("annotations"):
            continue  # 跳过没有annotations的图像
        
        record = {
            "file_name": os.path.join(image_root, result["file_name"]),
            "image_id": result["image_id"],
            "height": result["height"],
            "width": result["width"],
            "annotations": []
        }
        
        for ann in result["annotations"]:
            obj = {
                "bbox": ann["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": ann["category_id"],
                "score": ann.get("score", 1.0)
            }
            record["annotations"].append(obj)
        
        dataset_dicts.append(record)
    
    return dataset_dicts



def _unregister(names):
    for n in names:
        if n in DatasetCatalog.list():
            DatasetCatalog.remove(n)
        if n in MetadataCatalog:
            try:
                MetadataCatalog.remove(n)
            except Exception:
                pass


def register_city_datasets(root: str = DATA_ROOT, prefer_coco: bool = False) -> str:
    """
    与 Notebook 完全一致的注册逻辑：
    1. 若 prefer_coco=True 或 VOC 目录缺失，则检查 annotations/<split>.json + images/<split>/ 是否齐全；齐全就用 COCO 注册。
    2. 否则按 VOC 目录 (VOC2007/Annotations, JPEGImages, ImageSets/Main/<split>.txt) 走 register_pascal_voc。
    返回 "coco" 或 "voc"。
    """
    _unregister(NAME_TO_SPLIT.keys())

    voc_dir = os.path.join(root, "VOC2007")
    coco_dir = os.path.join(root, "annotations")
    image_root = os.path.join(root, "images")

    def has_coco():
        for split in NAME_TO_SPLIT.values():
            ann = os.path.join(coco_dir, f"{split}.json")
            imgs = os.path.join(image_root, split)
            if not (os.path.isfile(ann) and os.path.isdir(imgs)):
                return False
        return True

    def has_voc():
        return (
            os.path.isdir(voc_dir)
            and os.path.isdir(os.path.join(voc_dir, "ImageSets", "Main"))
            and os.path.isdir(os.path.join(voc_dir, "Annotations"))
            and os.path.isdir(os.path.join(voc_dir, "JPEGImages"))
        )

    use_coco = prefer_coco and has_coco()
    if not use_coco:
        use_coco = has_coco() and not has_voc()

    if use_coco:
        for name, split in NAME_TO_SPLIT.items():
            register_coco_instances(
                name,
                {},
                os.path.join(coco_dir, f"{split}.json"),
                os.path.join(image_root, split),
            )
        fmt = "coco"
    elif has_voc():
        for name, split in NAME_TO_SPLIT.items():
            register_pascal_voc(
                name,
                voc_dir,
                split,
                2007,
                CLASSES,
            )
        fmt = "voc"
    else:
        raise FileNotFoundError(
            "既没有完整的 COCO 标注，也没有 VOC2007 结构。\n"
            "请确认 Notebook 所用的数据已同步到脚本环境，比如：\n"
            f"- COCO: {coco_dir}/train_s.json 与 {image_root}/train_s/\n"
            f"- VOC:  {voc_dir}/ImageSets/Main/train_s.txt 等"
        )

    counts = {}
    for name in NAME_TO_SPLIT:
        try:
            counts[name] = len(DatasetCatalog.get(name))
        except Exception:
            counts[name] = "N/A"
    print(f"Cityscapes 注册完成 (format={fmt}): {counts}")
    return fmt

def register_city_pseudo_dataset(
    json_file: str,
    dataset_name: str = "city_trainT_pseudo",
    image_root: str = None
):
    """
    注册包含伪标签的 Cityscapes 目标域数据集。
    
    Args:
        json_file: 伪标签 JSON 文件路径（COCO 格式）
        dataset_name: 注册的数据集名称
        image_root: 图像根目录，如果为 None 则从 JSON 推断
    """
    
    # 如果已注册则先移除
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
    if dataset_name in MetadataCatalog.list():
        MetadataCatalog.remove(dataset_name)
    
    # 推断图像根目录
    if image_root is None:
        json_path = Path(json_file)
        # 假设伪标签 JSON 在项目根目录下，图像在 datasets/cityscape/VOC2007/JPEGImages
        image_root = str(json_path.parent.parent / "datasets" / "cityscape" / "VOC2007" / "JPEGImages")
    
    # 使用自定义加载器注册
    DatasetCatalog.register(dataset_name, lambda: load_pseudo_labels_custom(json_file, image_root))
    MetadataCatalog.get(dataset_name).set(
        thing_classes=["car", "person", "rider", "truck", "bus", "train", "motorcycle", "bicycle"],
        evaluator_type="coco"
    )
    
    # 设置类别名称（与源域保持一致）
    MetadataCatalog.get(dataset_name).thing_classes = [
        'car', 'person', 'rider', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]
    
    print(f"✓ Registered pseudo-labeled dataset: {dataset_name}")
    print(f"  - JSON: {json_file}")
    print(f"  - Images: {image_root}")
    
    return dataset_name
