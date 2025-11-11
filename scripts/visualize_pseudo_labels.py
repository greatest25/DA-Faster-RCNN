#!/usr/bin/env python3
"""
可视化伪标签脚本
用途: 可视化生成的伪标签,检查质量
"""

import os
import json
import random
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_pseudo_labels(json_file):
    """加载伪标签 JSON (支持多种格式)"""
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    
    # 检查JSON格式
    if 'results' in raw_data:
        # 新格式: {"dataset": "...", "score_threshold": 0.7, "results": [...]}
        results = raw_data['results']
        annotations = []
        images_dict = {}
        
        # 转换为COCO格式
        for img_result in results:
            img_id = img_result['image_id']
            file_name = img_result['file_name']
            
            if img_id not in images_dict:
                images_dict[img_id] = {'id': img_id, 'file_name': file_name}
            
            # 使用 'annotations' 字段
            for det in img_result.get('annotations', []):
                annotations.append({
                    'image_id': img_id,
                    'bbox': det['bbox'],
                    'category_id': det['category_id'],
                    'score': det['score']
                })
        
        images = list(images_dict.values())
        
        # Cityscapes类别
        categories = [
            {'id': 0, 'name': 'person'},
            {'id': 1, 'name': 'rider'},
            {'id': 2, 'name': 'car'},
            {'id': 3, 'name': 'truck'},
            {'id': 4, 'name': 'bus'},
            {'id': 5, 'name': 'train'},
            {'id': 6, 'name': 'motorcycle'},
            {'id': 7, 'name': 'bicycle'}
        ]
        
        return {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
    else:
        # 标准COCO格式
        return raw_data
    return data


def get_image_annotations(data, image_id):
    """获取指定图像的所有标注"""
    anns = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
    return anns


def visualize_image_with_boxes(image_path, annotations, categories, save_path=None):
    """可视化图像及其边界框"""
    # 加载图像
    img = Image.open(image_path)
    
    # 创建类别 ID 到名称的映射
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    # 创建图形
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # 绘制每个标注
    colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))
    cat_id_to_color = {cat['id']: colors[i] for i, cat in enumerate(categories)}
    
    for ann in annotations:
        # 获取边界框 [x, y, width, height]
        bbox = ann['bbox']
        x, y, w, h = bbox
        
        # 获取类别和置信度
        cat_id = ann['category_id']
        cat_name = cat_id_to_name.get(cat_id, f'class_{cat_id}')
        score = ann.get('score', 1.0)
        
        # 绘制矩形
        color = cat_id_to_color.get(cat_id, 'red')
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加标签
        label = f'{cat_name}: {score:.2f}'
        ax.text(
            x, y - 5,
            label,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
            fontsize=10,
            color='white',
            weight='bold'
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存可视化结果: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='可视化伪标签')
    parser.add_argument(
        '--json',
        type=str,
        default='pseudo_labels/city_trainT_full_pseudo_thr07_coco.json',
        help='伪标签 JSON 文件路径'
    )
    parser.add_argument(
        '--image-root',
        type=str,
        default='datasets/cityscape/train_t',
        help='图像根目录'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_vis_pseudo',
        help='可视化结果输出目录'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='可视化的图像数量'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    # 加载伪标签
    print(f"加载伪标签: {args.json}")
    data = load_pseudo_labels(args.json)
    
    # 统计信息
    print(f"\n数据集统计:")
    print(f"  - 图像数量: {len(data['images'])}")
    print(f"  - 标注数量: {len(data['annotations'])}")
    print(f"  - 类别数量: {len(data['categories'])}")
    
    # 计算平均置信度
    if data['annotations']:
        scores = [ann.get('score', 1.0) for ann in data['annotations']]
        print(f"  - 平均置信度: {np.mean(scores):.3f}")
        print(f"  - 置信度范围: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 随机选择图像
    random.seed(args.random_seed)
    sample_images = random.sample(data['images'], min(args.num_samples, len(data['images'])))
    
    print(f"\n可视化 {len(sample_images)} 张图像...")
    
    # 可视化每张图像
    for i, img_info in enumerate(sample_images, 1):
        image_id = img_info['id']
        image_filename = img_info['file_name']
        
        # 构建图像路径
        if os.path.isabs(image_filename):
            image_path = image_filename
        else:
            image_path = os.path.join(args.image_root, image_filename)
        
        if not os.path.exists(image_path):
            print(f"✗ 图像不存在: {image_path}")
            continue
        
        # 获取该图像的所有标注
        annotations = get_image_annotations(data, image_id)
        
        print(f"[{i}/{len(sample_images)}] {image_filename} - {len(annotations)} 个标注")
        
        # 可视化并保存
        save_path = os.path.join(
            args.output_dir,
            f'vis_{i:03d}_{Path(image_filename).stem}.png'
        )
        visualize_image_with_boxes(
            image_path,
            annotations,
            data['categories'],
            save_path=save_path
        )
    
    print(f"\n✓ 完成! 可视化结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
