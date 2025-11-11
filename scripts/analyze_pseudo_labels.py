#!/usr/bin/env python3
"""
分析伪标签质量脚本
用途: 统计分析生成的伪标签质量，生成详细报告
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


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


def analyze_confidence_distribution(annotations, output_dir=None):
    """分析置信度分布"""
    scores = [ann.get('score', 1.0) for ann in annotations]
    
    if not scores:
        print("警告: 没有标注数据")
        return
    
    # 统计信息
    print("\n置信度分布:")
    print(f"  - 平均值: {np.mean(scores):.3f}")
    print(f"  - 中位数: {np.median(scores):.3f}")
    print(f"  - 标准差: {np.std(scores):.3f}")
    print(f"  - 最小值: {np.min(scores):.3f}")
    print(f"  - 最大值: {np.max(scores):.3f}")
    
    # 分段统计
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print("\n置信度区间分布:")
    for i in range(len(bins) - 1):
        count = sum(bins[i] <= s < bins[i+1] for s in scores)
        percentage = count / len(scores) * 100
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:5d} ({percentage:5.1f}%)")
    
    # 绘制直方图
    if output_dir:
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Pseudo Label Confidence Distribution', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        save_path = Path(output_dir) / 'confidence_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 保存置信度分布图: {save_path}")
        plt.close()


def analyze_category_distribution(annotations, categories, output_dir=None):
    """分析类别分布"""
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    cat_counts = Counter(ann['category_id'] for ann in annotations)
    
    print("\n类别分布:")
    total = sum(cat_counts.values())
    
    sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
    for cat_id, count in sorted_cats:
        cat_name = cat_id_to_name.get(cat_id, f'class_{cat_id}')
        percentage = count / total * 100
        print(f"  {cat_name:12s}: {count:5d} ({percentage:5.1f}%)")
    
    # 绘制条形图
    if output_dir:
        cat_names = [cat_id_to_name.get(cid, f'class_{cid}') for cid, _ in sorted_cats]
        counts = [count for _, count in sorted_cats]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(cat_names, counts, edgecolor='black', alpha=0.7)
        
        # 为每个条形添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Pseudo Label Category Distribution', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = Path(output_dir) / 'category_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存类别分布图: {save_path}")
        plt.close()


def analyze_per_image_stats(data):
    """分析每张图像的标注统计"""
    image_id_to_anns = defaultdict(list)
    for ann in data['annotations']:
        image_id_to_anns[ann['image_id']].append(ann)
    
    ann_counts = [len(anns) for anns in image_id_to_anns.values()]
    images_without_anns = len(data['images']) - len(image_id_to_anns)
    
    print("\n每张图像标注统计:")
    print(f"  - 有标注的图像: {len(image_id_to_anns)}")
    print(f"  - 无标注的图像: {images_without_anns}")
    
    if ann_counts:
        print(f"  - 平均标注数: {np.mean(ann_counts):.2f}")
        print(f"  - 中位数: {np.median(ann_counts):.0f}")
        print(f"  - 最小值: {np.min(ann_counts)}")
        print(f"  - 最大值: {np.max(ann_counts)}")


def analyze_bbox_sizes(annotations):
    """分析边界框尺寸分布"""
    areas = []
    
    for ann in annotations:
        bbox = ann['bbox']
        w, h = bbox[2], bbox[3]
        area = w * h
        areas.append(area)
    
    if not areas:
        return
    
    print("\n边界框尺寸统计:")
    print(f"  - 平均面积: {np.mean(areas):.1f} 像素²")
    print(f"  - 中位数面积: {np.median(areas):.1f} 像素²")
    
    # 尺寸分类 (COCO 标准)
    small = sum(1 for a in areas if a < 32**2)
    medium = sum(1 for a in areas if 32**2 <= a < 96**2)
    large = sum(1 for a in areas if a >= 96**2)
    total = len(areas)
    
    print(f"\n按 COCO 标准分类:")
    print(f"  - Small  (< 32²):  {small:5d} ({small/total*100:5.1f}%)")
    print(f"  - Medium (32²-96²): {medium:5d} ({medium/total*100:5.1f}%)")
    print(f"  - Large  (≥ 96²):  {large:5d} ({large/total*100:5.1f}%)")


def generate_report(data, output_file):
    """生成文本报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("伪标签质量分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"数据集信息:\n")
        f.write(f"  - 图像总数: {len(data['images'])}\n")
        f.write(f"  - 标注总数: {len(data['annotations'])}\n")
        f.write(f"  - 类别总数: {len(data['categories'])}\n\n")
        
        if data['annotations']:
            scores = [ann.get('score', 1.0) for ann in data['annotations']]
            f.write(f"置信度统计:\n")
            f.write(f"  - 平均值: {np.mean(scores):.3f}\n")
            f.write(f"  - 中位数: {np.median(scores):.3f}\n")
            f.write(f"  - 最小值: {np.min(scores):.3f}\n")
            f.write(f"  - 最大值: {np.max(scores):.3f}\n\n")
        
        # 类别分布
        cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
        cat_counts = Counter(ann['category_id'] for ann in data['annotations'])
        
        f.write(f"类别分布:\n")
        for cat_id, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
            cat_name = cat_id_to_name.get(cat_id, f'class_{cat_id}')
            percentage = count / len(data['annotations']) * 100
            f.write(f"  {cat_name}: {count} ({percentage:.1f}%)\n")
    
    print(f"\n✓ 保存文本报告: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='分析伪标签质量')
    parser.add_argument(
        '--json',
        type=str,
        default='pseudo_labels/city_trainT_full_pseudo_thr07_coco.json',
        help='伪标签 JSON 文件路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_analysis_pseudo',
        help='分析结果输出目录'
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载伪标签: {args.json}")
    data = load_pseudo_labels(args.json)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("伪标签质量分析")
    print("=" * 60)
    
    # 基本统计
    print(f"\n数据集概览:")
    print(f"  - 图像总数: {len(data['images'])}")
    print(f"  - 标注总数: {len(data['annotations'])}")
    print(f"  - 类别总数: {len(data['categories'])}")
    
    if not data['annotations']:
        print("\n警告: 没有标注数据!")
        return
    
    # 各项分析
    analyze_confidence_distribution(data['annotations'], output_dir)
    analyze_category_distribution(data['annotations'], data['categories'], output_dir)
    analyze_per_image_stats(data)
    analyze_bbox_sizes(data['annotations'])
    
    # 生成报告
    report_file = output_dir / 'analysis_report.txt'
    generate_report(data, report_file)
    
    print("\n" + "=" * 60)
    print(f"✓ 分析完成! 结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
