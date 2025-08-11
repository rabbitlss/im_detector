# -*- coding: utf-8 -*-
"""
YOLO训练器
"""

import os
import yaml
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import random

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("请安装ultralytics: pip install ultralytics")
    exit(1)

from config import CLASS_NAMES, TRAINING_CONFIG, PATHS


class YOLOTrainer:
    """基于自动标注数据训练YOLO模型"""
    
    def __init__(self):
        """初始化训练器"""
        self.classes = CLASS_NAMES
        self.model = None
        self.training_config = TRAINING_CONFIG.copy()
        
    def prepare_dataset(self, labeled_data_folder: str, 
                       train_ratio: float = 0.8) -> str:
        """
        准备训练数据集
        
        Args:
            labeled_data_folder: 标注数据文件夹
            train_ratio: 训练集比例
            
        Returns:
            数据集配置文件路径
        """
        print("📁 准备数据集...")
        
        # 创建目录结构
        for path in PATHS.values():
            os.makedirs(path, exist_ok=True)
        
        # 获取所有已标注的图片
        images_folder = f"{labeled_data_folder}/images"
        labels_folder = f"{labeled_data_folder}/labels"
        
        if not os.path.exists(images_folder) or not os.path.exists(labels_folder):
            raise ValueError(f"标注数据文件夹不存在: {images_folder} 或 {labels_folder}")
        
        # 获取有效的图片-标注对
        valid_pairs = self._get_valid_pairs(images_folder, labels_folder)
        
        if len(valid_pairs) == 0:
            raise ValueError("未找到有效的图片-标注对")
        
        print(f"找到 {len(valid_pairs)} 对有效的图片-标注数据")
        
        # 随机打乱并分割数据
        random.shuffle(valid_pairs)
        train_count = int(len(valid_pairs) * train_ratio)
        
        train_pairs = valid_pairs[:train_count]
        val_pairs = valid_pairs[train_count:]
        
        print(f"训练集: {len(train_pairs)} 张, 验证集: {len(val_pairs)} 张")
        
        # 复制文件到对应目录
        self._copy_dataset_files(train_pairs, PATHS['train_images'], PATHS['train_labels'])
        self._copy_dataset_files(val_pairs, PATHS['val_images'], PATHS['val_labels'])
        
        # 创建数据集配置文件
        dataset_config = {
            'train': os.path.abspath(PATHS['train_images']),
            'val': os.path.abspath(PATHS['val_images']),
            'nc': len(self.classes),
            'names': self.classes
        }
        
        config_path = './im_dataset.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ 数据集准备完成，配置文件: {config_path}")
        return config_path
    
    def _get_valid_pairs(self, images_folder: str, labels_folder: str) -> List[tuple]:
        """获取有效的图片-标注对"""
        valid_pairs = []
        
        for img_file in Path(images_folder).glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                label_file = Path(labels_folder) / f"{img_file.stem}.txt"
                
                if label_file.exists() and self._validate_label_file(str(label_file)):
                    valid_pairs.append((str(img_file), str(label_file)))
        
        return valid_pairs
    
    def _validate_label_file(self, label_path: str) -> bool:
        """验证标注文件的有效性"""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) == 0:
                return False
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    return False
                
                class_id = int(parts[0])
                if class_id >= len(self.classes):
                    return False
                
                # 检查坐标范围
                coords = [float(x) for x in parts[1:]]
                if not all(0 <= x <= 1 for x in coords):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _copy_dataset_files(self, pairs: List[tuple], 
                           target_images: str, target_labels: str) -> None:
        """复制数据集文件"""
        for img_path, label_path in pairs:
            # 复制图片
            img_name = Path(img_path).name
            shutil.copy2(img_path, os.path.join(target_images, img_name))
            
            # 复制标注
            label_name = Path(label_path).name
            shutil.copy2(label_path, os.path.join(target_labels, label_name))
    
    def train_model(self, dataset_config: str, 
                   model_type: str = 'yolov8n',
                   pretrained: bool = True,
                   **kwargs) -> YOLO:
        """
        训练YOLO模型
        
        Args:
            dataset_config: 数据集配置文件路径
            model_type: 模型类型 ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
            pretrained: 是否使用预训练模型
            **kwargs: 其他训练参数
            
        Returns:
            训练后的模型
        """
        print("🚂 开始训练YOLO模型...")
        
        # 更新训练配置
        config = self.training_config.copy()
        config.update(kwargs)
        
        # 初始化模型
        if pretrained:
            model_path = f"{model_type}.pt"
            print(f"使用预训练模型: {model_path}")
        else:
            model_path = f"{model_type}.yaml"
            print(f"从头开始训练: {model_path}")
        
        self.model = YOLO(model_path)
        
        # 开始训练
        results = self.model.train(
            data=dataset_config,
            epochs=config['epochs'],
            imgsz=config['img_size'],
            batch=config['batch_size'],
            patience=config['patience'],
            workers=config['workers'],
            device=config['device'],
            name='im_detector',
            project=PATHS['results'],
            save=True,
            plots=True,
            verbose=True
        )
        
        print("✅ 训练完成!")
        return self.model
    
    def evaluate_model(self, test_images_folder: str = None) -> Dict:
        """
        评估模型效果
        
        Args:
            test_images_folder: 测试图片文件夹
            
        Returns:
            评估指标字典
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train_model()")
        
        print("📊 评估模型性能...")
        
        # 验证集评估
        val_results = self.model.val()
        
        metrics = {
            'mAP50': float(val_results.box.map50),
            'mAP50-95': float(val_results.box.map),
            'precision': float(val_results.box.mp),
            'recall': float(val_results.box.mr),
        }
        
        # 如果提供了测试图片，计算推理时间
        if test_images_folder and os.path.exists(test_images_folder):
            inference_times = []
            test_images = list(Path(test_images_folder).glob("*.jpg"))[:10]  # 测试10张
            
            for img_path in test_images:
                start_time = time.time()
                _ = self.model(str(img_path))
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
            
            avg_inference_time = sum(inference_times) / len(inference_times)
            metrics['avg_inference_time_ms'] = avg_inference_time * 1000
            metrics['fps'] = 1.0 / avg_inference_time
        
        print("📈 评估结果:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return metrics
    
    def export_model(self, formats: List[str] = None) -> Dict[str, str]:
        """
        导出模型为不同格式
        
        Args:
            formats: 导出格式列表，如 ['onnx', 'tensorrt', 'openvino']
            
        Returns:
            导出文件路径字典
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train_model()")
        
        if formats is None:
            formats = ['onnx']
        
        print(f"📦 导出模型格式: {formats}")
        
        export_paths = {}
        
        for format_name in formats:
            try:
                export_path = self.model.export(format=format_name)
                export_paths[format_name] = export_path
                print(f"✅ {format_name.upper()} 导出成功: {export_path}")
                
            except Exception as e:
                print(f"❌ {format_name.upper()} 导出失败: {e}")
        
        return export_paths
    
    def save_model(self, save_path: str = None) -> str:
        """
        保存训练好的模型
        
        Args:
            save_path: 保存路径
            
        Returns:
            实际保存路径
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train_model()")
        
        if save_path is None:
            save_path = os.path.join(PATHS['models'], 'best_im_detector.pt')
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 复制最佳模型
        best_model_path = self.model.trainer.best  # 训练过程中的最佳模型
        shutil.copy2(best_model_path, save_path)
        
        print(f"💾 模型已保存: {save_path}")
        return save_path


def main():
    """测试训练功能"""
    
    # 初始化训练器
    trainer = YOLOTrainer()
    
    try:
        # 1. 准备数据集
        dataset_config = trainer.prepare_dataset('./data/labeled_data/train')
        
        # 2. 训练模型
        model = trainer.train_model(
            dataset_config=dataset_config,
            model_type='yolov8n',  # 使用最小的模型快速测试
            epochs=50,  # 减少epoch用于测试
            batch_size=8
        )
        
        # 3. 评估模型
        metrics = trainer.evaluate_model()
        
        # 4. 导出模型
        export_paths = trainer.export_model(['onnx'])
        
        # 5. 保存模型
        saved_path = trainer.save_model()
        
        print("\n🎉 训练流程完成!")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")


if __name__ == "__main__":
    main()
