# -*- coding: utf-8 -*-
"""
YOLOv11 训练器
使用最新的YOLOv11进行IM界面元素检测
"""

import os
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from tqdm import tqdm
import json
from datetime import datetime


class YOLOv11Trainer:
    """YOLOv11模型训练器"""
    
    def __init__(self, project_name: str = "im_detection"):
        """
        初始化YOLOv11训练器
        
        Args:
            project_name: 项目名称
        """
        self.project_name = project_name
        self.model = None
        self.dataset_yaml = None
        
        # YOLOv11 模型变体
        self.model_variants = {
            'nano': 'yolov11n.pt',      # 5.4MB, 最快
            'small': 'yolov11s.pt',     # 11.1MB, 快速
            'medium': 'yolov11m.pt',    # 38.8MB, 平衡
            'large': 'yolov11l.pt',     # 65.9MB, 准确
            'xlarge': 'yolov11x.pt'     # 109.3MB, 最准确
        }
        
        # 类别映射
        self.class_names = [
            'receiver_avatar',
            'receiver_name',
            'input_box',
            'send_button',
            'chat_message',
            'contact_item',
            'user_avatar'
        ]
        
    def prepare_dataset(self, labeled_data_path: str, 
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.05) -> str:
        """
        准备YOLOv11数据集
        
        Args:
            labeled_data_path: 标注数据路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            数据集配置文件路径
        """
        print("📊 准备YOLOv11数据集...")
        
        # 创建数据集目录结构
        dataset_path = Path(f"./datasets/{self.project_name}")
        for split in ['train', 'val', 'test']:
            (dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # 获取所有标注文件
        labeled_path = Path(labeled_data_path)
        image_files = list((labeled_path / 'images').glob('*.[jp][pn][g]'))
        
        # 打乱并分割数据集
        np.random.shuffle(image_files)
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'train': image_files[:n_train],
            'val': image_files[n_train:n_train+n_val],
            'test': image_files[n_train+n_val:]
        }
        
        # 复制文件到对应目录
        for split, files in splits.items():
            print(f"  {split}: {len(files)} 张图片")
            for img_file in files:
                # 复制图片
                shutil.copy2(img_file, dataset_path / 'images' / split / img_file.name)
                
                # 复制标注
                label_file = labeled_path / 'labels' / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(label_file, dataset_path / 'labels' / split / label_file.name)
        
        # 创建YOLOv11数据集配置文件
        yaml_path = dataset_path / 'data.yaml'
        data_config = {
            'path': str(dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),
            'names': self.class_names,
            
            # YOLOv11 特定配置
            'download': None,
            
            # YOLOv11 数据增强配置
            'augmentation': {
                'hsv_h': 0.015,    # HSV色调
                'hsv_s': 0.7,      # HSV饱和度
                'hsv_v': 0.4,      # HSV明度
                'degrees': 0.0,    # 旋转角度（IM界面不需要旋转）
                'translate': 0.1,  # 平移
                'scale': 0.5,      # 缩放
                'shear': 0.0,      # 剪切（IM界面不需要）
                'perspective': 0.0, # 透视变换
                'flipud': 0.0,     # 上下翻转（IM界面不需要）
                'fliplr': 0.5,     # 左右翻转
                'mosaic': 1.0,     # Mosaic增强
                'mixup': 0.0,      # MixUp增强
                'copy_paste': 0.0  # Copy-Paste增强
            }
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, allow_unicode=True)
        
        self.dataset_yaml = str(yaml_path)
        print(f"✅ 数据集准备完成: {yaml_path}")
        
        return str(yaml_path)
    
    def train_model(self, 
                   dataset_config: Optional[str] = None,
                   model_size: str = 'nano',
                   epochs: int = 100,
                   batch_size: int = 16,
                   img_size: int = 640,
                   device: str = 'auto',
                   patience: int = 50,
                   workers: int = 8,
                   resume: bool = False,
                   optimizer: str = 'AdamW',
                   lr0: float = 0.01,
                   amp: bool = True) -> YOLO:
        """
        训练YOLOv11模型
        
        Args:
            dataset_config: 数据集配置文件路径
            model_size: 模型大小 (nano/small/medium/large/xlarge)
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 输入图片尺寸
            device: 设备 ('auto', 'cpu', '0', '0,1'等)
            patience: 早停耐心值
            workers: 数据加载线程数
            resume: 是否恢复训练
            optimizer: 优化器 (SGD, Adam, AdamW, RMSProp)
            lr0: 初始学习率
            amp: 是否使用自动混合精度
            
        Returns:
            训练后的模型
        """
        if dataset_config:
            self.dataset_yaml = dataset_config
        
        if not self.dataset_yaml:
            raise ValueError("请先准备数据集或提供dataset_config")
        
        print(f"🚂 开始训练YOLOv11-{model_size}模型...")
        print(f"  数据集: {self.dataset_yaml}")
        print(f"  设备: {device}")
        print(f"  批次大小: {batch_size}")
        print(f"  训练轮数: {epochs}")
        
        # 自动选择设备
        if device == 'auto':
            device = '0' if torch.cuda.is_available() else 'cpu'
        
        # 加载预训练模型
        model_path = self.model_variants.get(model_size, 'yolov11n.pt')
        self.model = YOLO(model_path)
        
        # YOLOv11 训练参数
        results = self.model.train(
            data=self.dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            workers=workers,
            patience=patience,
            
            # 优化器设置
            optimizer=optimizer,
            lr0=lr0,               # 初始学习率
            lrf=0.01,             # 最终学习率因子
            momentum=0.937,        # SGD动量/Adam beta1
            weight_decay=0.0005,   # 权重衰减
            warmup_epochs=3.0,     # 预热轮数
            warmup_momentum=0.8,   # 预热动量
            warmup_bias_lr=0.1,    # 预热偏置学习率
            
            # 损失函数权重 (YOLOv11优化)
            box=7.5,              # 边界框损失权重
            cls=0.5,              # 分类损失权重
            dfl=1.5,              # 分布式焦点损失权重 (v11新增)
            
            # 数据增强 (YOLOv11增强)
            hsv_h=0.015,          # HSV色调增强
            hsv_s=0.7,            # HSV饱和度增强
            hsv_v=0.4,            # HSV明度增强
            degrees=0.0,          # 旋转角度
            translate=0.1,        # 平移
            scale=0.5,            # 缩放
            shear=0.0,            # 剪切
            perspective=0.0,      # 透视
            flipud=0.0,           # 上下翻转
            fliplr=0.5,           # 左右翻转
            bgr=0.0,              # BGR通道翻转概率
            mosaic=1.0,           # Mosaic增强
            mixup=0.0,            # MixUp增强
            copy_paste=0.0,       # Copy-Paste增强
            auto_augment='randaugment',  # 自动增强策略 (v11新增)
            erasing=0.0,          # 随机擦除概率 (v11新增)
            
            # 训练设置
            pretrained=True,      # 使用预训练权重
            resume=resume,        # 恢复训练
            amp=amp,              # 自动混合精度
            fraction=1.0,         # 数据集使用比例
            profile=False,        # 性能分析
            freeze=None,          # 冻结层数
            
            # 多GPU设置
            multi_scale=False,    # 多尺度训练
            single_cls=False,     # 单类别训练
            
            # NMS设置
            nms_time_limit=10.0,  # NMS时间限制
            
            # 保存和日志
            save=True,            # 保存检查点
            save_period=-1,       # 保存间隔
            cache=False,          # 缓存图片到内存
            plots=True,           # 绘制训练图表
            
            # 验证设置
            val_period=1,         # 验证间隔
            sync_bn=False,        # 同步批归一化
            
            # 早停设置
            close_mosaic=10,      # 最后N轮关闭Mosaic
            
            # 项目设置
            project=f'runs/{self.project_name}',
            name=f'train_{model_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            exist_ok=False,
            
            # 其他
            seed=0,               # 随机种子
            deterministic=True,   # 确定性训练
            rect=False,           # 矩形训练
            cos_lr=False,         # 余弦学习率调度
            overlap_mask=True,    # 掩码重叠 (v11)
            mask_ratio=4,         # 掩码下采样比例 (v11)
            dropout=0.0,          # Dropout概率 (v11)
            verbose=True          # 详细输出
        )
        
        print(f"✅ 训练完成！")
        print(f"  最佳模型: runs/{self.project_name}/train_{model_size}_*/weights/best.pt")
        
        return self.model
    
    def predict(self, 
               source,
               model_path: Optional[str] = None,
               conf: float = 0.25,
               iou: float = 0.45,
               img_size: int = 640,
               device: str = 'auto',
               save: bool = False,
               save_txt: bool = False,
               save_conf: bool = False,
               save_crop: bool = False,
               show: bool = False,
               stream: bool = False,
               verbose: bool = True,
               half: bool = False,
               max_det: int = 300,
               vid_stride: int = 1,
               line_width: Optional[int] = None,
               visualize: bool = False,
               augment: bool = False,
               agnostic_nms: bool = False,
               classes: Optional[List[int]] = None,
               retina_masks: bool = False) -> List:
        """
        使用YOLOv11进行预测
        
        Args:
            source: 图片路径、视频路径、目录路径、URL等
            model_path: 模型路径（如果为None，使用self.model）
            conf: 置信度阈值
            iou: NMS的IoU阈值
            img_size: 推理图片尺寸
            device: 设备
            save: 是否保存预测结果
            save_txt: 是否保存文本结果
            save_conf: 是否在文本中保存置信度
            save_crop: 是否保存裁剪的检测框
            show: 是否显示结果
            stream: 是否使用流式处理（用于视频）
            verbose: 是否输出详细信息
            half: 是否使用FP16推理
            max_det: 每张图片最大检测数
            vid_stride: 视频帧步长
            line_width: 边界框线宽
            visualize: 是否可视化特征图
            augment: 是否使用测试时增强
            agnostic_nms: 是否使用类别无关的NMS
            classes: 只检测指定类别
            retina_masks: 是否使用高分辨率掩码
            
        Returns:
            预测结果列表
        """
        # 加载模型
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("请提供模型路径或先训练模型")
        
        # 自动选择设备
        if device == 'auto':
            device = '0' if torch.cuda.is_available() else 'cpu'
        
        # YOLOv11 预测
        results = model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=img_size,
            device=device,
            save=save,
            save_txt=save_txt,
            save_conf=save_conf,
            save_crop=save_crop,
            show=show,
            stream=stream,
            verbose=verbose,
            half=half,
            max_det=max_det,
            vid_stride=vid_stride,
            line_width=line_width,
            visualize=visualize,
            augment=augment,
            agnostic_nms=agnostic_nms,
            classes=classes,
            retina_masks=retina_masks,
            
            # YOLOv11 特定参数
            embed=None,           # 特征嵌入层
            project=f'runs/{self.project_name}',
            name='predict',
            exist_ok=True
        )
        
        return results
    
    def predict_single(self, image_path: str, 
                      model_path: Optional[str] = None,
                      return_format: str = 'dict') -> Dict:
        """
        预测单张图片（简化接口）
        
        Args:
            image_path: 图片路径
            model_path: 模型路径
            return_format: 返回格式 ('dict', 'json', 'yolo')
            
        Returns:
            检测结果字典
        """
        results = self.predict(
            source=image_path,
            model_path=model_path,
            verbose=False,
            stream=False
        )
        
        if not results:
            return {"objects": []}
        
        result = results[0]
        
        if return_format == 'yolo':
            return result
        
        # 转换为字典格式
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            
            detections.append({
                'class': self.class_names[cls],
                'class_id': cls,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': round(conf, 3)
            })
        
        output = {
            'image_path': image_path,
            'image_size': result.orig_shape,
            'objects': detections,
            'count': len(detections)
        }
        
        if return_format == 'json':
            return json.dumps(output, ensure_ascii=False, indent=2)
        
        return output
    
    def predict_batch(self, image_dir: str, 
                     model_path: Optional[str] = None,
                     save_results: bool = True) -> List[Dict]:
        """
        批量预测
        
        Args:
            image_dir: 图片目录
            model_path: 模型路径
            save_results: 是否保存结果
            
        Returns:
            所有预测结果列表
        """
        image_paths = list(Path(image_dir).glob('*.[jp][pn][g]'))
        
        all_results = []
        
        print(f"🔍 批量预测 {len(image_paths)} 张图片...")
        
        for img_path in tqdm(image_paths, desc="预测进度"):
            result = self.predict_single(str(img_path), model_path)
            all_results.append(result)
        
        if save_results:
            output_file = Path(image_dir) / 'predictions.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"✅ 结果已保存: {output_file}")
        
        # 统计
        total_objects = sum(r['count'] for r in all_results)
        avg_objects = total_objects / len(all_results) if all_results else 0
        
        print(f"\n📊 预测统计:")
        print(f"  总图片数: {len(all_results)}")
        print(f"  总检测数: {total_objects}")
        print(f"  平均每张: {avg_objects:.1f} 个对象")
        
        return all_results
    
    def evaluate_model(self, model_path: Optional[str] = None) -> Dict:
        """
        评估模型性能
        
        Args:
            model_path: 模型路径
            
        Returns:
            评估指标字典
        """
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("请提供模型路径或先训练模型")
        
        print("📈 评估模型性能...")
        
        # 在验证集上评估
        metrics = model.val(
            data=self.dataset_yaml,
            batch=16,
            imgsz=640,
            conf=0.001,
            iou=0.6,
            device='0' if torch.cuda.is_available() else 'cpu',
            
            # YOLOv11 评估参数
            plots=True,
            save_json=True,
            save_hybrid=False,
            max_det=300,
            half=True,
            dnn=False,
            verbose=True
        )
        
        # 提取关键指标
        results = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'f1_score': 2 * float(metrics.box.mp) * float(metrics.box.mr) / 
                       (float(metrics.box.mp) + float(metrics.box.mr) + 1e-6)
        }
        
        # 每个类别的性能
        class_metrics = {}
        for i, cls_name in enumerate(self.class_names):
            class_metrics[cls_name] = {
                'AP50': float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0,
                'AP': float(metrics.box.ap[i]) if i < len(metrics.box.ap) else 0
            }
        
        results['per_class'] = class_metrics
        
        print(f"\n📊 评估结果:")
        print(f"  mAP@50: {results['mAP50']:.3f}")
        print(f"  mAP@50-95: {results['mAP50-95']:.3f}")
        print(f"  Precision: {results['precision']:.3f}")
        print(f"  Recall: {results['recall']:.3f}")
        print(f"  F1-Score: {results['f1_score']:.3f}")
        
        return results
    
    def export_model(self, model_path: Optional[str] = None,
                    formats: List[str] = ['onnx']) -> Dict[str, str]:
        """
        导出模型为其他格式
        
        Args:
            model_path: 模型路径
            formats: 导出格式列表 
                    ['onnx', 'torchscript', 'coreml', 'tflite', 
                     'paddle', 'ncnn', 'engine']
                     
        Returns:
            导出文件路径字典
        """
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("请提供模型路径或先训练模型")
        
        exported_paths = {}
        
        for fmt in formats:
            print(f"📦 导出为 {fmt} 格式...")
            
            try:
                path = model.export(
                    format=fmt,
                    imgsz=640,
                    half=False,
                    dynamic=True if fmt in ['onnx', 'engine'] else False,
                    simplify=True if fmt == 'onnx' else False,
                    opset=17 if fmt == 'onnx' else None,
                    workspace=4 if fmt == 'engine' else None,
                    nms=False,
                    batch=1
                )
                exported_paths[fmt] = path
                print(f"  ✅ {fmt}: {path}")
                
            except Exception as e:
                print(f"  ❌ {fmt} 导出失败: {e}")
        
        return exported_paths
    
    def benchmark(self, model_path: Optional[str] = None,
                 test_images: Optional[str] = None,
                 num_images: int = 100) -> Dict:
        """
        性能基准测试
        
        Args:
            model_path: 模型路径
            test_images: 测试图片目录
            num_images: 测试图片数量
            
        Returns:
            性能指标
        """
        import time
        
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("请提供模型路径或先训练模型")
        
        # 获取测试图片
        if test_images:
            img_paths = list(Path(test_images).glob('*.[jp][pn][g]'))[:num_images]
        else:
            # 使用验证集
            dataset_path = Path(self.dataset_yaml).parent
            img_paths = list((dataset_path / 'images' / 'val').glob('*.[jp][pn][g]'))[:num_images]
        
        if not img_paths:
            raise ValueError("没有找到测试图片")
        
        print(f"⚡ 性能基准测试 ({len(img_paths)} 张图片)...")
        
        # 预热
        for _ in range(3):
            _ = model.predict(str(img_paths[0]), verbose=False)
        
        # 测试
        times = []
        for img_path in tqdm(img_paths, desc="基准测试"):
            start = time.time()
            _ = model.predict(str(img_path), verbose=False)
            times.append(time.time() - start)
        
        # 计算统计
        times = np.array(times)
        
        results = {
            'num_images': len(img_paths),
            'avg_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'fps': float(1.0 / np.mean(times)),
            'ms_per_image': float(np.mean(times) * 1000)
        }
        
        print(f"\n📊 性能结果:")
        print(f"  平均耗时: {results['ms_per_image']:.2f} ms")
        print(f"  FPS: {results['fps']:.1f}")
        print(f"  最快: {results['min_time']*1000:.2f} ms")
        print(f"  最慢: {results['max_time']*1000:.2f} ms")
        
        return results


def demo():
    """演示YOLOv11训练和预测"""
    
    trainer = YOLOv11Trainer("im_detection_v11")
    
    # 1. 准备数据集
    dataset_yaml = trainer.prepare_dataset("./data/labeled_data")
    
    # 2. 训练模型
    model = trainer.train_model(
        dataset_config=dataset_yaml,
        model_size='nano',    # 使用nano版本快速训练
        epochs=50,            # 演示用50轮
        batch_size=16,
        device='auto',
        patience=20,
        amp=True             # 使用混合精度加速
    )
    
    # 3. 预测单张图片
    result = trainer.predict_single(
        "./test_images/test1.jpg",
        return_format='dict'
    )
    print(f"\n检测结果: {result}")
    
    # 4. 批量预测
    batch_results = trainer.predict_batch(
        "./test_images",
        save_results=True
    )
    
    # 5. 评估模型
    metrics = trainer.evaluate_model()
    
    # 6. 导出模型
    exported = trainer.export_model(formats=['onnx'])
    
    # 7. 性能测试
    benchmark = trainer.benchmark(num_images=50)
    
    print("\n🎉 YOLOv11演示完成！")


if __name__ == "__main__":
    demo()
