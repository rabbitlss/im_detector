# -*- coding: utf-8 -*-
"""
端到端IM检测管道：自动标注 → 训练 → 部署
"""

import os
import time
from typing import Dict, List, Optional
import shutil
from pathlib import Path

from auto_labeler import GPT4VAutoLabeler
from yolo_trainer import YOLOTrainer
from im_detector import IMDetector
from config import PATHS


class End2EndIMDetector:
    """端到端IM检测系统"""
    
    def __init__(self):
        """初始化端到端系统"""
        self.labeler = GPT4VAutoLabeler()
        self.trainer = YOLOTrainer()
        self.detector = None
        self.model_path = None
        
        # 创建必要的目录
        for path in PATHS.values():
            os.makedirs(path, exist_ok=True)
    
    def build_detector(self, 
                      raw_images_folder: str,
                      max_images: int = None,
                      train_epochs: int = 100,
                      model_type: str = 'yolov8n') -> str:
        """
        构建检测器：完整流程
        
        Args:
            raw_images_folder: 原始图片文件夹
            max_images: 最大处理图片数量
            train_epochs: 训练轮数
            model_type: YOLO模型类型
            
        Returns:
            训练好的模型路径
        """
        print("🚀 开始端到端IM检测器构建...")
        print("=" * 50)
        
        # 检查原始图片
        if not os.path.exists(raw_images_folder):
            raise ValueError(f"原始图片文件夹不存在: {raw_images_folder}")
        
        image_files = list(Path(raw_images_folder).glob("*.jpg")) + \
                     list(Path(raw_images_folder).glob("*.png"))
        
        if len(image_files) == 0:
            raise ValueError(f"在 {raw_images_folder} 中未找到图片文件")
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"📁 找到 {len(image_files)} 张原始图片")
        
        try:
            # 阶段1: 自动标注
            print("\n🏷️  阶段1: GPT-4V自动标注数据...")
            print("-" * 30)
            
            labeled_folder = PATHS['labeled_data'] + '/raw_labeled'
            self.labeler.batch_labeling(
                raw_images_folder, 
                labeled_folder,
                max_images=max_images
            )
            
            # 验证标注结果
            labeled_images = list(Path(f"{labeled_folder}/images").glob("*"))
            print(f"✅ 标注完成，成功标注 {len(labeled_images)} 张图片")
            
            if len(labeled_images) == 0:
                raise ValueError("标注失败，没有有效的标注数据")
            
            # 阶段2: 训练模型
            print("\n🚂 阶段2: 训练YOLO模型...")
            print("-" * 30)
            
            dataset_config = self.trainer.prepare_dataset(labeled_folder)
            
            model = self.trainer.train_model(
                dataset_config=dataset_config,
                model_type=model_type,
                epochs=train_epochs,
                pretrained=True
            )
            
            # 评估模型
            print("\n📊 阶段3: 评估模型性能...")
            print("-" * 30)
            
            metrics = self.trainer.evaluate_model()
            
            # 保存模型
            model_name = f"im_detector_{model_type}_e{train_epochs}.pt"
            self.model_path = self.trainer.save_model(
                os.path.join(PATHS['models'], model_name)
            )
            
            # 阶段3: 模型优化
            print("\n⚡ 阶段4: 模型优化...")
            print("-" * 30)
            
            export_paths = self.trainer.export_model(['onnx'])
            
            # 初始化检测器
            self.detector = IMDetector(self.model_path)
            
            print("\n✅ 端到端构建完成!")
            print("=" * 50)
            print(f"📄 训练报告:")
            print(f"  - 使用图片数量: {len(image_files)}")
            print(f"  - 成功标注数量: {len(labeled_images)}")
            print(f"  - 模型类型: {model_type}")
            print(f"  - 训练轮数: {train_epochs}")
            print(f"  - mAP50: {metrics.get('mAP50', 'N/A'):.4f}")
            print(f"  - 模型路径: {self.model_path}")
            print(f"  - ONNX路径: {export_paths.get('onnx', 'N/A')}")
            
            return self.model_path
            
        except Exception as e:
            print(f"\n❌ 构建失败: {e}")
            raise
    
    def quick_build(self, raw_images_folder: str, num_images: int = 50) -> str:
        """
        快速构建（用于测试和原型）
        
        Args:
            raw_images_folder: 原始图片文件夹
            num_images: 图片数量（建议50-100张）
            
        Returns:
            模型路径
        """
        print("🚀 快速构建模式（适合测试）")
        
        return self.build_detector(
            raw_images_folder=raw_images_folder,
            max_images=num_images,
            train_epochs=50,  # 快速训练
            model_type='yolov8n'  # 最小模型
        )
    
    def production_build(self, raw_images_folder: str, num_images: int = 500) -> str:
        """
        生产级构建
        
        Args:
            raw_images_folder: 原始图片文件夹
            num_images: 图片数量（建议300-1000张）
            
        Returns:
            模型路径
        """
        print("🏭 生产级构建模式")
        
        return self.build_detector(
            raw_images_folder=raw_images_folder,
            max_images=num_images,
            train_epochs=200,  # 充分训练
            model_type='yolov8s'  # 平衡性能和精度
        )
    
    def predict(self, image_path: str, return_image: bool = False) -> Dict:
        """
        使用训练好的模型进行预测
        
        Args:
            image_path: 图片路径
            return_image: 是否返回标注图片
            
        Returns:
            检测结果
        """
        if self.detector is None:
            if self.model_path and os.path.exists(self.model_path):
                self.detector = IMDetector(self.model_path)
            else:
                raise ValueError("检测器未构建，请先运行build_detector()")
        
        return self.detector.predict(image_path, return_image=return_image)
    
    def extract_im_info(self, image_path: str) -> Dict:
        """
        提取IM关键信息
        
        Args:
            image_path: 图片路径
            
        Returns:
            IM信息字典
        """
        if self.detector is None:
            if self.model_path and os.path.exists(self.model_path):
                self.detector = IMDetector(self.model_path)
            else:
                raise ValueError("检测器未构建，请先运行build_detector()")
        
        return self.detector.extract_im_info(image_path)
    
    def benchmark(self, test_images_folder: str, num_images: int = 20) -> Dict:
        """
        性能基准测试
        
        Args:
            test_images_folder: 测试图片文件夹
            num_images: 测试图片数量
            
        Returns:
            性能指标
        """
        if self.detector is None:
            if self.model_path and os.path.exists(self.model_path):
                self.detector = IMDetector(self.model_path)
            else:
                raise ValueError("检测器未构建，请先运行build_detector()")
        
        return self.detector.benchmark(test_images_folder, num_images)
    
    def load_pretrained_model(self, model_path: str) -> None:
        """
        加载预训练模型
        
        Args:
            model_path: 模型文件路径
        """
        if not os.path.exists(model_path):
            raise ValueError(f"模型文件不存在: {model_path}")
        
        self.model_path = model_path
        self.detector = IMDetector(model_path)
        print(f"✅ 预训练模型加载成功: {model_path}")
    
    def save_demo_results(self, test_images_folder: str, 
                         output_folder: str = './demo_results') -> None:
        """
        保存演示结果
        
        Args:
            test_images_folder: 测试图片文件夹
            output_folder: 输出文件夹
        """
        if self.detector is None:
            raise ValueError("检测器未构建")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取测试图片
        test_images = list(Path(test_images_folder).glob("*.jpg"))[:10]
        
        print(f"💾 保存演示结果到: {output_folder}")
        
        for i, img_path in enumerate(test_images):
            # 检测
            result = self.detector.predict(str(img_path), return_image=True)
            
            # 保存标注图片
            if 'annotated_image' in result and result['annotated_image'] is not None:
                output_path = os.path.join(output_folder, f"demo_{i+1}.jpg")
                cv2.imwrite(output_path, result['annotated_image'])
            
            # 保存检测信息
            im_info = self.detector.extract_im_info(str(img_path))
            
            info_file = os.path.join(output_folder, f"info_{i+1}.txt")
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"图片: {img_path.name}\n")
                f.write(f"接收者: {'检测到' if im_info['receiver'] else '未检测到'}\n")
                f.write(f"输入框: {'检测到' if im_info['input_text'] else '未检测到'}\n")
                f.write(f"联系人数量: {len(im_info['contacts'])}\n")
                f.write(f"消息数量: {len(im_info['messages'])}\n")
                
                if 'metadata' in result:
                    metadata = result['metadata']
                    f.write(f"推理时间: {metadata['inference_time_ms']:.2f} ms\n")
                    f.write(f"FPS: {metadata['fps']:.2f}\n")
        
        print(f"✅ 演示结果已保存，共 {len(test_images)} 张")


def main():
    """主函数 - 演示完整流程"""
    
    # 检查原始数据
    raw_images_folder = "./data/raw_images"
    
    if not os.path.exists(raw_images_folder):
        print(f"❌ 原始图片文件夹不存在: {raw_images_folder}")
        print("请将IM截图放到该文件夹中")
        print("支持的格式: .jpg, .jpeg, .png")
        return
    
    # 初始化端到端系统
    pipeline = End2EndIMDetector()
    
    try:
        # 选择构建模式
        mode = input("选择模式 (1: 快速测试, 2: 生产级构建): ").strip()
        
        if mode == "1":
            print("\n🔥 开始快速构建...")
            model_path = pipeline.quick_build(raw_images_folder, num_images=20)
        
        elif mode == "2":
            print("\n🏭 开始生产级构建...")
            model_path = pipeline.production_build(raw_images_folder, num_images=100)
        
        else:
            print("使用默认快速模式...")
            model_path = pipeline.quick_build(raw_images_folder, num_images=20)
        
        # 测试检测器
        test_images = list(Path(raw_images_folder).glob("*.jpg"))
        if test_images:
            print(f"\n🔍 测试检测功能...")
            test_image = str(test_images[0])
            
            # 检测测试
            result = pipeline.predict(test_image, return_image=True)
            print("检测结果:")
            for class_name, objects in result.items():
                if class_name not in ['metadata', 'annotated_image']:
                    print(f"  {class_name}: {len(objects)} 个")
            
            # 提取信息测试
            im_info = pipeline.extract_im_info(test_image)
            print("\nIM信息提取:")
            print(f"  接收者: {'✅' if im_info['receiver'] else '❌'}")
            print(f"  输入框: {'✅' if im_info['input_text'] else '❌'}")
            print(f"  联系人: {len(im_info['contacts'])} 个")
            
            # 性能测试
            print("\n⚡ 性能测试...")
            metrics = pipeline.benchmark(raw_images_folder, num_images=5)
            
            # 保存演示结果
            pipeline.save_demo_results(raw_images_folder)
        
        print(f"\n🎉 完成！模型已保存: {model_path}")
        
    except KeyboardInterrupt:
        print("\n👋 用户取消操作")
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()


def demo_usage():
    """演示如何使用已训练的模型"""
    
    print("📖 使用预训练模型演示")
    
    # 加载预训练模型
    model_path = "./models/best_im_detector.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行训练流程生成模型")
        return
    
    pipeline = End2EndIMDetector()
    pipeline.load_pretrained_model(model_path)
    
    # 测试图片
    test_image = "./data/test_image.jpg"
    if os.path.exists(test_image):
        # 快速检测
        start_time = time.time()
        result = pipeline.predict(test_image)
        inference_time = time.time() - start_time
        
        print(f"⚡ 推理时间: {inference_time*1000:.2f} ms")
        print("🎯 检测结果:")
        
        for class_name, objects in result.items():
            if class_name != 'metadata':
                print(f"  {class_name}: {len(objects)} 个")
        
        # 提取关键信息
        im_info = pipeline.extract_im_info(test_image)
        
        print("\n📋 提取的IM信息:")
        if im_info['receiver']:
            print(f"  接收者: 检测成功")
        if im_info['input_text']:
            print(f"  输入框: 检测成功")
        print(f"  联系人: {len(im_info['contacts'])} 个")
        print(f"  消息: {len(im_info['messages'])} 条")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_usage()
    else:
        main()
