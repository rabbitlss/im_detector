# -*- coding: utf-8 -*-
"""
IM检测器 - 快速推理模块
"""

import os
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("请安装ultralytics: pip install ultralytics")
    exit(1)

from config import CLASS_NAMES


class IMDetector:
    """IM界面元素检测器"""
    
    def __init__(self, model_path: str = None, confidence: float = 0.5):
        """
        初始化检测器
        
        Args:
            model_path: 模型文件路径
            confidence: 置信度阈值
        """
        self.classes = CLASS_NAMES
        self.confidence = confidence
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
        """
        try:
            self.model = YOLO(model_path)
            print(f"✅ 模型加载成功: {model_path}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def predict(self, image_path: str, return_image: bool = False) -> Dict:
        """
        对单张图片进行检测
        
        Args:
            image_path: 图片路径
            return_image: 是否返回标注后的图片
            
        Returns:
            检测结果字典
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model()")
        
        # 推理
        start_time = time.time()
        results = self.model(image_path, conf=self.confidence)
        inference_time = time.time() - start_time
        
        # 解析结果
        detections = self._parse_results(results)
        
        # 添加元数据
        detections['metadata'] = {
            'inference_time_ms': inference_time * 1000,
            'fps': 1.0 / inference_time,
            'image_path': image_path,
            'confidence_threshold': self.confidence
        }
        
        # 可选: 返回标注图片
        if return_image:
            annotated_image = self._annotate_image(image_path, detections)
            detections['annotated_image'] = annotated_image
        
        return detections
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        批量检测
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            检测结果列表
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model()")
        
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"检测失败 {image_path}: {e}")
                results.append({'error': str(e), 'image_path': image_path})
        
        return results
    
    def _parse_results(self, results) -> Dict:
        """解析YOLO检测结果"""
        detections = {}
        
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                class_id = int(box.cls)
                if class_id >= len(self.classes):
                    continue
                    
                class_name = self.classes[class_id]
                confidence = float(box.conf)
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                if class_name not in detections:
                    detections[class_name] = []
                
                detections[class_name].append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'center': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                })
        
        # 按置信度排序
        for class_name in detections:
            detections[class_name].sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def _annotate_image(self, image_path: str, detections: Dict) -> np.ndarray:
        """在图片上标注检测结果"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # 颜色配置 (BGR格式)
        colors = {
            'receiver_avatar': (0, 255, 0),    # 绿色
            'receiver_name': (255, 0, 0),      # 蓝色
            'input_box': (0, 0, 255),          # 红色
            'send_button': (255, 255, 0),      # 青色
            'chat_message': (255, 0, 255),     # 紫色
            'contact_item': (0, 255, 255),     # 黄色
            'user_avatar': (128, 0, 128)       # 深紫色
        }
        
        for class_name, objects in detections.items():
            if class_name == 'metadata':
                continue
                
            color = colors.get(class_name, (128, 128, 128))
            
            for obj in objects:
                bbox = obj['bbox']
                confidence = obj['confidence']
                
                # 画边界框
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # 画标签
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def extract_im_info(self, image_path: str) -> Dict:
        """
        提取IM界面的关键信息
        
        Args:
            image_path: 图片路径
            
        Returns:
            提取的信息字典
        """
        detections = self.predict(image_path)
        
        info = {
            'receiver': None,
            'input_text': None,
            'contacts': [],
            'messages': [],
            'ui_elements': detections
        }
        
        # 提取接收者信息
        if 'receiver_name' in detections and detections['receiver_name']:
            # 选择置信度最高的
            best_receiver = detections['receiver_name'][0]
            info['receiver'] = {
                'name_bbox': best_receiver['bbox'],
                'confidence': best_receiver['confidence']
            }
            
            # 尝试关联头像
            if 'receiver_avatar' in detections:
                receiver_avatar = self._find_nearest_avatar(
                    best_receiver, detections['receiver_avatar']
                )
                if receiver_avatar:
                    info['receiver']['avatar_bbox'] = receiver_avatar['bbox']
        
        # 提取输入框信息
        if 'input_box' in detections and detections['input_box']:
            info['input_text'] = detections['input_box'][0]['bbox']
        
        # 提取联系人列表
        if 'contact_item' in detections:
            info['contacts'] = [
                {
                    'bbox': item['bbox'], 
                    'confidence': item['confidence']
                }
                for item in detections['contact_item'][:10]  # 最多10个联系人
            ]
        
        # 提取聊天消息
        if 'chat_message' in detections:
            info['messages'] = [
                {
                    'bbox': msg['bbox'],
                    'confidence': msg['confidence']
                }
                for msg in detections['chat_message']
            ]
        
        return info
    
    def _find_nearest_avatar(self, name_obj: Dict, avatars: List[Dict]) -> Optional[Dict]:
        """找到最近的头像"""
        if not avatars:
            return None
        
        name_center = name_obj['center']
        min_distance = float('inf')
        nearest_avatar = None
        
        for avatar in avatars:
            avatar_center = avatar['center']
            distance = np.sqrt(
                (name_center[0] - avatar_center[0]) ** 2 + 
                (name_center[1] - avatar_center[1]) ** 2
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_avatar = avatar
        
        # 只有距离合理才返回（不超过200像素）
        if min_distance < 200:
            return nearest_avatar
        
        return None
    
    def benchmark(self, test_images_folder: str, num_images: int = 10) -> Dict:
        """
        性能基准测试
        
        Args:
            test_images_folder: 测试图片文件夹
            num_images: 测试图片数量
            
        Returns:
            性能指标
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model()")
        
        # 获取测试图片
        test_images = list(Path(test_images_folder).glob("*.jpg"))[:num_images]
        
        if len(test_images) == 0:
            raise ValueError(f"在 {test_images_folder} 中未找到测试图片")
        
        print(f"🔥 开始性能测试，共 {len(test_images)} 张图片...")
        
        inference_times = []
        detection_counts = []
        
        for i, img_path in enumerate(test_images):
            start_time = time.time()
            detections = self.predict(str(img_path))
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
            
            # 统计检测到的目标数量
            total_detections = sum(
                len(objects) for class_name, objects in detections.items()
                if class_name != 'metadata'
            )
            detection_counts.append(total_detections)
            
            if (i + 1) % 5 == 0:
                print(f"  已处理: {i + 1}/{len(test_images)}")
        
        # 计算统计指标
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        min_inference_time = np.min(inference_times)
        max_inference_time = np.max(inference_times)
        
        metrics = {
            'num_images': len(test_images),
            'avg_inference_time_ms': avg_inference_time * 1000,
            'std_inference_time_ms': std_inference_time * 1000,
            'min_inference_time_ms': min_inference_time * 1000,
            'max_inference_time_ms': max_inference_time * 1000,
            'avg_fps': 1.0 / avg_inference_time,
            'min_fps': 1.0 / max_inference_time,
            'max_fps': 1.0 / min_inference_time,
            'avg_detections_per_image': np.mean(detection_counts),
            'total_detections': sum(detection_counts)
        }
        
        print("\n📊 性能测试结果:")
        print(f"  平均推理时间: {metrics['avg_inference_time_ms']:.2f} ms")
        print(f"  平均FPS: {metrics['avg_fps']:.2f}")
        print(f"  平均检测数量/图: {metrics['avg_detections_per_image']:.1f}")
        
        return metrics


def main():
    """测试检测功能"""
    
    # 初始化检测器
    model_path = "./models/best_im_detector.pt"
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行训练程序生成模型")
        return
    
    detector = IMDetector(model_path, confidence=0.5)
    
    # 测试单张图片检测
    test_image = "./data/test_image.jpg"
    if os.path.exists(test_image):
        print("🔍 测试单张图片检测...")
        result = detector.predict(test_image, return_image=True)
        
        print("检测结果:")
        for class_name, objects in result.items():
            if class_name != 'metadata' and class_name != 'annotated_image':
                print(f"  {class_name}: {len(objects)} 个")
        
        # 保存标注图片
        if 'annotated_image' in result and result['annotated_image'] is not None:
            cv2.imwrite("./results/annotated_result.jpg", result['annotated_image'])
            print("标注图片已保存: ./results/annotated_result.jpg")
        
        # 提取关键信息
        im_info = detector.extract_im_info(test_image)
        print("\n提取的IM信息:")
        print(f"  接收者: {'有' if im_info['receiver'] else '无'}")
        print(f"  输入框: {'有' if im_info['input_text'] else '无'}")
        print(f"  联系人数量: {len(im_info['contacts'])}")
        print(f"  消息数量: {len(im_info['messages'])}")
    
    # 性能测试
    test_folder = "./data/test_images"
    if os.path.exists(test_folder):
        print("\n🚀 性能测试...")
        metrics = detector.benchmark(test_folder, num_images=5)


if __name__ == "__main__":
    main()
