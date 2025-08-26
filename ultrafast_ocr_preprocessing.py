# -*- coding: utf-8 -*-
"""
图像预处理模块
"""

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from typing import Tuple, List


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self):
        """初始化预处理器"""
        pass
    
    def preprocess_for_recognition(self, 
                                 image: np.ndarray, 
                                 target_height: int = 48,
                                 max_width: int = 320) -> np.ndarray:
        """
        识别模型预处理
        
        Args:
            image: 输入图片(BGR格式)
            target_height: 目标高度
            max_width: 最大宽度
            
        Returns:
            预处理后的张量 [1, 3, H, W]
        """
        # 1. 处理输入格式
        if len(image.shape) == 2:  # 灰度图转RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA转RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        # 2. 计算缩放比例
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            # 处理空图片
            return np.zeros((1, 3, target_height, max_width), dtype=np.float32)
        
        # 保持宽高比缩放
        ratio = target_height / h
        new_w = int(w * ratio)
        
        # 限制最大宽度
        if new_w > max_width:
            ratio = max_width / w
            new_w = max_width
            new_h = int(h * ratio)
            # 如果缩放后高度不是目标高度，再次调整
            if new_h != target_height:
                new_h = target_height
        else:
            new_h = target_height
        
        # 确保最小尺寸
        new_w = max(16, new_w)
        new_h = max(16, new_h)
        
        # 3. 缩放图片
        try:
            if new_w != w or new_h != h:
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                resized = image.copy()
        except Exception as e:
            print(f"⚠️ 图片缩放失败: {e}")
            # 创建默认图片
            resized = np.ones((target_height, max_width, 3), dtype=np.uint8) * 255
        
        # 4. 填充到固定宽度(如果需要)
        if new_w < max_width:
            # 右侧填充白色
            padded = np.ones((new_h, max_width, 3), dtype=np.uint8) * 255
            padded[:, :new_w, :] = resized
            resized = padded
        
        # 5. 归一化到[-1, 1]
        normalized = resized.astype(np.float32)
        normalized = (normalized - 127.5) / 127.5
        
        # 6. 转换为CHW格式
        if len(normalized.shape) == 3:
            normalized = np.transpose(normalized, (2, 0, 1))
        
        # 7. 添加batch维度
        if len(normalized.shape) == 3:
            normalized = np.expand_dims(normalized, axis=0)
        
        return normalized
    
    def preprocess_for_detection(self, 
                               image: np.ndarray,
                               max_side: int = 960) -> Tuple[np.ndarray, float]:
        """
        检测模型预处理
        
        Args:
            image: 输入图片
            max_side: 最大边长
            
        Returns:
            (预处理后的张量, 缩放比例)
        """
        h, w = image.shape[:2]
        
        # 1. 计算缩放比例
        ratio = 1.0
        if max(h, w) > max_side:
            if h > w:
                ratio = max_side / h
            else:
                ratio = max_side / w
        
        # 2. 计算新尺寸(32的倍数)
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        
        # 确保是32的倍数
        new_h = 32 * ((new_h + 31) // 32)
        new_w = 32 * ((new_w + 31) // 32)
        
        # 3. 缩放
        resized = cv2.resize(image, (new_w, new_h))
        
        # 4. 归一化到[-1, 1]
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        
        # 5. 转换格式
        normalized = np.transpose(normalized, (2, 0, 1))
        normalized = np.expand_dims(normalized, axis=0)
        
        return normalized, ratio
    
    def decode_detection(self, 
                        pred: np.ndarray, 
                        ratio: float,
                        thresh: float = 0.3,
                        box_thresh: float = 0.6,
                        max_candidates: int = 1000) -> List[np.ndarray]:
        """
        解码检测结果
        
        Args:
            pred: 检测模型输出
            ratio: 缩放比例
            thresh: 分割阈值
            box_thresh: 框置信度阈值
            max_candidates: 最大候选框数
            
        Returns:
            文字框列表
        """
        try:
            # 获取概率图
            if len(pred.shape) == 4:
                pred = pred[0, 0, :, :]  # [1, 1, H, W] -> [H, W]
            elif len(pred.shape) == 3:
                pred = pred[0, :, :]     # [1, H, W] -> [H, W]
            
            # 二值化
            bitmap = pred > thresh
            
            # 查找轮廓
            bitmap_uint8 = (bitmap * 255).astype(np.uint8)
            contours, _ = cv2.findContours(bitmap_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            boxes = []
            for contour in contours[:max_candidates]:
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                if area < 10:
                    continue
                
                # 获取最小外接矩形
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                
                # 检查框的合理性
                box_h = max(rect[1])
                box_w = min(rect[1])
                if box_h < 5 or box_w < 5:
                    continue
                
                # 还原到原始尺寸
                box = box / ratio
                
                # 确保坐标为整数
                box = np.round(box).astype(np.int32)
                
                boxes.append(box)
            
            return boxes
            
        except Exception as e:
            print(f"⚠️ 检测解码失败: {e}")
            return []
    
    def normalize_box(self, box: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
        """
        规范化边界框
        
        Args:
            box: 边界框坐标
            img_shape: 图片尺寸 (height, width)
            
        Returns:
            规范化后的边界框
        """
        h, w = img_shape
        
        # 限制坐标范围
        box[:, 0] = np.clip(box[:, 0], 0, w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, h - 1)
        
        return box
    
    def crop_image_by_box(self, 
                         image: np.ndarray, 
                         box: np.ndarray,
                         padding: int = 2) -> np.ndarray:
        """
        根据边界框裁剪图片
        
        Args:
            image: 原图
            box: 边界框
            padding: 边距
            
        Returns:
            裁剪后的图片
        """
        h, w = image.shape[:2]
        
        # 获取边界框的外接矩形
        x_coords = box[:, 0]
        y_coords = box[:, 1]
        
        x_min = max(0, int(min(x_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_min = max(0, int(min(y_coords)) - padding)
        y_max = min(h, int(max(y_coords)) + padding)
        
        if x_max <= x_min or y_max <= y_min:
            # 无效区域，返回空白图片
            return np.ones((32, 32, 3), dtype=np.uint8) * 255
        
        return image[y_min:y_max, x_min:x_max]
    
    def enhance_image(self, image: np.ndarray, method: str = 'sharpen') -> np.ndarray:
        """
        图像增强
        
        Args:
            image: 输入图片
            method: 增强方法
            
        Returns:
            增强后的图片
        """
        if method == 'sharpen':
            # 锐化
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            return sharpened
        
        elif method == 'denoise':
            # 去噪
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            return denoised
        
        elif method == 'contrast':
            # 增强对比度
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            return enhanced
        
        else:
            return image
