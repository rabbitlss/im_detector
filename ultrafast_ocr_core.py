# -*- coding: utf-8 -*-
"""
UltraFast OCR 核心实现
基于ONNX Runtime的超快速OCR引擎
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import time
from typing import List, Tuple, Optional
from pathlib import Path

from .preprocessing import ImagePreprocessor
from .postprocessing import TextDecoder
from .utils import validate_image, get_default_model_path


class UltraFastOCR:
    """
    超快速OCR引擎
    
    基于ONNX Runtime实现，无需深度学习框架
    识别速度：3-10ms
    """
    
    def __init__(self, 
                 det_model_path: Optional[str] = None,
                 rec_model_path: Optional[str] = None,
                 dict_path: Optional[str] = None,
                 use_gpu: bool = True,
                 providers: Optional[List[str]] = None):
        """
        初始化OCR引擎
        
        Args:
            det_model_path: 检测模型路径(可选)
            rec_model_path: 识别模型路径
            dict_path: 字符字典路径
            use_gpu: 是否使用GPU
            providers: ONNX Runtime providers
        """
        
        # 设置providers
        if providers is None:
            if use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        
        # 获取默认模型路径
        if rec_model_path is None:
            rec_model_path = get_default_model_path('rec')
        if dict_path is None:
            dict_path = get_default_model_path('dict')
        if det_model_path is None:
            det_model_path = get_default_model_path('det')
        
        # 验证模型文件
        if not os.path.exists(rec_model_path):
            raise FileNotFoundError(f"识别模型不存在: {rec_model_path}")
        
        # 初始化组件
        self.preprocessor = ImagePreprocessor()
        self.decoder = TextDecoder(dict_path)
        
        # 加载识别模型(必需)
        try:
            self.rec_session = ort.InferenceSession(rec_model_path, providers=providers)
            self.rec_input_name = self.rec_session.get_inputs()[0].name
            self.rec_input_shape = self.rec_session.get_inputs()[0].shape
        except Exception as e:
            raise RuntimeError(f"加载识别模型失败: {e}")
        
        # 加载检测模型(可选)
        self.det_session = None
        if det_model_path and os.path.exists(det_model_path):
            try:
                self.det_session = ort.InferenceSession(det_model_path, providers=providers)
                self.det_input_name = self.det_session.get_inputs()[0].name
            except Exception as e:
                print(f"⚠️ 检测模型加载失败: {e}")
        
        # 预热模型
        self._warmup()
        
        # 统计信息
        self.total_calls = 0
        self.total_time = 0.0
    
    def _warmup(self):
        """模型预热"""
        try:
            # 识别模型预热
            if hasattr(self, 'rec_session'):
                dummy_input = np.zeros((1, 3, 48, 320), dtype=np.float32)
                _ = self.rec_session.run(None, {self.rec_input_name: dummy_input})
            
            # 检测模型预热
            if self.det_session:
                dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
                _ = self.det_session.run(None, {self.det_input_name: dummy_input})
                
        except Exception as e:
            print(f"⚠️ 模型预热失败: {e}")
    
    def recognize_single_line(self, 
                            image: np.ndarray, 
                            return_confidence: bool = False,
                            return_time: bool = False) -> Tuple:
        """
        识别单行文字
        
        Args:
            image: 输入图片(BGR格式)
            return_confidence: 是否返回置信度
            return_time: 是否返回耗时
            
        Returns:
            根据参数返回 (text,) 或 (text, conf) 或 (text, conf, time_ms)
        """
        start_time = time.time()
        
        # 验证输入
        if not validate_image(image):
            if return_time and return_confidence:
                return "", 0.0, 0.0
            elif return_confidence:
                return "", 0.0
            else:
                return ""
        
        try:
            # 预处理
            input_tensor = self.preprocessor.preprocess_for_recognition(image)
            
            # ONNX推理
            outputs = self.rec_session.run(None, {self.rec_input_name: input_tensor})
            
            # 后处理
            text, confidence = self.decoder.decode_recognition(outputs[0])
            
            # 统计
            elapsed_time = (time.time() - start_time) * 1000
            self.total_calls += 1
            self.total_time += elapsed_time
            
            # 返回结果
            if return_time and return_confidence:
                return text, confidence, elapsed_time
            elif return_confidence:
                return text, confidence
            elif return_time:
                return text, elapsed_time
            else:
                return text
                
        except Exception as e:
            print(f"❌ OCR识别失败: {e}")
            if return_time and return_confidence:
                return "", 0.0, 0.0
            elif return_confidence:
                return "", 0.0
            else:
                return ""
    
    def recognize_multiline(self, 
                          image: np.ndarray,
                          return_boxes: bool = False) -> List:
        """
        识别多行文字
        
        Args:
            image: 输入图片
            return_boxes: 是否返回文字框坐标
            
        Returns:
            文字列表或[(文字, 置信度, 坐标), ...]
        """
        if self.det_session is None:
            # 没有检测模型，当作单行处理
            text = self.recognize_single_line(image)
            if return_boxes:
                h, w = image.shape[:2]
                return [(text, 0.8, [[0, 0], [w, 0], [w, h], [0, h]])]
            else:
                return [text] if text else []
        
        try:
            # 文字检测
            det_input, ratio = self.preprocessor.preprocess_for_detection(image)
            det_outputs = self.det_session.run(None, {self.det_input_name: det_input})
            boxes = self.preprocessor.decode_detection(det_outputs[0], ratio)
            
            # 逐个识别
            results = []
            for box in boxes:
                # 获取ROI
                x_coords = box[:, 0]
                y_coords = box[:, 1]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # 边界检查
                h, w = image.shape[:2]
                x_min = max(0, min(x_min, w))
                x_max = max(0, min(x_max, w))
                y_min = max(0, min(y_min, h))
                y_max = max(0, min(y_max, h))
                
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                roi = image[y_min:y_max, x_min:x_max]
                
                # 识别文字
                text, confidence = self.recognize_single_line(roi, return_confidence=True)
                
                if text:
                    if return_boxes:
                        results.append((text, confidence, box.tolist()))
                    else:
                        results.append(text)
            
            return results
            
        except Exception as e:
            print(f"❌ 多行识别失败: {e}")
            return []
    
    def recognize(self, 
                 image: np.ndarray, 
                 single_line: bool = True,
                 **kwargs) -> str:
        """
        通用识别接口
        
        Args:
            image: 输入图片
            single_line: 是否单行模式
            **kwargs: 其他参数
            
        Returns:
            识别的文字
        """
        if single_line:
            return self.recognize_single_line(image, **kwargs)
        else:
            results = self.recognize_multiline(image, return_boxes=False)
            return ' '.join(results)
    
    def batch_recognize(self, 
                       images: List[np.ndarray],
                       single_line: bool = True) -> List[str]:
        """
        批量识别
        
        Args:
            images: 图片列表
            single_line: 是否单行模式
            
        Returns:
            识别结果列表
        """
        results = []
        for image in images:
            if single_line:
                text = self.recognize_single_line(image)
            else:
                texts = self.recognize_multiline(image, return_boxes=False)
                text = ' '.join(texts)
            results.append(text)
        return results
    
    def benchmark(self, 
                 test_images: List[np.ndarray],
                 rounds: int = 100) -> dict:
        """
        性能基准测试
        
        Args:
            test_images: 测试图片列表
            rounds: 测试轮数
            
        Returns:
            性能统计
        """
        if not test_images:
            return {}
        
        # 预热
        for _ in range(3):
            _ = self.recognize_single_line(test_images[0])
        
        # 测试
        times = []
        for _ in range(rounds):
            for img in test_images:
                start = time.time()
                _ = self.recognize_single_line(img)
                times.append((time.time() - start) * 1000)
        
        times = np.array(times)
        
        return {
            'num_images': len(test_images),
            'rounds': rounds,
            'total_calls': len(times),
            'avg_time_ms': float(np.mean(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'std_time_ms': float(np.std(times)),
            'fps': float(1000 / np.mean(times)),
            'percentile_95_ms': float(np.percentile(times, 95)),
            'percentile_99_ms': float(np.percentile(times, 99))
        }
    
    def get_statistics(self) -> dict:
        """获取使用统计"""
        if self.total_calls == 0:
            return {
                'total_calls': 0,
                'avg_time_ms': 0,
                'total_time_ms': 0
            }
        
        return {
            'total_calls': self.total_calls,
            'avg_time_ms': self.total_time / self.total_calls,
            'total_time_ms': self.total_time,
            'estimated_fps': 1000 / (self.total_time / self.total_calls) if self.total_calls > 0 else 0
        }
    
    def __repr__(self):
        """字符串表示"""
        return f"UltraFastOCR(calls={self.total_calls}, avg_time={self.total_time/max(1,self.total_calls):.2f}ms)"
    
    def __del__(self):
        """析构函数"""
        # 清理ONNX Session
        if hasattr(self, 'rec_session'):
            del self.rec_session
        if hasattr(self, 'det_session') and self.det_session:
            del self.det_session
