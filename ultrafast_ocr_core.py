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
                 providers: Optional[List[str]] = None,
                 enable_detection: bool = True,
                 fast_mode: bool = True):
        """
        初始化OCR引擎
        
        Args:
            det_model_path: 检测模型路径
            rec_model_path: 识别模型路径
            dict_path: 字符字典路径
            use_gpu: 是否使用GPU
            providers: ONNX Runtime providers
            enable_detection: 是否启用检测模型（用于多行文字）
            fast_mode: 快速模式（牺牲少量精度换取速度）
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
            
            # 打印实际使用的Provider
            actual_providers = self.rec_session.get_providers()
            if 'CUDAExecutionProvider' in actual_providers:
                print(f"✅ OCR使用GPU加速: {actual_providers[0]}")
            else:
                print(f"⚠️ OCR使用CPU: {actual_providers[0]}")
                
        except Exception as e:
            raise RuntimeError(f"加载识别模型失败: {e}")
        
        # 加载检测模型(用于多行文字识别)
        self.det_session = None
        self.enable_detection = enable_detection
        self.fast_mode = fast_mode
        
        if enable_detection:
            if det_model_path and os.path.exists(det_model_path):
                try:
                    # 为检测模型设置优化的session选项
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    sess_options.inter_op_num_threads = 4  # 限制线程数以提升效率
                    sess_options.intra_op_num_threads = 4
                    
                    self.det_session = ort.InferenceSession(det_model_path, sess_options, providers=providers)
                    self.det_input_name = self.det_session.get_inputs()[0].name
                    self.det_input_shape = self.det_session.get_inputs()[0].shape
                    
                    mode_desc = "快速模式" if self.fast_mode else "标准模式"
                    print(f"✅ 检测模型加载成功，支持多行文字识别 ({mode_desc})")
                except Exception as e:
                    print(f"⚠️ 检测模型加载失败: {e}")
                    print(f"   将退化为单行识别模式")
            else:
                print(f"⚠️ 检测模型路径无效: {det_model_path}")
                print(f"   多行文字识别功能不可用")
        
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
                          return_boxes: bool = False,
                          return_confidence: bool = False,
                          min_confidence: float = 0.5,
                          sort_output: bool = True) -> List:
        """
        识别多行文字（完整的检测+识别流程）
        
        Args:
            image: 输入图片
            return_boxes: 是否返回文字框坐标
            return_confidence: 是否返回置信度
            min_confidence: 最小置信度阈值
            sort_output: 是否按位置排序（从上到下，从左到右）
            
        Returns:
            根据参数返回不同格式：
            - 默认: ['文字1', '文字2', ...]
            - return_boxes=True: [('文字', 置信度, [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]), ...]
            - return_confidence=True: [('文字', 置信度), ...]
        """
        # 检查是否有检测模型
        if self.det_session is None:
            # 没有检测模型，退化为单行处理
            print("⚠️ 无检测模型，使用单行识别模式")
            if return_confidence:
                text, conf = self.recognize_single_line(image, return_confidence=True)
                if text and conf >= min_confidence:
                    if return_boxes:
                        h, w = image.shape[:2]
                        return [(text, conf, [[0, 0], [w, 0], [w, h], [0, h]])]
                    else:
                        return [(text, conf)]
            else:
                text = self.recognize_single_line(image)
                if text:
                    if return_boxes:
                        h, w = image.shape[:2]
                        return [(text, 0.95, [[0, 0], [w, 0], [w, h], [0, h]])]
                    else:
                        return [text]
            return []
        
        start_time = time.time()
        
        try:
            # ========== 步骤1: 文字检测 ==========
            print("🔍 执行文字检测...")
            det_start = time.time()
            
            # 预处理图片用于检测（使用快速模式）
            det_input, ratio = self.preprocessor.preprocess_for_detection(
                image, 
                max_side=640 if self.fast_mode else 960,
                fast_mode=self.fast_mode
            )
            
            # 运行检测模型
            det_outputs = self.det_session.run(None, {self.det_input_name: det_input})
            
            # 解码检测结果
            boxes = self.preprocessor.decode_detection(det_outputs[0], ratio)
            
            det_time = (time.time() - det_start) * 1000
            print(f"   检测到 {len(boxes)} 个文字区域 ({det_time:.1f}ms)")
            
            if not boxes:
                print("   未检测到文字区域")
                return []
            
            # ========== 步骤2: 排序检测框 ==========
            if sort_output and len(boxes) > 1:
                # 按y坐标(从上到下)，然后x坐标(从左到右)排序
                sorted_boxes = []
                for box in boxes:
                    # 计算中心点
                    center_x = np.mean(box[:, 0])
                    center_y = np.mean(box[:, 1])
                    sorted_boxes.append((center_y, center_x, box))
                
                # 排序：先按y，再按x
                sorted_boxes.sort(key=lambda x: (x[0], x[1]))
                boxes = [item[2] for item in sorted_boxes]
            
            # ========== 步骤3: 逐行识别 ==========
            print(f"📖 识别 {len(boxes)} 行文字...")
            rec_start = time.time()
            
            results = []
            for i, box in enumerate(boxes):
                # 裁剪文字区域
                roi = self.preprocessor.crop_image_by_box(image, box)
                
                if roi.size == 0:
                    continue
                
                # 识别单行文字
                text, confidence = self.recognize_single_line(roi, return_confidence=True)
                
                # 过滤低置信度结果
                if text and confidence >= min_confidence:
                    # 根据参数返回不同格式
                    if return_boxes:
                        results.append((text, confidence, box.tolist()))
                    elif return_confidence:
                        results.append((text, confidence))
                    else:
                        results.append(text)
                    
                    print(f"   行{i+1}: '{text[:30]}...' (置信度: {confidence:.3f})")
            
            rec_time = (time.time() - rec_start) * 1000
            total_time = (time.time() - start_time) * 1000
            
            print(f"   识别完成 ({rec_time:.1f}ms)")
            print(f"✅ 多行识别总耗时: {total_time:.1f}ms")
            
            return results
            
        except Exception as e:
            print(f"❌ 多行识别失败: {e}")
            import traceback
            traceback.print_exc()
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
