# -*- coding: utf-8 -*-
"""
优化的OCR实现
包含缓存、并行等优化策略
"""

import cv2
import numpy as np
import hashlib
import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

from .core import UltraFastOCR


class OptimizedOCR:
    """
    优化的OCR引擎
    
    包含缓存、模板匹配、并行处理等优化
    """
    
    def __init__(self, 
                 cache_size: int = 1000,
                 use_template_matching: bool = True,
                 **kwargs):
        """
        初始化优化OCR
        
        Args:
            cache_size: 缓存大小
            use_template_matching: 是否使用模板匹配
            **kwargs: UltraFastOCR的参数
        """
        # 基础OCR引擎
        self.ocr = UltraFastOCR(**kwargs)
        
        # LRU缓存
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 模板匹配
        self.use_template_matching = use_template_matching
        self.templates = {}
        if use_template_matching:
            self._init_templates()
        
        # 性能统计
        self.total_calls = 0
        self.total_time = 0.0
    
    def _init_templates(self):
        """初始化常用文字模板"""
        common_texts = {
            '发送': (60, 40),
            'Send': (80, 40),
            '确定': (60, 40),
            '取消': (60, 40),
            'OK': (50, 40),
            'Cancel': (100, 40),
            '登录': (60, 40),
            '注册': (60, 40),
            '返回': (60, 40),
        }
        
        for text, (width, height) in common_texts.items():
            template = self._create_text_template(text, width, height)
            if template is not None:
                self.templates[text] = template
    
    def _create_text_template(self, text: str, width: int, height: int) -> Optional[np.ndarray]:
        """创建文字模板"""
        try:
            # 创建白色背景
            template = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # 计算文字大小和位置
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # 获取文字尺寸
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # 居中放置文字
            x = (width - text_width) // 2
            y = (height + text_height) // 2
            
            # 绘制文字
            cv2.putText(template, text, (x, y), font, font_scale, (0, 0, 0), thickness)
            
            return template
        except Exception as e:
            print(f"⚠️ 创建模板失败 {text}: {e}")
            return None
    
    def _get_image_hash(self, image: np.ndarray, size: int = 8) -> str:
        """计算图片感知哈希"""
        try:
            # 转灰度
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 缩放到固定大小
            resized = cv2.resize(gray, (size, size))
            
            # 计算平均值
            avg = resized.mean()
            
            # 生成哈希
            hash_str = ''
            for i in range(size):
                for j in range(size):
                    hash_str += '1' if resized[i, j] > avg else '0'
            
            return hash_str
        except Exception:
            # 备用哈希方法
            return hashlib.md5(image.tobytes()).hexdigest()[:16]
    
    def _template_match(self, image: np.ndarray, threshold: float = 0.8) -> Optional[str]:
        """模板匹配"""
        if not self.use_template_matching or not self.templates:
            return None
        
        try:
            # 调整图片大小以匹配模板
            h, w = image.shape[:2]
            if h < 20 or w < 20:
                return None
            
            best_match = None
            best_score = 0
            
            for text, template in self.templates.items():
                # 调整模板大小匹配输入图片
                t_h, t_w = template.shape[:2]
                if abs(h - t_h) > 10 or abs(w - t_w) > 20:
                    # 尺寸差异太大，跳过
                    continue
                
                # 缩放模板
                scaled_template = cv2.resize(template, (w, h))
                
                # 计算相似度
                if len(image.shape) == 3 and len(scaled_template.shape) == 3:
                    diff = cv2.absdiff(image, scaled_template)
                    score = 1.0 - (np.mean(diff) / 255.0)
                else:
                    # 处理灰度图
                    if len(image.shape) == 3:
                        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        image_gray = image
                    
                    if len(scaled_template.shape) == 3:
                        template_gray = cv2.cvtColor(scaled_template, cv2.COLOR_BGR2GRAY)
                    else:
                        template_gray = scaled_template
                    
                    diff = cv2.absdiff(image_gray, template_gray)
                    score = 1.0 - (np.mean(diff) / 255.0)
                
                if score > best_score and score > threshold:
                    best_score = score
                    best_match = text
            
            if best_match:
                return best_match
                
        except Exception as e:
            pass  # 模板匹配失败，继续用OCR
        
        return None
    
    def recognize_with_cache(self, image: np.ndarray) -> Tuple[str, bool, float]:
        """
        带缓存的识别
        
        Args:
            image: 输入图片
            
        Returns:
            (识别文字, 是否缓存命中, 耗时ms)
        """
        start_time = time.time()
        
        # 计算图片哈希
        img_hash = self._get_image_hash(image)
        
        # 检查缓存
        if img_hash in self.cache:
            self.cache_hits += 1
            # 移到末尾(LRU)
            result = self.cache[img_hash]
            self.cache.move_to_end(img_hash)
            elapsed = (time.time() - start_time) * 1000
            return result, True, elapsed
        
        # 缓存未命中
        self.cache_misses += 1
        
        # 1. 先尝试模板匹配
        template_result = self._template_match(image)
        if template_result:
            self.cache[img_hash] = template_result
            elapsed = (time.time() - start_time) * 1000
            return template_result, False, elapsed
        
        # 2. 使用OCR识别
        text = self.ocr.recognize_single_line(image)
        
        # 添加到缓存
        self.cache[img_hash] = text
        
        # 限制缓存大小
        if len(self.cache) > self.cache_size:
            # 删除最旧的项
            self.cache.popitem(last=False)
        
        # 统计
        elapsed = (time.time() - start_time) * 1000
        self.total_calls += 1
        self.total_time += elapsed
        
        return text, False, elapsed
    
    def recognize(self, image: np.ndarray) -> str:
        """简单识别接口"""
        text, _, _ = self.recognize_with_cache(image)
        return text
    
    def batch_recognize(self, 
                       images: List[np.ndarray],
                       use_parallel: bool = True,
                       max_workers: int = 4) -> List[str]:
        """
        批量识别
        
        Args:
            images: 图片列表
            use_parallel: 是否并行处理
            max_workers: 最大工作线程数
            
        Returns:
            识别结果列表
        """
        if not use_parallel or len(images) < 4:
            # 串行处理
            return [self.recognize(img) for img in images]
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.recognize, images))
        
        return results
    
    def recognize_im_element(self, 
                           image: np.ndarray,
                           element_type: str,
                           bbox: List[int]) -> str:
        """
        识别IM界面元素
        
        Args:
            image: 完整图片
            element_type: 元素类型
            bbox: 边界框 [x1, y1, x2, y2]
            
        Returns:
            识别的文字
        """
        x1, y1, x2, y2 = bbox
        
        # 边界检查
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return ""
        
        # 裁剪区域
        roi = image[y1:y2, x1:x2]
        
        # 根据元素类型优化处理
        if element_type == 'send_button':
            # 发送按钮优先模板匹配
            template_result = self._template_match(roi)
            if template_result:
                return template_result
        
        elif element_type == 'input_box':
            # 输入框可能为空
            if self._is_empty_region(roi):
                return ""
        
        elif element_type in ['receiver_name', 'contact_item']:
            # 名称类元素，可能有特殊字符
            pass  # 使用标准OCR
        
        # 使用OCR识别
        return self.recognize(roi)
    
    def _is_empty_region(self, roi: np.ndarray, threshold: int = 240) -> bool:
        """判断区域是否为空(主要是白色)"""
        try:
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            mean_val = np.mean(gray)
            return mean_val > threshold
        except:
            return False
    
    def get_statistics(self) -> Dict:
        """获取性能统计"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        avg_time = self.total_time / max(1, self.total_calls)
        
        return {
            'total_calls': self.total_calls,
            'total_time_ms': self.total_time,
            'avg_time_ms': avg_time,
            'estimated_fps': 1000 / avg_time if avg_time > 0 else 0,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'templates_loaded': len(self.templates)
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def add_template(self, text: str, template_image: np.ndarray):
        """添加自定义模板"""
        self.templates[text] = template_image.copy()
    
    def benchmark(self, test_images: List[np.ndarray], rounds: int = 100) -> Dict:
        """性能基准测试"""
        if not test_images:
            return {}
        
        # 清空缓存确保公平测试
        original_cache = self.cache.copy()
        self.clear_cache()
        
        try:
            # 预热
            for _ in range(3):
                _ = self.recognize(test_images[0])
            
            # 第一轮：测试无缓存性能
            times_no_cache = []
            for _ in range(rounds):
                for img in test_images:
                    self.clear_cache()
                    start = time.time()
                    _ = self.recognize(img)
                    times_no_cache.append((time.time() - start) * 1000)
            
            # 第二轮：测试有缓存性能
            self.clear_cache()
            times_with_cache = []
            for _ in range(rounds):
                for img in test_images:
                    start = time.time()
                    _ = self.recognize(img)
                    times_with_cache.append((time.time() - start) * 1000)
            
            stats = self.get_statistics()
            
            return {
                'test_images': len(test_images),
                'rounds': rounds,
                'no_cache': {
                    'avg_time_ms': float(np.mean(times_no_cache)),
                    'min_time_ms': float(np.min(times_no_cache)),
                    'max_time_ms': float(np.max(times_no_cache)),
                    'fps': float(1000 / np.mean(times_no_cache))
                },
                'with_cache': {
                    'avg_time_ms': float(np.mean(times_with_cache)),
                    'min_time_ms': float(np.min(times_with_cache)),
                    'max_time_ms': float(np.max(times_with_cache)),
                    'fps': float(1000 / np.mean(times_with_cache)),
                    'hit_rate': stats['cache_hit_rate']
                },
                'speedup': float(np.mean(times_no_cache) / np.mean(times_with_cache))
            }
            
        finally:
            # 恢复原始缓存
            self.cache = original_cache
    
    def __repr__(self):
        stats = self.get_statistics()
        return (f"OptimizedOCR(calls={stats['total_calls']}, "
                f"avg_time={stats['avg_time_ms']:.2f}ms, "
                f"hit_rate={stats['cache_hit_rate']*100:.1f}%)")


class BatchOCR:
    """批量并行OCR处理器"""
    
    def __init__(self, num_workers: int = 4, **kwargs):
        """
        初始化批量处理器
        
        Args:
            num_workers: 工作进程数
            **kwargs: OCR参数
        """
        self.num_workers = num_workers
        self.ocr_pool = [OptimizedOCR(**kwargs) for _ in range(num_workers)]
    
    def process_batch(self, 
                     images: List[np.ndarray],
                     batch_size: int = 32) -> List[str]:
        """
        批量并行处理
        
        Args:
            images: 图片列表
            batch_size: 批大小
            
        Returns:
            识别结果列表
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # 分配OCR实例
                futures = []
                for j, img in enumerate(batch):
                    ocr = self.ocr_pool[j % self.num_workers]
                    future = executor.submit(ocr.recognize, img)
                    futures.append(future)
                
                # 收集结果
                batch_results = [future.result() for future in futures]
                results.extend(batch_results)
        
        return results
    
    def get_statistics(self) -> List[Dict]:
        """获取所有OCR实例的统计"""
        return [ocr.get_statistics() for ocr in self.ocr_pool]
