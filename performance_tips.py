# -*- coding: utf-8 -*-
"""
OCR性能优化技巧
包含缓存、批处理、并行等优化方法
"""

import cv2
import numpy as np
import hashlib
import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
from collections import OrderedDict
from ultra_fast_ocr import UltraFastOCR


class OptimizedOCR:
    """优化后的OCR（稳定3-5ms）"""
    
    def __init__(self, cache_size: int = 1000):
        """
        初始化优化OCR
        
        Args:
            cache_size: 缓存大小
        """
        self.ocr = UltraFastOCR(use_gpu=True)
        
        # LRU缓存
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 预编译的模板
        self.templates = {
            '发送': self._create_template('发送'),
            'Send': self._create_template('Send'),
            '确定': self._create_template('确定'),
            '取消': self._create_template('取消'),
        }
    
    def _create_template(self, text: str) -> np.ndarray:
        """创建文字模板用于快速匹配"""
        img = np.ones((48, len(text) * 30, 3), dtype=np.uint8) * 255
        cv2.putText(img, text, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return img
    
    def get_image_hash(self, image: np.ndarray) -> str:
        """计算图片哈希（用于缓存）"""
        # 缩小图片以加速哈希计算
        small = cv2.resize(image, (32, 32))
        return hashlib.md5(small.tobytes()).hexdigest()
    
    def recognize_with_cache(self, image_region: np.ndarray) -> Tuple[str, bool]:
        """
        带缓存的识别（重复内容0ms）
        
        Args:
            image_region: 图片区域
            
        Returns:
            (识别的文字, 是否缓存命中)
        """
        # 计算哈希
        img_hash = self.get_image_hash(image_region)
        
        # 检查缓存
        if img_hash in self.cache:
            self.cache_hits += 1
            # 移到最后（LRU）
            self.cache.move_to_end(img_hash)
            return self.cache[img_hash], True
        
        # 缓存未命中，进行OCR
        self.cache_misses += 1
        text, _, _ = self.ocr.recognize_single_line(image_region)
        
        # 添加到缓存
        self.cache[img_hash] = text
        
        # 限制缓存大小
        if len(self.cache) > self.cache_size:
            # 删除最旧的项
            self.cache.popitem(last=False)
        
        return text, False
    
    def batch_recognize(self, image_regions: List[np.ndarray], 
                       use_parallel: bool = True) -> List[str]:
        """
        批量识别（并行处理）
        
        Args:
            image_regions: 图片区域列表
            use_parallel: 是否使用并行
            
        Returns:
            识别的文字列表
        """
        if not use_parallel or len(image_regions) < 4:
            # 串行处理
            results = []
            for region in image_regions:
                text, _ = self.recognize_with_cache(region)
                results.append(text)
            return results
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.recognize_with_cache, region) 
                      for region in image_regions]
            results = [future.result()[0] for future in futures]
        
        return results
    
    def recognize_im_element(self, image: np.ndarray, 
                           element_type: str, 
                           bbox: List[int]) -> str:
        """
        针对IM元素优化的识别
        
        Args:
            image: 完整图片
            element_type: 元素类型
            bbox: 边界框 [x1, y1, x2, y2]
            
        Returns:
            识别的文字
        """
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        
        # 根据元素类型优化
        if element_type == 'send_button':
            # 发送按钮通常是固定文字，先尝试模板匹配
            for template_text, template_img in self.templates.items():
                if self._quick_match(roi, template_img):
                    return template_text
            # 模板匹配失败，使用OCR
            return self.recognize_with_cache(roi)[0]
            
        elif element_type == 'receiver_name':
            # 昵称通常是单行，可能有表情符号
            text, _ = self.recognize_with_cache(roi)
            return text
            
        elif element_type == 'chat_message':
            # 消息可能多行，使用多行识别
            return self.ocr.recognize(roi, single_line=False)
            
        elif element_type == 'input_box':
            # 输入框可能是空的或有提示文字
            if self._is_empty_input(roi):
                return ""
            return self.recognize_with_cache(roi)[0]
        
        else:
            return self.recognize_with_cache(roi)[0]
    
    def _quick_match(self, roi: np.ndarray, template: np.ndarray) -> bool:
        """快速模板匹配（1ms）"""
        try:
            # 调整大小
            roi_resized = cv2.resize(roi, (template.shape[1], template.shape[0]))
            # 计算相似度
            diff = cv2.absdiff(roi_resized, template)
            score = np.mean(diff)
            return score < 30  # 阈值
        except:
            return False
    
    def _is_empty_input(self, roi: np.ndarray) -> bool:
        """判断输入框是否为空"""
        # 检查是否主要是白色/浅色
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        return mean_val > 240  # 主要是白色
    
    def get_statistics(self) -> Dict:
        """获取性能统计"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'total_requests': total_requests
        }


class ParallelOCR:
    """并行OCR处理器"""
    
    def __init__(self, num_workers: int = 4):
        """
        初始化并行处理器
        
        Args:
            num_workers: 工作线程数
        """
        self.num_workers = num_workers
        self.ocr_pool = [UltraFastOCR(use_gpu=True) for _ in range(num_workers)]
    
    def process_batch(self, images: List[np.ndarray], 
                     batch_size: int = 10) -> List[str]:
        """
        批量并行处理
        
        Args:
            images: 图片列表
            batch_size: 每批大小
            
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
                    future = executor.submit(ocr.recognize_single_line, img)
                    futures.append(future)
                
                # 收集结果
                for future in futures:
                    text, _, _ = future.result()
                    results.append(text)
        
        return results


class StreamingOCR:
    """流式OCR处理器（用于视频/实时流）"""
    
    def __init__(self, buffer_size: int = 30):
        """
        初始化流式处理器
        
        Args:
            buffer_size: 缓冲区大小
        """
        self.ocr = OptimizedOCR()
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.result_buffer = []
        
    def process_frame(self, frame: np.ndarray, 
                     regions: List[Dict]) -> List[Dict]:
        """
        处理单帧
        
        Args:
            frame: 视频帧
            regions: 需要OCR的区域列表
                [{'bbox': [x1,y1,x2,y2], 'type': 'chat_message'}, ...]
                
        Returns:
            OCR结果列表
        """
        results = []
        
        for region in regions:
            bbox = region['bbox']
            element_type = region.get('type', 'unknown')
            
            # 提取文字
            text = self.ocr.recognize_im_element(frame, element_type, bbox)
            
            results.append({
                'bbox': bbox,
                'type': element_type,
                'text': text,
                'timestamp': time.time()
            })
        
        # 添加到缓冲区
        self.result_buffer.append(results)
        if len(self.result_buffer) > self.buffer_size:
            self.result_buffer.pop(0)
        
        return results
    
    def get_stable_text(self, bbox: List[int], 
                       min_occurrences: int = 3) -> Optional[str]:
        """
        获取稳定的文字（多帧中出现的）
        
        Args:
            bbox: 边界框
            min_occurrences: 最小出现次数
            
        Returns:
            稳定的文字或None
        """
        texts = []
        
        for frame_results in self.result_buffer[-10:]:  # 检查最近10帧
            for result in frame_results:
                if self._bbox_overlap(result['bbox'], bbox) > 0.8:
                    texts.append(result['text'])
        
        # 统计出现次数
        from collections import Counter
        text_counts = Counter(texts)
        
        for text, count in text_counts.most_common(1):
            if count >= min_occurrences:
                return text
        
        return None
    
    def _bbox_overlap(self, bbox1: List[int], bbox2: List[int]) -> float:
        """计算边界框重叠度（IoU）"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def benchmark_optimizations():
    """测试各种优化效果"""
    
    print("🔬 OCR优化效果测试")
    print("=" * 50)
    
    # 创建测试数据
    test_images = []
    for i in range(20):
        img = np.ones((48, 200, 3), dtype=np.uint8) * 255
        text = f"Test {i % 5}"  # 有重复的文字
        cv2.putText(img, text, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        test_images.append(img)
    
    # 1. 基础OCR（无优化）
    print("\n1. 基础OCR（无优化）:")
    basic_ocr = UltraFastOCR()
    start = time.time()
    for img in test_images:
        _ = basic_ocr.recognize_single_line(img)
    basic_time = time.time() - start
    print(f"   总耗时: {basic_time*1000:.1f}ms")
    print(f"   平均: {basic_time*1000/len(test_images):.1f}ms/image")
    
    # 2. 带缓存的OCR
    print("\n2. 带缓存的OCR:")
    cached_ocr = OptimizedOCR()
    start = time.time()
    for img in test_images:
        _ = cached_ocr.recognize_with_cache(img)
    cached_time = time.time() - start
    stats = cached_ocr.get_statistics()
    print(f"   总耗时: {cached_time*1000:.1f}ms")
    print(f"   平均: {cached_time*1000/len(test_images):.1f}ms/image")
    print(f"   缓存命中率: {stats['hit_rate']*100:.1f}%")
    print(f"   加速比: {basic_time/cached_time:.1f}x")
    
    # 3. 批量并行处理
    print("\n3. 批量并行处理:")
    cached_ocr_batch = OptimizedOCR()
    start = time.time()
    _ = cached_ocr_batch.batch_recognize(test_images, use_parallel=True)
    parallel_time = time.time() - start
    print(f"   总耗时: {parallel_time*1000:.1f}ms")
    print(f"   平均: {parallel_time*1000/len(test_images):.1f}ms/image")
    print(f"   加速比: {basic_time/parallel_time:.1f}x")
    
    # 4. 多实例并行
    print("\n4. 多实例并行:")
    parallel_ocr = ParallelOCR(num_workers=4)
    start = time.time()
    _ = parallel_ocr.process_batch(test_images)
    multi_time = time.time() - start
    print(f"   总耗时: {multi_time*1000:.1f}ms")
    print(f"   平均: {multi_time*1000/len(test_images):.1f}ms/image")
    print(f"   加速比: {basic_time/multi_time:.1f}x")
    
    print("\n" + "=" * 50)
    print("📊 优化效果总结:")
    print(f"   基础方案: {basic_time*1000:.1f}ms")
    print(f"   缓存优化: {cached_time*1000:.1f}ms ({basic_time/cached_time:.1f}x)")
    print(f"   并行优化: {parallel_time*1000:.1f}ms ({basic_time/parallel_time:.1f}x)")
    print(f"   多实例: {multi_time*1000:.1f}ms ({basic_time/multi_time:.1f}x)")


def demo_im_optimization():
    """演示IM场景的优化"""
    
    print("\n💬 IM场景OCR优化演示")
    print("=" * 50)
    
    # 初始化优化OCR
    ocr = OptimizedOCR()
    
    # 模拟IM界面元素
    elements = [
        {'type': 'receiver_name', 'text': '张三', 'bbox': [150, 20, 250, 50]},
        {'type': 'chat_message', 'text': '你好', 'bbox': [50, 100, 150, 140]},
        {'type': 'chat_message', 'text': '在吗？', 'bbox': [50, 150, 150, 190]},
        {'type': 'send_button', 'text': '发送', 'bbox': [400, 500, 450, 530]},
        {'type': 'input_box', 'text': '', 'bbox': [50, 500, 390, 530]},
    ]
    
    # 创建模拟图片
    img = np.ones((600, 500, 3), dtype=np.uint8) * 240
    
    for element in elements:
        x1, y1, x2, y2 = element['bbox']
        # 绘制元素
        if element['text']:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
            cv2.putText(img, element['text'], (x1+10, y1+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # 识别测试
    print("\n识别结果:")
    total_time = 0
    
    for element in elements:
        start = time.time()
        text = ocr.recognize_im_element(img, element['type'], element['bbox'])
        elapsed = (time.time() - start) * 1000
        total_time += elapsed
        
        print(f"  {element['type']:15s}: '{text}' ({elapsed:.1f}ms)")
    
    print(f"\n总耗时: {total_time:.1f}ms")
    print(f"平均: {total_time/len(elements):.1f}ms/element")
    
    # 显示缓存统计
    stats = ocr.get_statistics()
    print(f"\n缓存统计:")
    print(f"  命中: {stats['cache_hits']}")
    print(f"  未命中: {stats['cache_misses']}")
    print(f"  命中率: {stats['hit_rate']*100:.1f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("OCR性能优化演示")
    print("=" * 60)
    
    # 1. 测试各种优化
    benchmark_optimizations()
    
    # 2. IM场景优化演示
    demo_im_optimization()
    
    print("\n" + "=" * 60)
    print("💡 优化建议:")
    print("1. 使用缓存：相同内容0ms")
    print("2. 批量处理：减少开销")
    print("3. 并行处理：多核加速")
    print("4. 模板匹配：固定文字1ms")
    print("5. 预处理优化：二值化、去噪等")
    print("=" * 60)
