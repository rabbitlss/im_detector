# -*- coding: utf-8 -*-
"""
优化版智能多行OCR系统
通过智能缓存和批量处理策略大幅减少OCR调用次数
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import hashlib


@dataclass
class TextRegion:
    """文字区域信息"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: int
    height: int
    width: int
    aspect_ratio: float
    centroid: Tuple[float, float]


@dataclass  
class LineInfo:
    """文本行信息"""
    image: np.ndarray
    bbox: Tuple[int, int, int, int]
    estimated_chars: int
    char_height: int
    char_width_avg: float
    confidence: float
    image_hash: str = None  # 图像哈希，用于缓存


class IntelligentMultilineOCROptimized:
    """优化版智能多行OCR系统 - 最大化减少OCR调用"""
    
    def __init__(self, 
                 ocr_engine,
                 max_concat_width: int = 2560,  # 增大最大宽度
                 target_height: int = 48,
                 enable_cache: bool = True):
        """
        初始化智能多行OCR
        
        Args:
            ocr_engine: 单行OCR引擎
            max_concat_width: 最大拼接宽度（增大到2560）
            target_height: OCR模型目标高度
            enable_cache: 是否启用缓存
        """
        self.ocr = ocr_engine
        self.max_concat_width = max_concat_width
        self.target_height = target_height
        self.enable_cache = enable_cache
        
        # OCR结果缓存
        self.ocr_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 批量处理缓存
        self.batch_cache = {}
    
    def _compute_image_hash(self, image: np.ndarray) -> str:
        """计算图像的哈希值用于缓存"""
        # 使用快速哈希算法
        return hashlib.md5(image.tobytes()).hexdigest()
    
    def _ocr_with_cache(self, image: np.ndarray) -> str:
        """带缓存的OCR识别"""
        if not self.enable_cache:
            return self.ocr.recognize_single_line(image)
        
        # 计算图像哈希
        img_hash = self._compute_image_hash(image)
        
        # 检查缓存
        if img_hash in self.ocr_cache:
            self.cache_hits += 1
            return self.ocr_cache[img_hash]
        
        # 缓存未命中，执行OCR
        self.cache_misses += 1
        result = self.ocr.recognize_single_line(image)
        self.ocr_cache[img_hash] = result
        
        return result
    
    def analyze_text_structure(self, image: np.ndarray) -> Dict:
        """
        步骤1: 分析图像中的文字结构
        """
        # 1.1 预处理
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1.2 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 1.3 形态学操作
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        connected_text = cv2.dilate(binary, kernel_horizontal, iterations=1)
        
        # 1.4 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # 1.5 提取有效文字区域
        text_regions = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            if (area > 20 and h > 5 and w > 3 and 0.1 < h/w < 10):
                region = TextRegion(
                    bbox=(x, y, w, h),
                    area=area,
                    height=h,
                    width=w,
                    aspect_ratio=h/w,
                    centroid=centroids[i]
                )
                text_regions.append(region)
        
        # 1.6 统计分析
        if not text_regions:
            return {
                'char_height': 20,
                'line_height': 30,
                'line_spacing': 10,
                'text_regions': [],
                'num_chars': 0
            }
        
        heights = [r.height for r in text_regions]
        char_height = int(np.median(heights))
        char_height_std = int(np.std(heights))
        
        line_height = int(char_height * 1.2)
        line_spacing = max(5, int(char_height * 0.3))
        
        return {
            'char_height': char_height,
            'char_height_std': char_height_std,
            'line_height': line_height,
            'line_spacing': line_spacing,
            'text_regions': text_regions,
            'num_chars': len(text_regions),
            'height_distribution': np.histogram(heights, bins=10)[0].tolist()
        }
    
    def detect_text_lines(self, image: np.ndarray, structure_info: Dict) -> List[LineInfo]:
        """
        步骤2: 检测和切割文本行
        """
        char_height = structure_info['char_height']
        line_spacing = structure_info['line_spacing']
        
        # 2.1 水平投影分析
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h_projection = np.sum(binary == 255, axis=1)
        
        # 2.2 动态阈值检测行边界
        min_line_height = max(8, int(char_height * 0.6))
        merge_threshold = max(3, int(line_spacing * 0.8))
        
        # 2.3 检测行边界
        line_ranges = []
        in_line = False
        start_y = 0
        
        proj_threshold = max(1, int(np.mean(h_projection[h_projection > 0]) * 0.1))
        
        for y in range(len(h_projection)):
            if h_projection[y] > proj_threshold and not in_line:
                start_y = y
                in_line = True
            elif h_projection[y] <= proj_threshold and in_line:
                if y - start_y >= min_line_height:
                    line_ranges.append((start_y, y))
                in_line = False
        
        if in_line and len(h_projection) - start_y >= min_line_height:
            line_ranges.append((start_y, len(h_projection)))
        
        # 2.4 合并过于接近的行
        merged_ranges = self._merge_close_lines(line_ranges, merge_threshold)
        
        # 2.5 创建LineInfo对象
        lines = []
        for i, (start_y, end_y) in enumerate(merged_ranges):
            padding = max(2, char_height // 8)
            y1 = max(0, start_y - padding)
            y2 = min(image.shape[0], end_y + padding)
            
            line_img = image[y1:y2, :]
            if line_img.size == 0:
                continue
            
            line_height = y2 - y1
            line_width = image.shape[1]
            
            avg_char_width = char_height * 0.8
            estimated_chars = max(1, int(line_width / avg_char_width))
            
            line_projection_sum = np.sum(h_projection[start_y:end_y])
            max_possible_sum = (end_y - start_y) * line_width
            confidence = min(1.0, line_projection_sum / max_possible_sum) if max_possible_sum > 0 else 0.5
            
            # 计算图像哈希
            img_hash = self._compute_image_hash(line_img) if self.enable_cache else None
            
            line_info = LineInfo(
                image=line_img,
                bbox=(0, y1, line_width, line_height),
                estimated_chars=estimated_chars,
                char_height=char_height,
                char_width_avg=avg_char_width,
                confidence=confidence,
                image_hash=img_hash
            )
            lines.append(line_info)
        
        return lines
    
    def optimize_concatenation_aggressive(self, lines: List[LineInfo]) -> List[Tuple[np.ndarray, List[int]]]:
        """
        步骤3: 激进的拼接优化策略 - 最大化减少OCR调用
        
        策略：
        1. 移除行数限制
        2. 放宽相似性约束
        3. 充分利用最大宽度
        4. 使用动态分组算法
        """
        if not lines:
            return []
        
        # 预估每行缩放后的宽度
        estimated_widths = []
        for line in lines:
            h, w = line.image.shape[:2]
            scale_ratio = self.target_height / h if h > 0 else 1.0
            scaled_width = int(w * scale_ratio)
            estimated_widths.append(scaled_width)
        
        # 使用动态规划找到最优分组
        groups = self._dynamic_grouping(lines, estimated_widths)
        
        return groups
    
    def _dynamic_grouping(self, lines: List[LineInfo], widths: List[int]) -> List[Tuple[np.ndarray, List[int]]]:
        """
        动态规划算法：找到最少的OCR调用次数
        """
        n = len(lines)
        gap_width = 20
        
        # dp[i] = (最少组数, 分组方案)
        dp = [(float('inf'), [])] * (n + 1)
        dp[0] = (0, [])
        
        for i in range(n):
            if dp[i][0] == float('inf'):
                continue
            
            current_width = 0
            for j in range(i, n):
                # 计算将lines[i:j+1]作为一组的宽度
                if j == i:
                    current_width = widths[j]
                else:
                    current_width += gap_width + widths[j]
                
                # 检查是否超过最大宽度
                if current_width > self.max_concat_width:
                    break
                
                # 更新dp[j+1]
                new_groups = dp[i][0] + 1
                if new_groups < dp[j + 1][0]:
                    new_plan = dp[i][1] + [(i, j + 1)]
                    dp[j + 1] = (new_groups, new_plan)
        
        # 构建最终的分组
        groups = []
        for start, end in dp[n][1]:
            group_lines = lines[start:end]
            group_indices = list(range(start, end))
            concat_img = self._create_concatenated_image(group_lines)
            groups.append((concat_img, group_indices))
        
        return groups
    
    def _batch_ocr_processing(self, groups: List[Tuple[np.ndarray, List[int]]]) -> Dict[int, str]:
        """
        批量OCR处理，使用缓存优化
        """
        results = {}
        
        # 批量处理所有组
        for concat_img, indices in groups:
            # 使用缓存的OCR
            combined_text = self._ocr_with_cache(concat_img)
            
            if len(indices) == 1:
                results[indices[0]] = combined_text
            else:
                # 智能分割结果
                split_texts = self._intelligent_split(combined_text, len(indices))
                for idx, text in zip(indices, split_texts):
                    results[idx] = text
        
        return results
    
    def _intelligent_split(self, combined_text: str, num_parts: int) -> List[str]:
        """
        改进的智能分割算法
        """
        if not combined_text:
            return [''] * num_parts
        
        # 尝试多种分隔符
        separators = ['|', '｜', 'l', '丨', ' | ', '  ', '\t']
        for sep in separators:
            if combined_text.count(sep) >= num_parts - 1:
                parts = combined_text.split(sep)
                # 清理并返回正确数量的部分
                cleaned = [p.strip() for p in parts if p.strip()]
                if len(cleaned) >= num_parts:
                    return cleaned[:num_parts]
                elif len(cleaned) > 0:
                    # 补齐缺失的部分
                    return cleaned + [''] * (num_parts - len(cleaned))
        
        # 基于长度均分
        if len(combined_text) >= num_parts:
            avg_len = len(combined_text) // num_parts
            parts = []
            for i in range(num_parts):
                start = i * avg_len
                end = start + avg_len if i < num_parts - 1 else len(combined_text)
                parts.append(combined_text[start:end].strip())
            return parts
        
        # 默认处理
        return [combined_text] + [''] * (num_parts - 1)
    
    def recognize_multiline_optimized(self, image: np.ndarray) -> List[str]:
        """
        优化的多行识别流程 - 最少OCR调用
        """
        # 步骤1: 分析文字结构
        structure_info = self.analyze_text_structure(image)
        
        # 步骤2: 检测和切割文本行
        lines = self.detect_text_lines(image, structure_info)
        
        if not lines:
            return []
        
        # 步骤3: 激进的拼接优化
        concat_groups = self.optimize_concatenation_aggressive(lines)
        
        # 步骤4: 批量OCR处理（使用缓存）
        results_dict = self._batch_ocr_processing(concat_groups)
        
        # 按顺序返回结果
        results = []
        for i in range(len(lines)):
            if i in results_dict:
                results.append(results_dict[i])
        
        return [r for r in results if r]
    
    def _create_concatenated_image(self, lines: List[LineInfo]) -> np.ndarray:
        """创建拼接图像"""
        if len(lines) == 1:
            h, w = lines[0].image.shape[:2]
            if h != self.target_height:
                scale = self.target_height / h
                new_w = int(w * scale)
                return cv2.resize(lines[0].image, (new_w, self.target_height))
            return lines[0].image
        
        # 多行拼接
        resized_lines = []
        for line in lines:
            h, w = line.image.shape[:2]
            scale = self.target_height / h if h > 0 else 1.0
            new_w = max(1, int(w * scale))
            resized = cv2.resize(line.image, (new_w, self.target_height))
            resized_lines.append(resized)
        
        # 创建分隔符
        gap = np.ones((self.target_height, 20, 3), dtype=np.uint8) * 255
        cv2.line(gap, (10, 0), (10, self.target_height), (200, 200, 200), 1)
        
        # 拼接
        parts = []
        for i, line in enumerate(resized_lines):
            parts.append(line)
            if i < len(resized_lines) - 1:
                parts.append(gap)
        
        return np.hstack(parts)
    
    def _merge_close_lines(self, ranges: List[Tuple[int, int]], threshold: int) -> List[Tuple[int, int]]:
        """合并距离过近的行"""
        if not ranges:
            return ranges
        
        merged = []
        current_start, current_end = ranges[0]
        
        for i in range(1, len(ranges)):
            next_start, next_end = ranges[i]
            
            if next_start - current_end <= threshold:
                current_end = next_end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        
        merged.append((current_start, current_end))
        return merged
    
    def get_performance_stats_optimized(self, image: np.ndarray) -> Dict:
        """获取优化后的性能统计信息"""
        # 重置缓存统计
        self.cache_hits = 0
        self.cache_misses = 0
        
        start_time = time.time()
        
        # 执行完整流程
        structure_info = self.analyze_text_structure(image)
        analysis_time = time.time() - start_time
        
        lines = self.detect_text_lines(image, structure_info)
        detection_time = time.time() - start_time - analysis_time
        
        concat_groups = self.optimize_concatenation_aggressive(lines)
        concat_time = time.time() - start_time - analysis_time - detection_time
        
        # OCR识别
        ocr_start = time.time()
        results_dict = self._batch_ocr_processing(concat_groups)
        results = [results_dict.get(i, '') for i in range(len(lines))]
        results = [r for r in results if r]
        ocr_time = time.time() - ocr_start
        
        total_time = time.time() - start_time
        
        # 计算实际的OCR调用次数（考虑缓存）
        actual_ocr_calls = self.cache_misses
        theoretical_calls = len(concat_groups)
        
        return {
            'total_time_ms': total_time * 1000,
            'analysis_time_ms': analysis_time * 1000,
            'detection_time_ms': detection_time * 1000,
            'concat_time_ms': concat_time * 1000,
            'ocr_time_ms': ocr_time * 1000,
            'detected_lines': len(lines),
            'concat_groups': len(concat_groups),
            'theoretical_ocr_calls': theoretical_calls,
            'actual_ocr_calls': actual_ocr_calls,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'efficiency_ratio': len(lines) / theoretical_calls if theoretical_calls else 1,
            'actual_efficiency_ratio': len(lines) / actual_ocr_calls if actual_ocr_calls else 1,
            'structure_info': structure_info,
            'results': results
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.ocr_cache.clear()
        self.batch_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


if __name__ == "__main__":
    print("优化版智能多行OCR系统")
    print("="*70)
    print("核心优化策略：")
    print("1. ✅ 激进的拼接策略 - 最大化利用宽度限制")
    print("2. ✅ 动态规划分组 - 找到最少OCR调用方案")
    print("3. ✅ OCR结果缓存 - 避免重复识别相同图像")
    print("4. ✅ 批量处理优化 - 减少接口调用开销")
    print("5. ✅ 智能结果分割 - 准确拆分拼接结果")
