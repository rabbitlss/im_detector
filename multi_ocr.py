# -*- coding: utf-8 -*-
"""
智能多行OCR系统
完整的文字检测、分析、切割、拼接、识别流程
只使用单行识别模型，通过图像处理实现多行识别
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


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
    confidence: float  # 这是一行的置信度
    effective_width: int = 0  # 文字的有效宽度（非空白部分）


class IntelligentMultilineOCR:
    """智能多行OCR系统"""
    
    def __init__(self, 
                 ocr_engine,
                 max_concat_width: int = 3840,  # 增大到3840，允许更多拼接
                 target_height: int = 48,
                 dynamic_width: bool = True,
                 width_strategy: str = 'adaptive'):
        """
        初始化智能多行OCR
        
        Args:
            ocr_engine: 单行OCR引擎
            max_concat_width: 最大拼接宽度（当dynamic_width=False时使用）
            target_height: OCR模型目标高度
            dynamic_width: 是否启用动态宽度计算
            width_strategy: 宽度策略 ('conservative', 'balanced', 'aggressive', 'adaptive')
        """
        self.ocr = ocr_engine
        self.max_concat_width = max_concat_width
        self.target_height = target_height
        self.dynamic_width = dynamic_width
        self.width_strategy = width_strategy
    
    def _calculate_effective_width(self, image: np.ndarray) -> int:
        """
        计算图像中文字的有效宽度（非空白部分）
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 二值化找到文字区域
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 垂直投影，找到有文字的列
        v_projection = np.sum(binary == 255, axis=0)
        
        # 找到第一个和最后一个有文字的列
        non_zero_cols = np.where(v_projection > 0)[0]
        
        if len(non_zero_cols) == 0:
            return image.shape[1]  # 如果没有文字，返回整个宽度
        
        # 有效宽度 = 最右列 - 最左列 + padding
        effective_width = non_zero_cols[-1] - non_zero_cols[0] + 1
        
        # 添加一些padding
        padding = 20
        return min(effective_width + padding, image.shape[1])
    
    def _calculate_dynamic_width(self, lines: List[LineInfo]) -> int:
        """
        根据文字特征动态计算最大拼接宽度
        """
        if not self.dynamic_width or not lines:
            return self.max_concat_width
        
        # 分析文字特征
        effective_widths = [line.effective_width for line in lines]
        char_heights = [line.char_height for line in lines]
        
        avg_width = np.mean(effective_widths)
        max_width = max(effective_widths)
        avg_height = np.mean(char_heights)
        
        # 根据不同策略计算动态宽度
        if self.width_strategy == 'conservative':
            # 保守策略：基于平均宽度的2-3倍
            dynamic_width = int(avg_width * 2.5)
            
        elif self.width_strategy == 'balanced':
            # 平衡策略：基于平均宽度和最大宽度
            dynamic_width = int(avg_width * 3 + (max_width - avg_width) * 0.5)
            
        elif self.width_strategy == 'aggressive':
            # 激进策略：尽可能多拼接
            dynamic_width = int(max_width * 6)  # 允许6行最宽内容拼接
            
        elif self.width_strategy == 'adaptive':
            # 自适应策略：根据文字密度和OCR模型能力
            text_density = avg_width / (avg_height * 20)  # 估算文字密度
            
            if text_density < 1:  # 稀疏文字
                dynamic_width = int(avg_width * 8)  # 可以拼接更多
            elif text_density < 2:  # 中等密度
                dynamic_width = int(avg_width * 5)
            else:  # 密集文字
                dynamic_width = int(avg_width * 3)
            
            # 考虑OCR模型处理能力（经验值）
            if avg_height < 20:  # 小字体，OCR处理能力强
                dynamic_width = int(dynamic_width * 1.5)
            elif avg_height > 60:  # 大字体，适当限制
                dynamic_width = int(dynamic_width * 0.8)
                
        else:
            dynamic_width = self.max_concat_width
        
        # 设置合理的上下限
        min_width = int(max_width + 100)  # 至少能容纳最宽的一行
        max_possible_width = 8192  # 技术上限
        
        dynamic_width = max(min_width, min(dynamic_width, max_possible_width))
        
        return dynamic_width
    
    def analyze_text_structure(self, image: np.ndarray) -> Dict:
        """
        步骤1: 分析图像中的文字结构
        通过图像变换识别出文字区域，检测文字大小和像素特征
        
        Args:
            image: 输入图像
            
        Returns:
            文字结构分析结果
        """
        # 1.1 预处理
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1.2 二值化 - 突出文字区域
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 1.3 形态学操作 - 连接同一行的字符
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        connected_text = cv2.dilate(binary, kernel_horizontal, iterations=1)
        
        # 1.4 连通组件分析 - 找到所有文字块
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # 1.5 提取有效文字区域
        text_regions = []
        for i in range(1, num_labels):  # 跳过背景
            x, y, w, h, area = stats[i]
            
            # 过滤条件：面积、宽高比、尺寸
            if (area > 20 and 
                h > 5 and w > 3 and 
                0.1 < h/w < 10):  # 合理的宽高比
                
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
        
        # 字符高度分析
        heights = [r.height for r in text_regions]
        char_height = int(np.median(heights))
        char_height_std = int(np.std(heights))
        
        # 行高和行间距估算
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
        步骤2: 基于结构分析结果，智能检测和切割文本行
        根据文字大小决定切割成几行
        
        Args:
            image: 输入图像
            structure_info: 文字结构信息
            
        Returns:
            检测到的文本行列表
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
        # 基于字符高度设置最小行高
        min_line_height = max(8, int(char_height * 0.6))
        
        # 基于行间距设置合并阈值  
        merge_threshold = max(3, int(line_spacing * 0.8))
        
        # 2.3 检测行边界
        line_ranges = []
        in_line = False
        start_y = 0
        
        # 使用自适应阈值（基于投影的统计特征）
        proj_threshold = max(1, int(np.mean(h_projection[h_projection > 0]) * 0.1))
        
        for y in range(len(h_projection)):
            if h_projection[y] > proj_threshold and not in_line:
                start_y = y
                in_line = True
            elif h_projection[y] <= proj_threshold and in_line:
                if y - start_y >= min_line_height:
                    line_ranges.append((start_y, y))
                in_line = False
        
        # 处理最后一行
        if in_line and len(h_projection) - start_y >= min_line_height:
            line_ranges.append((start_y, len(h_projection)))
        
        # 2.4 合并过于接近的行
        merged_ranges = self._merge_close_lines(line_ranges, merge_threshold)
        
        # 2.5 创建LineInfo对象
        lines = []
        for i, (start_y, end_y) in enumerate(merged_ranges):
            # 添加适当边距
            padding = max(2, char_height // 8)
            y1 = max(0, start_y - padding)
            y2 = min(image.shape[0], end_y + padding)
            
            line_img = image[y1:y2, :]
            if line_img.size == 0:
                continue
                
            # 分析这一行的特征
            line_height = y2 - y1
            line_width = image.shape[1]
            
            # 关键：计算文字的有效宽度
            effective_width = self._calculate_effective_width(line_img)
            
            # 估算字符数量 - 使用有效宽度而不是整个图像宽度
            avg_char_width = char_height * 0.8  # 估算字符宽度
            estimated_chars = max(1, int(effective_width / avg_char_width))
            
            # 计算置信度（基于投影强度和高度一致性）
            line_projection_sum = np.sum(h_projection[start_y:end_y])
            max_possible_sum = (end_y - start_y) * line_width
            confidence = min(1.0, line_projection_sum / max_possible_sum) if max_possible_sum > 0 else 0.5
            
            line_info = LineInfo(
                image=line_img,
                bbox=(0, y1, line_width, line_height),
                estimated_chars=estimated_chars,
                char_height=char_height,
                char_width_avg=avg_char_width,
                confidence=confidence,
                effective_width=effective_width  # 保存有效宽度
            )
            lines.append(line_info)
        
        return lines
    
    def optimize_concatenation(self, lines: List[LineInfo]) -> List[Tuple[np.ndarray, List[int]]]:
        """
        步骤3: 智能拼接优化
        根据文字特征决定最佳拼接策略
        
        Args:
            lines: 文本行列表
            
        Returns:
            [(拼接后的图片, 原始行索引列表), ...]
        """
        if not lines:
            return []
        
        # 3.1 动态计算最大拼接宽度
        dynamic_max_width = self._calculate_dynamic_width(lines)
        
        # 3.2 预估每行缩放后的宽度 - 使用有效宽度而不是图像宽度
        estimated_widths = []
        for line in lines:
            h = line.image.shape[0]
            scale_ratio = self.target_height / h if h > 0 else 1.0
            # 关键修改：使用有效宽度而不是图像宽度
            scaled_width = int(line.effective_width * scale_ratio)
            estimated_widths.append(scaled_width)
        
        # 3.3 智能分组策略
        groups = []
        current_group = []
        current_indices = []
        current_width = 0
        gap_width = 20  # 行间分隔符宽度
        
        for i, (line, width) in enumerate(zip(lines, estimated_widths)):
            # 3.3 分组决策
            needed_width = width + (gap_width if current_group else 0)
            
            # 考虑多个因素决定是否分组 - 使用动态宽度限制
            can_group = (
                current_width + needed_width <= dynamic_max_width and  # 动态宽度限制
                len(current_group) < 20 and  # 行数限制（进一步增加到20）
                self._should_group_lines_relaxed(current_group, line) if current_group else True  # 放松相似性
            )
            
            if can_group:
                current_group.append(line)
                current_indices.append(i)
                current_width += needed_width
            else:
                # 保存当前组
                if current_group:
                    concat_img = self._create_concatenated_image(current_group)
                    groups.append((concat_img, current_indices))
                
                # 开始新组
                current_group = [line]
                current_indices = [i]
                current_width = width
        
        # 保存最后一组
        if current_group:
            concat_img = self._create_concatenated_image(current_group)
            groups.append((concat_img, current_indices))
        
        return groups
    
    def _should_group_lines(self, current_group: List[LineInfo], new_line: LineInfo) -> bool:
        """
        判断是否应该将新行加入当前组
        基于字符高度、密度等相似性
        """
        if not current_group:
            return True
        
        # 获取当前组的平均特征
        avg_char_height = np.mean([line.char_height for line in current_group])
        avg_confidence = np.mean([line.confidence for line in current_group])
        
        # 相似性检查
        height_similarity = abs(new_line.char_height - avg_char_height) / avg_char_height < 0.5
        confidence_similarity = abs(new_line.confidence - avg_confidence) < 0.3
        
        return height_similarity and confidence_similarity
    
    def _should_group_lines_relaxed(self, current_group: List[LineInfo], new_line: LineInfo) -> bool:
        """
        放松的行分组判断 - 增加拼接概率
        """
        if not current_group:
            return True
        
        # 更宽松的条件
        avg_char_height = np.mean([line.char_height for line in current_group])
        
        # 只检查字符高度，允许更大的差异
        height_similarity = abs(new_line.char_height - avg_char_height) / avg_char_height < 1.0  # 从0.5放松到1.0
        
        return height_similarity
    
    def _create_concatenated_image(self, lines: List[LineInfo]) -> np.ndarray:
        """创建拼接图像"""
        if len(lines) == 1:
            # 单行：直接缩放到目标高度
            h, w = lines[0].image.shape[:2]
            if h != self.target_height:
                scale = self.target_height / h
                new_w = int(w * scale)
                return cv2.resize(lines[0].image, (new_w, self.target_height))
            return lines[0].image
        
        # 多行：缩放后拼接
        resized_lines = []
        for line in lines:
            h, w = line.image.shape[:2]
            scale = self.target_height / h if h > 0 else 1.0
            new_w = max(1, int(w * scale))
            resized = cv2.resize(line.image, (new_w, self.target_height))
            resized_lines.append(resized)
        
        # 创建分隔符
        gap = np.ones((self.target_height, 20, 3), dtype=np.uint8) * 255
        # 绘制细分割线
        cv2.line(gap, (10, 0), (10, self.target_height), (200, 200, 200), 1)
        
        # 拼接所有部分
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
    
    def recognize_multiline(self, image: np.ndarray) -> List[str]:
        """
        完整的多行识别流程
        
        Args:
            image: 输入图像
            
        Returns:
            识别结果列表
        """
        # 步骤1: 分析文字结构
        structure_info = self.analyze_text_structure(image)
        
        # 步骤2: 检测和切割文本行
        lines = self.detect_text_lines(image, structure_info)
        
        if not lines:
            return []
        
        # 步骤3: 智能拼接优化
        concat_groups = self.optimize_concatenation(lines)
        
        # 步骤4: OCR识别
        all_results = [''] * len(lines)
        
        for concat_img, indices in concat_groups:
            if len(indices) == 1:
                # 单行直接识别
                text = self.ocr.recognize_single_line(concat_img)
                all_results[indices[0]] = text
            else:
                # 多行拼接识别后分割
                combined_text = self.ocr.recognize_single_line(concat_img)
                split_texts = self._split_concatenated_result(combined_text, len(indices))
                
                for idx, text in zip(indices, split_texts):
                    all_results[idx] = text
        
        return [r for r in all_results if r]  # 过滤空结果
    
    def _split_concatenated_result(self, combined_text: str, num_parts: int) -> List[str]:
        """
        分割拼接后的识别结果
        尝试多种分割策略
        """
        if not combined_text:
            return [''] * num_parts
        
        # 策略1: 基于分隔符分割
        separators = ['|', '｜', 'l', '丨', ' | ']
        for sep in separators:
            if sep in combined_text:
                parts = combined_text.split(sep)
                if len(parts) == num_parts:
                    return [p.strip() for p in parts]
        
        # 策略2: 智能分割（基于字符密度）
        if len(combined_text) > num_parts * 2:
            avg_len = len(combined_text) // num_parts
            parts = []
            for i in range(num_parts):
                start = i * avg_len
                end = start + avg_len if i < num_parts - 1 else len(combined_text)
                parts.append(combined_text[start:end].strip())
            return parts
        
        # 策略3: 单字符分配
        return [combined_text] + [''] * (num_parts - 1)
    
    def get_performance_stats(self, image: np.ndarray) -> Dict:
        """获取性能统计信息"""
        start_time = time.time()
        
        # 执行完整流程
        structure_info = self.analyze_text_structure(image)
        analysis_time = time.time() - start_time
        
        lines = self.detect_text_lines(image, structure_info)
        detection_time = time.time() - start_time - analysis_time
        
        concat_groups = self.optimize_concatenation(lines)
        concat_time = time.time() - start_time - analysis_time - detection_time
        
        # OCR识别 - 只执行OCR部分，不重复前面的步骤
        ocr_start = time.time()
        all_results = [''] * len(lines)
        
        for concat_img, indices in concat_groups:
            if len(indices) == 1:
                # 单行直接识别
                text = self.ocr.recognize_single_line(concat_img)
                all_results[indices[0]] = text
            else:
                # 多行拼接识别后分割
                combined_text = self.ocr.recognize_single_line(concat_img)
                split_texts = self._split_concatenated_result(combined_text, len(indices))
                
                for idx, text in zip(indices, split_texts):
                    all_results[idx] = text
        
        results = [r for r in all_results if r]  # 过滤空结果
        ocr_time = time.time() - ocr_start
        
        total_time = time.time() - start_time
        
        return {
            'total_time_ms': total_time * 1000,
            'analysis_time_ms': analysis_time * 1000,
            'detection_time_ms': detection_time * 1000,
            'concat_time_ms': concat_time * 1000,
            'ocr_time_ms': ocr_time * 1000,
            'detected_lines': len(lines),
            'concat_groups': len(concat_groups),
            'ocr_calls': len(concat_groups),
            'efficiency_ratio': len(lines) / len(concat_groups) if concat_groups else 1,
            'structure_info': structure_info,
            'results': results
        }


if __name__ == "__main__":
    print("智能多行OCR系统测试")
    print("="*70)
    
    # 这里需要实际的OCR引擎来测试
    # from ultrafast_ocr import UltraFastOCR
    # ocr = UltraFastOCR()
    # intelligent_ocr = IntelligentMultilineOCR(ocr)
    
    print("系统设计完成，包含以下核心模块：")
    print("1. ✅ 图像变换与文字区域检测")
    print("2. ✅ 文字大小和像素特征分析") 
    print("3. ✅ 智能行检测和切割")
    print("4. ✅ 自适应拼接优化")
    print("5. ✅ 单行OCR识别集成")
