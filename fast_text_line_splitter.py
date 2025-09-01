# -*- coding: utf-8 -*-
"""
快速文字行分割器
多种方法实现5-10ms内完成文字行分割
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TextLine:
    """文字行信息"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center_y: int
    height: int
    confidence: float  # 是文字行的置信度


class FastTextLineSplitter:
    """快速文字行分割器"""
    
    def __init__(self, method: str = 'projection'):
        """
        初始化
        
        Args:
            method: 分割方法 'projection'(投影法) / 'contour'(轮廓法) / 'component'(连通域法)
        """
        self.method = method
        self.debug = False
    
    def split_lines(self, 
                   image: np.ndarray, 
                   min_line_height: int = 15,
                   min_line_width: int = 20) -> List[TextLine]:
        """
        快速分割文字行
        
        Args:
            image: 输入图像（文字区域）
            min_line_height: 最小行高
            min_line_width: 最小行宽
            
        Returns:
            文字行列表
        """
        if self.method == 'projection':
            return self._projection_method(image, min_line_height, min_line_width)
        elif self.method == 'contour':
            return self._contour_method(image, min_line_height, min_line_width)
        elif self.method == 'component':
            return self._component_method(image, min_line_height, min_line_width)
        else:
            raise ValueError(f"不支持的方法: {self.method}")


class ProjectionLineSplitter(FastTextLineSplitter):
    """投影法行分割器（最快，适合规整文字）"""
    
    def __init__(self):
        super().__init__('projection')
    
    def _projection_method(self, 
                          image: np.ndarray,
                          min_line_height: int = 15,
                          min_line_width: int = 20) -> List[TextLine]:
        """
        水平投影法分割文字行
        
        原理：
        1. 转为灰度图并二值化
        2. 计算水平投影（每行像素累加）
        3. 找到投影值的波峰和波谷
        4. 波谷之间就是文字行
        
        预期耗时：3-5ms
        """
        start_time = time.time()
        
        # 1. 预处理（1ms）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 自适应二值化（对光照变化鲁棒）
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        h, w = binary.shape
        
        # 2. 计算水平投影（1ms）
        # 每行的像素值累加
        projection = np.sum(binary, axis=1, dtype=np.int32)
        
        if self.debug:
            print(f"投影计算耗时: {(time.time() - start_time) * 1000:.1f}ms")
        
        # 3. 平滑投影曲线，去除噪声（0.5ms）
        kernel_size = max(3, min_line_height // 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 使用简单的移动平均代替高斯滤波
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(projection.astype(np.float32), kernel, mode='same')
        else:
            smoothed = projection.astype(np.float32)
        
        # 4. 找文字行边界（1ms）
        lines = self._find_text_lines_from_projection(
            smoothed, min_line_height, w, min_line_width
        )
        
        total_time = (time.time() - start_time) * 1000
        if self.debug:
            print(f"投影法总耗时: {total_time:.1f}ms, 找到{len(lines)}行")
        
        return lines
    
    def _find_text_lines_from_projection(self, 
                                        projection: np.ndarray,
                                        min_height: int,
                                        image_width: int,
                                        min_width: int) -> List[TextLine]:
        """从投影中找到文字行"""
        lines = []
        
        # 计算阈值：投影的平均值
        threshold = np.mean(projection) * 0.3  # 30%的平均值作为阈值
        
        # 找到所有高于阈值的区间
        above_threshold = projection > threshold
        
        # 找到连续区间的开始和结束
        diff = np.diff(above_threshold.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # 处理边界情况
        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(projection)]])
        
        # 生成文字行
        for start_y, end_y in zip(starts, ends):
            height = end_y - start_y
            
            # 过滤太小的行
            if height >= min_height:
                # 在这个行内找左右边界
                row_slice = projection[start_y:end_y]
                confidence = np.mean(row_slice) / 255.0  # 归一化置信度
                
                lines.append(TextLine(
                    bbox=(0, start_y, image_width, end_y),
                    center_y=(start_y + end_y) // 2,
                    height=height,
                    confidence=min(1.0, confidence)
                ))
        
        return lines


class ContourLineSplitter(FastTextLineSplitter):
    """轮廓法行分割器（适合复杂布局）"""
    
    def __init__(self):
        super().__init__('contour')
    
    def _contour_method(self, 
                       image: np.ndarray,
                       min_line_height: int = 15,
                       min_line_width: int = 20) -> List[TextLine]:
        """
        基于轮廓的文字行分割
        
        原理：
        1. 形态学处理连接文字
        2. 找轮廓
        3. 合并同一行的轮廓
        
        预期耗时：8-12ms
        """
        start_time = time.time()
        
        # 1. 预处理
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. 形态学操作连接文字（关键步骤）
        # 水平方向的核，用于连接同一行的文字
        kernel_width = max(5, min_line_width // 4)
        kernel_height = max(3, min_line_height // 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
        
        # 闭操作：先膨胀再腐蚀，连接断开的文字
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 3. 找轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4. 将轮廓转换为文字行
        lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 过滤太小的区域
            if h >= min_line_height and w >= min_line_width:
                # 计算置信度（基于轮廓面积与外接矩形面积的比值）
                contour_area = cv2.contourArea(contour)
                rect_area = w * h
                confidence = contour_area / rect_area if rect_area > 0 else 0
                
                lines.append(TextLine(
                    bbox=(x, y, x + w, y + h),
                    center_y=y + h // 2,
                    height=h,
                    confidence=min(1.0, confidence * 2)  # 放大置信度
                ))
        
        # 5. 按Y坐标排序
        lines.sort(key=lambda line: line.center_y)
        
        total_time = (time.time() - start_time) * 1000
        if self.debug:
            print(f"轮廓法总耗时: {total_time:.1f}ms, 找到{len(lines)}行")
        
        return lines


class ComponentLineSplitter(FastTextLineSplitter):
    """连通域法行分割器（平衡效果和速度）"""
    
    def __init__(self):
        super().__init__('component')
    
    def _component_method(self, 
                         image: np.ndarray,
                         min_line_height: int = 15,
                         min_line_width: int = 20) -> List[TextLine]:
        """
        基于连通域的文字行分割
        
        原理：
        1. 找所有连通域（字符）
        2. 按Y坐标聚类成行
        3. 计算每行的边界框
        
        预期耗时：6-10ms
        """
        start_time = time.time()
        
        # 1. 预处理
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. 找连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # 3. 过滤连通域并按Y坐标分组
        components = []
        for i in range(1, num_labels):  # 跳过背景(label=0)
            x, y, w, h, area = stats[i]
            
            # 过滤太小的连通域
            if h >= 8 and w >= 3 and area >= 20:  # 最小字符尺寸
                components.append({
                    'bbox': (x, y, w, h),
                    'center_y': y + h // 2,
                    'area': area
                })
        
        # 4. 按Y坐标聚类成行
        if not components:
            return []
        
        lines = self._cluster_components_into_lines(components, min_line_height)
        
        total_time = (time.time() - start_time) * 1000
        if self.debug:
            print(f"连通域法总耗时: {total_time:.1f}ms, 找到{len(lines)}行")
        
        return lines
    
    def _cluster_components_into_lines(self, 
                                      components: List[dict],
                                      min_line_height: int) -> List[TextLine]:
        """将连通域聚类成文字行"""
        if not components:
            return []
        
        # 按Y坐标排序
        components.sort(key=lambda c: c['center_y'])
        
        lines = []
        current_line_components = [components[0]]
        
        for i in range(1, len(components)):
            current = components[i]
            prev = components[i-1]
            
            # 如果Y坐标相近，认为是同一行
            y_diff = abs(current['center_y'] - prev['center_y'])
            max_height = max(current['bbox'][3], prev['bbox'][3])
            
            if y_diff <= max_height * 0.5:  # 50%高度差以内认为同行
                current_line_components.append(current)
            else:
                # 结束当前行，开始新行
                if current_line_components:
                    line = self._create_line_from_components(current_line_components)
                    if line and line.height >= min_line_height:
                        lines.append(line)
                current_line_components = [current]
        
        # 处理最后一行
        if current_line_components:
            line = self._create_line_from_components(current_line_components)
            if line and line.height >= min_line_height:
                lines.append(line)
        
        return lines
    
    def _create_line_from_components(self, components: List[dict]) -> Optional[TextLine]:
        """从连通域列表创建文字行"""
        if not components:
            return None
        
        # 计算整行的边界框
        min_x = min(c['bbox'][0] for c in components)
        max_x = max(c['bbox'][0] + c['bbox'][2] for c in components)
        min_y = min(c['bbox'][1] for c in components)
        max_y = max(c['bbox'][1] + c['bbox'][3] for c in components)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # 计算置信度（基于连通域数量和密度）
        num_components = len(components)
        total_area = sum(c['area'] for c in components)
        line_area = width * height
        confidence = min(1.0, (total_area / line_area) * (num_components / 10))
        
        return TextLine(
            bbox=(min_x, min_y, max_x, max_y),
            center_y=(min_y + max_y) // 2,
            height=height,
            confidence=confidence
        )


class HybridLineSplitter:
    """混合策略行分割器（自动选择最佳方法）"""
    
    def __init__(self):
        self.projection_splitter = ProjectionLineSplitter()
        self.contour_splitter = ContourLineSplitter()
        self.component_splitter = ComponentLineSplitter()
    
    def split_lines(self, image: np.ndarray, **kwargs) -> List[TextLine]:
        """
        自动选择最佳分割方法
        
        策略：
        1. 先用最快的投影法
        2. 如果效果不好，用轮廓法
        3. 特殊情况用连通域法
        """
        h, w = image.shape[:2]
        
        # 图像特征分析
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 计算图像复杂度（边缘密度）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # 根据复杂度选择方法
        if edge_density < 0.1:  # 简单图像，文字清晰
            method = 'projection'
        elif edge_density < 0.3:  # 中等复杂度
            method = 'component'
        else:  # 复杂图像
            method = 'contour'
        
        # 执行分割
        if method == 'projection':
            lines = self.projection_splitter._projection_method(image, **kwargs)
        elif method == 'component':
            lines = self.component_splitter._component_method(image, **kwargs)
        else:
            lines = self.contour_splitter._contour_method(image, **kwargs)
        
        # 后处理：合并过近的行
        lines = self._merge_close_lines(lines, min_gap=5)
        
        return lines
    
    def _merge_close_lines(self, lines: List[TextLine], min_gap: int = 5) -> List[TextLine]:
        """合并过于接近的文字行"""
        if len(lines) <= 1:
            return lines
        
        merged = []
        current = lines[0]
        
        for i in range(1, len(lines)):
            next_line = lines[i]
            gap = next_line.bbox[1] - current.bbox[3]  # 上一行底部到下一行顶部的距离
            
            if gap < min_gap:  # 行距太小，合并
                # 合并边界框
                merged_bbox = (
                    min(current.bbox[0], next_line.bbox[0]),
                    current.bbox[1],
                    max(current.bbox[2], next_line.bbox[2]),
                    next_line.bbox[3]
                )
                current = TextLine(
                    bbox=merged_bbox,
                    center_y=(merged_bbox[1] + merged_bbox[3]) // 2,
                    height=merged_bbox[3] - merged_bbox[1],
                    confidence=max(current.confidence, next_line.confidence)
                )
            else:
                merged.append(current)
                current = next_line
        
        merged.append(current)
        return merged


def create_test_image():
    """创建测试图像"""
    img = np.ones((300, 600, 3), dtype=np.uint8) * 255
    
    # 添加多行文字
    lines = [
        "This is line 1",
        "Second line with more text",
        "Line 3 is shorter",
        "The fourth line contains numbers 123",
        "Final line for testing"
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, text in enumerate(lines):
        y = 50 + i * 50
        cv2.putText(img, text, (20, y), font, 0.8, (0, 0, 0), 2)
    
    return img


def main():
    """测试不同的行分割方法"""
    print("="*70)
    print("快速文字行分割测试")
    print("="*70)
    
    # 创建测试图像
    test_img = create_test_image()
    cv2.imwrite("test_line_split_input.jpg", test_img)
    print("测试图像已保存: test_line_split_input.jpg")
    
    methods = [
        ('投影法', ProjectionLineSplitter()),
        ('轮廓法', ContourLineSplitter()),
        ('连通域法', ComponentLineSplitter()),
        ('混合策略', HybridLineSplitter())
    ]
    
    for name, splitter in methods:
        print(f"\n{name}测试:")
        print("-" * 40)
        
        # 测试10次取平均
        times = []
        for _ in range(10):
            start = time.time()
            if hasattr(splitter, 'split_lines'):
                lines = splitter.split_lines(test_img)
            else:
                lines = splitter._projection_method(test_img)  # 投影法
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        print(f"平均耗时: {avg_time:.1f}ms")
        print(f"检测到行数: {len(lines)}")
        
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line.bbox
            print(f"  行{i+1}: ({x1}, {y1}) -> ({x2}, {y2}), 置信度: {line.confidence:.3f}")
    
    print(f"\n✅ 测试完成！")


if __name__ == "__main__":
    main()
