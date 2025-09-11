# -*- coding: utf-8 -*-
"""
批量OCR识别处理器
实现：检测多个UI元素后，只调用一次OCR识别模型获取所有文字结果
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
import os
from dataclasses import dataclass

@dataclass 
class TextRegion:
    """文字区域信息"""
    region_id: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    image_crop: np.ndarray
    source_type: str  # 'chat_area', 'input_area', etc.
    confidence: float = 0.0

@dataclass
class BatchOCRResult:
    """批量OCR结果"""
    region_id: str
    text_content: str
    confidence: float
    processing_time_ms: float

class BatchOCRProcessor:
    """批量OCR识别处理器"""
    
    def __init__(self, max_batch_size: int = 32, 
                 target_height: int = 48,  # 增加到48px以保留更多细节
                 padding: int = 5,
                 use_separator: bool = True):
        """
        初始化批量OCR处理器
        
        Args:
            max_batch_size: 最大批处理大小
            target_height: 标准化高度
            padding: 图像间填充像素
            use_separator: 是否使用分隔符
        """
        self.max_batch_size = max_batch_size
        self.target_height = target_height
        self.padding = padding
        self.use_separator = use_separator
        self.total_processing_time = 0
        self.total_regions_processed = 0
        
        # 文字大小分类阈值
        self.size_thresholds = {
            'small': (0, 20),     # 小字体
            'medium': (20, 35),   # 中等字体  
            'large': (35, 100)    # 大字体
        }
        
        # 创建分隔符图像（黑色竖线）
        if self.use_separator:
            self.separator = self._create_separator()
    
    def _create_separator(self) -> np.ndarray:
        """创建分隔符图像（黑色竖线，易于OCR识别为 | ）"""
        # 创建白色背景
        separator = np.ones((self.target_height, self.padding, 3), dtype=np.uint8) * 255
        # 在中间画黑色竖线
        mid_x = self.padding // 2
        separator[:, mid_x:mid_x+1, :] = 0  # 黑色竖线
        return separator
        
    def collect_text_regions(self, image: np.ndarray, 
                           yolo_detections: List[Dict],
                           line_splitter) -> List[TextRegion]:
        """
        从YOLO检测结果中收集所有文字区域
        
        Args:
            image: 原始图像
            yolo_detections: YOLO检测结果
            line_splitter: 文字行分割器
            
        Returns:
            文字区域列表
        """
        
        print(f"📥 收集文字区域中...")
        text_regions = []
        
        for detection in yolo_detections:
            class_name = detection['class']
            bbox = detection['bbox']
            confidence = detection.get('confidence', 0.0)
            
            # 裁剪检测区域
            x1, y1, x2, y2 = bbox
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                continue
                
            # 使用行分割器获取文字行
            lines = line_splitter._projection_method(region)
            
            # 为每个文字行创建TextRegion
            for i, line in enumerate(lines):
                line_x1, line_y1, line_x2, line_y2 = line.bbox
                
                # 裁剪单行文字图像
                line_crop = region[line_y1:line_y2, line_x1:line_x2]
                
                if line_crop.size == 0:
                    continue
                    
                region_id = f"{class_name}_{i+1}"
                
                text_region = TextRegion(
                    region_id=region_id,
                    bbox=(x1 + line_x1, y1 + line_y1, 
                          x1 + line_x2, y1 + line_y2),
                    image_crop=line_crop,
                    source_type=class_name,
                    confidence=line.confidence
                )
                
                text_regions.append(text_region)
                
        print(f"✅ 收集到 {len(text_regions)} 个文字区域")
        return text_regions
    
    def create_batch_image(self, text_regions: List[TextRegion]) -> Tuple[np.ndarray, List[Dict]]:
        """
        将多个文字区域拼接成批处理图像
        
        Args:
            text_regions: 文字区域列表
            
        Returns:
            批处理图像和区域映射信息
        """
        
        if not text_regions:
            return np.array([]), []
            
        print(f"🔧 创建批处理图像，包含 {len(text_regions)} 个区域...")
        
        # 根据原始高度分组，避免过度缩放
        height_groups = self._group_by_similar_height(text_regions)
        
        normalized_regions = []
        region_mappings = []
        
        current_x = 0
        
        for region in text_regions:
            # 调整图像大小到目标高度
            crop = region.image_crop
            h, w = crop.shape[:2]
            
            # 智能缩放策略
            scale = self._calculate_optimal_scale(h)
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # 调整大小
            if len(crop.shape) == 3:
                resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            
            # 填充到标准高度
            padded = self._pad_to_target_height(resized)
                
            normalized_regions.append(padded)
            
            # 记录区域在批处理图像中的位置
            region_mapping = {
                'region_id': region.region_id,
                'source_type': region.source_type,
                'original_bbox': region.bbox,
                'batch_bbox': (current_x, 0, current_x + new_w, self.target_height),
                'confidence': region.confidence,
                'original_height': h,
                'scale_factor': scale
            }
            region_mappings.append(region_mapping)
            
            current_x += new_w + self.padding
            
        # 拼接所有区域
        if normalized_regions:
            # 如果使用分隔符，调整总宽度计算
            if self.use_separator:
                # 每个区域后面都有分隔符（除了最后一个）
                total_width = sum(r.shape[1] for r in normalized_regions) + self.padding * (len(normalized_regions) - 1)
            else:
                total_width = current_x - self.padding  # 去掉最后一个padding
                
            batch_image = np.ones((self.target_height, total_width, 3), dtype=np.uint8) * 255
            
            current_x = 0
            for i, resized_region in enumerate(normalized_regions):
                h, w = resized_region.shape[:2]
                batch_image[0:h, current_x:current_x+w] = resized_region
                current_x += w
                
                # 添加分隔符（除了最后一个区域）
                if self.use_separator and i < len(normalized_regions) - 1:
                    batch_image[:, current_x:current_x+self.padding] = self.separator
                    current_x += self.padding
                elif not self.use_separator:
                    current_x += self.padding
                
            print(f"✅ 批处理图像创建完成: {batch_image.shape}")
            return batch_image, region_mappings
        else:
            return np.array([]), []
    
    def _calculate_optimal_scale(self, original_height: int) -> float:
        """
        计算最优缩放比例，避免过度缩放
        
        Args:
            original_height: 原始高度
            
        Returns:
            缩放比例
        """
        # 根据原始高度选择缩放策略
        if original_height < 20:
            # 小字体：放大到目标高度
            return self.target_height / original_height
        elif original_height < 35:
            # 中等字体：适度缩放
            return min(self.target_height / original_height, 1.5)
        else:
            # 大字体：限制缩放比例
            return min(self.target_height / original_height, 1.0)
    
    def _pad_to_target_height(self, image: np.ndarray) -> np.ndarray:
        """
        将图像填充到目标高度
        
        Args:
            image: 输入图像
            
        Returns:
            填充后的图像
        """
        h, w = image.shape[:2]
        
        if h >= self.target_height:
            # 如果高度超过目标，裁剪中心部分
            start_y = (h - self.target_height) // 2
            return image[start_y:start_y + self.target_height, :]
        
        # 填充到目标高度
        pad_top = (self.target_height - h) // 2
        pad_bottom = self.target_height - h - pad_top
        
        padded = np.ones((self.target_height, w, 3), dtype=np.uint8) * 255
        padded[pad_top:pad_top + h, :] = image
        
        return padded
    
    def _group_by_similar_height(self, text_regions: List[TextRegion]) -> Dict[str, List[TextRegion]]:
        """
        根据相似高度分组文字区域
        
        Args:
            text_regions: 文字区域列表
            
        Returns:
            按高度分组的字典
        """
        groups = {'small': [], 'medium': [], 'large': []}
        
        for region in text_regions:
            h = region.image_crop.shape[0]
            
            for size_type, (min_h, max_h) in self.size_thresholds.items():
                if min_h <= h < max_h:
                    groups[size_type].append(region)
                    break
                    
        return groups
    
    def batch_ocr_recognition(self, batch_image: np.ndarray, 
                               region_mappings: List[Dict]) -> List[str]:
        """
        真实OCR识别过程（单次调用处理整个批次）
        
        Args:
            batch_image: 批处理图像  
            region_mappings: 区域映射信息
            
        Returns:
            各个区域的识别文本列表
        """
        
        if batch_image.size == 0:
            return []
            
        print(f"🔍 执行批量OCR识别...")
        start_time = time.time()
        
        # 真实OCR模型调用
        from ultrafast_ocr.core import UltraFastOCR
        ocr = UltraFastOCR()
        
        # 关键：只调用一次OCR识别整个拼接后的图像
        combined_text = ocr.recognize_single_line(batch_image)
        
        # 计算总处理时间
        total_time_ms = (time.time() - start_time) * 1000
        
        print(f"✅ 批量OCR识别完成（单次调用）:")
        print(f"   - OCR调用次数: 1 次")
        print(f"   - 处理区域数: {len(region_mappings)}")
        print(f"   - 总耗时: {total_time_ms:.1f}ms")
        print(f"   - 识别结果: {combined_text}")
        
        # 分割识别结果到各个区域，返回文本列表
        text_parts = self._split_combined_result(combined_text, len(region_mappings))
        
        # 更新统计信息
        self.total_processing_time += total_time_ms
        self.total_regions_processed += len(region_mappings)
        
        # 直接返回各部分文本的列表
        return text_parts
    
    def _split_combined_result(self, combined_text: str, num_regions: int) -> List[str]:
        """
        分割合并的OCR结果
        
        Args:
            combined_text: 合并的识别文本
            num_regions: 区域数量
            
        Returns:
            分割后的文本列表
        """
        if not combined_text:
            return [''] * num_regions
        
        # 如果只有一个区域，直接返回
        if num_regions == 1:
            return [combined_text]
        
        # 尝试多种分割策略
        # 策略1: 按竖线分割（使用分隔符时，OCR应该识别为竖线）
        if self.use_separator:
            # 尝试多种可能的竖线字符
            separators = ['|', '｜', 'I', 'l', '1']  # 竖线可能被识别为这些字符
            for sep in separators:
                if sep in combined_text:
                    parts = combined_text.split(sep)
                    # 允许一定的容错（±1）
                    if abs(len(parts) - num_regions) <= 1:
                        # 调整到正确数量
                        if len(parts) > num_regions:
                            # 合并最后的部分
                            parts = parts[:num_regions-1] + [sep.join(parts[num_regions-1:])]
                        elif len(parts) < num_regions:
                            # 添加空字符串
                            parts.extend([''] * (num_regions - len(parts)))
                        return [p.strip() for p in parts]
        
        # 策略2: 按多个空格分割（不使用分隔符时）
        import re
        parts = re.split(r'\s{2,}', combined_text)
        if abs(len(parts) - num_regions) <= 1:
            if len(parts) > num_regions:
                parts = parts[:num_regions]
            elif len(parts) < num_regions:
                parts.extend([''] * (num_regions - len(parts)))
            return [p.strip() for p in parts]
        
        # 策略3: 按固定长度分割（兜底方案）
        avg_len = len(combined_text) // num_regions
        parts = []
        for i in range(num_regions):
            start = i * avg_len
            end = start + avg_len if i < num_regions - 1 else len(combined_text)
            parts.append(combined_text[start:end].strip())
        return parts
    
    def simulate_ocr_recognition(self, batch_image: np.ndarray, 
                               region_mappings: List[Dict]) -> List[BatchOCRResult]:
        """
        模拟OCR识别（用于测试，不需要真实OCR模型）
        """
        # 调用真实OCR获取文本列表
        text_results = self.batch_ocr_recognition(batch_image, region_mappings)
        
        # 转换为BatchOCRResult对象列表
        results = []
        for text, mapping in zip(text_results, region_mappings):
            result = BatchOCRResult(
                region_id=mapping['region_id'],
                text_content=text,
                confidence=0.95,
                processing_time_ms=0
            )
            results.append(result)
        
        return results
    
    def process_im_image_batch(self, image: np.ndarray,
                             yolo_detections: List[Dict],
                             line_splitter) -> Dict:
        """
        批量处理IM图像中的所有文字区域
        
        Args:
            image: IM截图
            yolo_detections: YOLO检测结果
            line_splitter: 文字行分割器
            
        Returns:
            批量处理结果
        """
        
        print(f"\n🚀 开始批量OCR处理...")
        print(f"   输入: {len(yolo_detections)} 个YOLO检测区域")
        
        start_time = time.time()
        
        # 步骤1: 收集所有文字区域
        text_regions = self.collect_text_regions(image, yolo_detections, line_splitter)
        
        if not text_regions:
            print("❌ 未发现任何文字区域")
            return {
                'success': False,
                'message': '未发现文字区域',
                'results': [],
                'stats': {}
            }
        
        # 步骤2: 创建批处理图像
        batch_image, region_mappings = self.create_batch_image(text_regions)
        
        # 步骤3: 执行批量OCR识别（核心：只调用一次）
        text_results = self.batch_ocr_recognition(batch_image, region_mappings)
        
        # 将文本结果转换为BatchOCRResult对象
        ocr_results = []
        for i, (text, mapping) in enumerate(zip(text_results, region_mappings)):
            result = BatchOCRResult(
                region_id=mapping['region_id'],
                text_content=text,
                confidence=0.95,  # 默认置信度
                processing_time_ms=0  # 稍后计算
            )
            ocr_results.append(result)
        
        # 步骤4: 整理结果
        total_time = (time.time() - start_time) * 1000
        
        # 按源类型分组结果
        grouped_results = {}
        for result in ocr_results:
            source_type = None
            for mapping in region_mappings:
                if mapping['region_id'] == result.region_id:
                    source_type = mapping['source_type']
                    break
                    
            if source_type not in grouped_results:
                grouped_results[source_type] = []
                
            grouped_results[source_type].append({
                'region_id': result.region_id,
                'text': result.text_content,
                'confidence': result.confidence,
                'bbox': next(m['original_bbox'] for m in region_mappings 
                           if m['region_id'] == result.region_id)
            })
        
        # 性能统计
        stats = {
            'total_regions': len(text_regions),
            'total_time_ms': total_time,
            'ocr_calls': 1,  # 关键指标：只调用1次OCR
            'avg_time_per_region': total_time / len(text_regions),
            'regions_per_second': len(text_regions) / (total_time / 1000),
            'batch_image_size': batch_image.shape if batch_image.size > 0 else None
        }
        
        print(f"\n🎯 批量处理完成:")
        print(f"   ✅ OCR调用次数: {stats['ocr_calls']} 次（关键优势！）")
        print(f"   ✅ 处理区域数: {stats['total_regions']}")
        print(f"   ✅ 总耗时: {stats['total_time_ms']:.1f}ms")  
        print(f"   ✅ 处理速度: {stats['regions_per_second']:.1f} 区域/秒")
        
        return {
            'success': True,
            'results': grouped_results,
            'stats': stats,
            'batch_image': batch_image,
            'region_mappings': region_mappings
        }
        
    def save_batch_visualization(self, batch_result: Dict, output_dir: str = "batch_ocr_results"):
        """保存批处理可视化结果"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not batch_result['success']:
            return
            
        batch_image = batch_result['batch_image'] 
        region_mappings = batch_result['region_mappings']
        results = batch_result['results']
        
        # 保存批处理图像
        if batch_image.size > 0:
            cv2.imwrite(f"{output_dir}/batch_image.jpg", batch_image)
            
            # 在批处理图像上标注区域信息
            annotated_batch = batch_image.copy()
            
            for mapping in region_mappings:
                bbox = mapping['batch_bbox']
                x1, y1, x2, y2 = bbox
                
                # 绘制区域边界
                cv2.rectangle(annotated_batch, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 标注区域ID
                cv2.putText(annotated_batch, mapping['region_id'], 
                          (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            cv2.imwrite(f"{output_dir}/annotated_batch.jpg", annotated_batch)
            
        # 保存文字识别结果
        with open(f"{output_dir}/batch_ocr_results.txt", "w", encoding="utf-8") as f:
            f.write("批量OCR识别结果\n")
            f.write("="*50 + "\n\n")
            
            stats = batch_result['stats']
            f.write("性能统计:\n")
            f.write(f"- OCR调用次数: {stats['ocr_calls']} 次\n")
            f.write(f"- 处理区域数: {stats['total_regions']}\n")
            f.write(f"- 总耗时: {stats['total_time_ms']:.1f}ms\n")
            f.write(f"- 平均耗时: {stats['avg_time_per_region']:.1f}ms/区域\n")
            f.write(f"- 处理速度: {stats['regions_per_second']:.1f} 区域/秒\n\n")
            
            f.write("识别结果:\n")
            f.write("-" * 30 + "\n")
            
            for source_type, regions in results.items():
                f.write(f"\n{source_type}:\n")
                for region in regions:
                    f.write(f"  {region['region_id']}: {region['text']} "
                           f"(置信度:{region['confidence']:.3f})\n")
        
        print(f"💾 批处理结果已保存到: {output_dir}/")
        
    def get_performance_summary(self) -> Dict:
        """获取性能统计摘要"""
        
        if self.total_regions_processed == 0:
            return {'message': '尚未处理任何区域'}
            
        return {
            'total_regions_processed': self.total_regions_processed,
            'total_processing_time_ms': self.total_processing_time,
            'average_time_per_region': self.total_processing_time / self.total_regions_processed,
            'estimated_speedup_vs_individual': self.total_regions_processed,  # 理论加速倍数
            'regions_per_second': self.total_regions_processed / (self.total_processing_time / 1000)
        }


def demo_batch_ocr_processing():
    """演示批量OCR处理效果"""
    
    print("🎭 批量OCR处理演示")
    print("="*60)
    
    # 导入依赖
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from fast_text_line_splitter import ProjectionLineSplitter
    
    # 初始化处理器
    batch_processor = BatchOCRProcessor(max_batch_size=32, target_height=32)
    line_splitter = ProjectionLineSplitter()
    
    # 模拟IM图像和YOLO检测结果
    # 这里可以使用真实的WeChat截图
    test_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # 模拟多个检测区域
    mock_detections = [
        {
            'class': 'chat_area',
            'bbox': [50, 100, 750, 300],
            'confidence': 0.95
        },
        {
            'class': 'input_area', 
            'bbox': [50, 350, 750, 450],
            'confidence': 0.90
        },
        {
            'class': 'side_info',
            'bbox': [50, 470, 750, 550],
            'confidence': 0.85
        }
    ]
    
    # 在测试图像上绘制一些模拟文字区域
    for detection in mock_detections:
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(test_image, (x1, y1), (x2, y2), (200, 200, 200), -1)
        
        # 添加模拟文字行
        for i in range(3):  # 每个区域3行文字
            line_y = y1 + 20 + i * 40
            cv2.rectangle(test_image, (x1+10, line_y), (x2-10, line_y+25), (100, 100, 100), -1)
    
    print(f"📸 测试图像准备完成: {test_image.shape}")
    print(f"🎯 模拟YOLO检测: {len(mock_detections)} 个区域")
    
    # 执行批量处理
    batch_result = batch_processor.process_im_image_batch(
        test_image, mock_detections, line_splitter
    )
    
    # 保存可视化结果
    batch_processor.save_batch_visualization(batch_result)
    
    # 显示性能统计
    performance = batch_processor.get_performance_summary()
    print(f"\n📊 总体性能统计:")
    print(f"   处理区域总数: {performance['total_regions_processed']}")
    print(f"   总耗时: {performance['total_processing_time_ms']:.1f}ms")
    print(f"   平均耗时: {performance['average_time_per_region']:.1f}ms/区域")
    print(f"   相比单独调用加速: {performance['estimated_speedup_vs_individual']:.1f}x")
    print(f"   处理速度: {performance['regions_per_second']:.1f} 区域/秒")
    
    print(f"\n🎉 批量OCR处理演示完成！")
    print(f"💡 核心优势：{batch_result['stats']['ocr_calls']}次模型调用 vs {batch_result['stats']['total_regions']}次单独调用")

if __name__ == "__main__":
    demo_batch_ocr_processing()
