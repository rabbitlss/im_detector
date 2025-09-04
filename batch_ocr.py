# -*- coding: utf-8 -*-
"""
灵活的批量OCR处理器 - 支持不确定个数的图片列表
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Union, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class OCRInput:
    """OCR输入数据"""
    source_id: str  # 唯一标识
    image: np.ndarray  # 图像数据
    metadata: Dict = None  # 可选元数据

class FlexibleBatchOCRProcessor:
    """灵活的批量OCR处理器"""
    
    def __init__(self, 
                 max_batch_size: int = 32,
                 auto_split_large_batch: bool = True):
        """
        初始化
        
        Args:
            max_batch_size: 单批次最大处理数量
            auto_split_large_batch: 自动分割大批次
        """
        self.max_batch_size = max_batch_size
        self.auto_split_large_batch = auto_split_large_batch
        self.target_height = 48
        
    def process_dynamic_list(self, 
                            inputs: Union[List[np.ndarray], 
                                        List[str], 
                                        List[Dict],
                                        List[OCRInput]]) -> Dict:
        """
        处理不确定个数的输入列表
        
        支持的输入格式：
        1. 图像数组列表: [img1, img2, ...]
        2. 图像路径列表: ["path1.jpg", "path2.png", ...]
        3. 字典列表: [{'image': img1, 'id': '1'}, ...]
        4. OCRInput对象列表: [OCRInput(...), ...]
        
        Args:
            inputs: 各种格式的输入列表
            
        Returns:
            处理结果字典
        """
        
        print(f"📥 接收到 {len(inputs)} 个输入项")
        
        # 步骤1: 标准化输入
        ocr_inputs = self._standardize_inputs(inputs)
        
        if not ocr_inputs:
            return {
                'success': False,
                'message': '无有效输入',
                'results': []
            }
        
        print(f"✅ 标准化完成: {len(ocr_inputs)} 个有效输入")
        
        # 步骤2: 处理大批次
        if self.auto_split_large_batch and len(ocr_inputs) > self.max_batch_size:
            return self._process_large_batch(ocr_inputs)
        
        # 步骤3: 创建批处理图像
        batch_image, mappings = self._create_flexible_batch(ocr_inputs)
        
        # 步骤4: 执行OCR（单次调用）
        results = self._execute_batch_ocr(batch_image, mappings)
        
        return {
            'success': True,
            'total_inputs': len(ocr_inputs),
            'ocr_calls': 1,
            'results': results,
            'batch_info': {
                'batch_size': len(ocr_inputs),
                'image_shape': batch_image.shape
            }
        }
    
    def process_to_text_list(self,
                            inputs: Union[List[np.ndarray], List[str], List[Dict]]) -> List[str]:
        """
        处理图片列表，返回文字列表（保持顺序一一对应）
        
        输入: [img1, img2, img3, ...]
        输出: ["text1", "text2", "text3", ...]
        
        Args:
            inputs: 图片输入列表
            
        Returns:
            文字识别结果列表，与输入顺序一一对应
        """
        
        if not inputs:
            return []
        
        # 记录原始索引和有效输入的映射
        input_mapping = {}
        valid_inputs = []
        
        for i, item in enumerate(inputs):
            ocr_input = self._convert_to_ocr_input(item, i)
            if ocr_input:
                input_mapping[ocr_input.source_id] = i
                valid_inputs.append(ocr_input)
        
        # 初始化结果列表（保持原始长度）
        results = [""] * len(inputs)
        
        # 批处理识别
        if valid_inputs:
            # 分批处理
            for batch_start in range(0, len(valid_inputs), self.max_batch_size):
                batch_end = min(batch_start + self.max_batch_size, len(valid_inputs))
                batch = valid_inputs[batch_start:batch_end]
                
                # 创建批处理图像
                batch_image, mappings = self._create_flexible_batch(batch)
                
                # 执行OCR
                batch_results = self._execute_batch_ocr(batch_image, mappings)
                
                # 将结果放回对应位置
                for result in batch_results:
                    source_id = result['source_id']
                    if source_id in input_mapping:
                        original_idx = input_mapping[source_id]
                        results[original_idx] = result['text']
        
        return results
    
    def process_to_dict_list(self,
                            inputs: Union[List[np.ndarray], List[str], List[Dict]]) -> List[Dict]:
        """
        处理图片列表，返回包含文字和置信度的字典列表
        
        输入: [img1, img2, img3, ...]
        输出: [
            {"text": "text1", "confidence": 0.95},
            {"text": "text2", "confidence": 0.92},
            {"text": "text3", "confidence": 0.88}
        ]
        
        Args:
            inputs: 图片输入列表
            
        Returns:
            包含文字和置信度的字典列表
        """
        
        if not inputs:
            return []
        
        # 记录映射关系
        input_mapping = {}
        valid_inputs = []
        
        for i, item in enumerate(inputs):
            ocr_input = self._convert_to_ocr_input(item, i)
            if ocr_input:
                input_mapping[ocr_input.source_id] = i
                valid_inputs.append(ocr_input)
        
        # 初始化结果
        results = [{"text": "", "confidence": 0.0} for _ in range(len(inputs))]
        
        # 批处理
        if valid_inputs:
            for batch_start in range(0, len(valid_inputs), self.max_batch_size):
                batch_end = min(batch_start + self.max_batch_size, len(valid_inputs))
                batch = valid_inputs[batch_start:batch_end]
                
                batch_image, mappings = self._create_flexible_batch(batch)
                batch_results = self._execute_batch_ocr(batch_image, mappings)
                
                # 映射结果
                for result in batch_results:
                    source_id = result['source_id']
                    if source_id in input_mapping:
                        original_idx = input_mapping[source_id]
                        results[original_idx] = {
                            "text": result['text'],
                            "confidence": result['confidence']
                        }
        
        return results
    
    def _convert_to_ocr_input(self, item: Any, index: int) -> OCRInput:
        """将单个输入转换为OCRInput对象"""
        
        if isinstance(item, np.ndarray):
            return OCRInput(
                source_id=f"img_{index}",
                image=item
            )
        elif isinstance(item, str):
            if Path(item).exists():
                img = cv2.imread(item)
                if img is not None:
                    return OCRInput(
                        source_id=f"img_{index}",
                        image=img
                    )
        elif isinstance(item, dict) and 'image' in item:
            return OCRInput(
                source_id=f"img_{index}",
                image=item['image'],
                metadata=item
            )
        elif isinstance(item, OCRInput):
            item.source_id = f"img_{index}"  # 确保有索引信息
            return item
        
        return None
    
    def _standardize_inputs(self, inputs: Any) -> List[OCRInput]:
        """标准化各种输入格式"""
        
        standardized = []
        
        for i, item in enumerate(inputs):
            ocr_input = None
            
            # 处理不同输入类型
            if isinstance(item, np.ndarray):
                # 直接的图像数组
                ocr_input = OCRInput(
                    source_id=f"image_{i}",
                    image=item
                )
            
            elif isinstance(item, str):
                # 图像路径
                if Path(item).exists():
                    img = cv2.imread(item)
                    if img is not None:
                        ocr_input = OCRInput(
                            source_id=Path(item).stem,
                            image=img
                        )
            
            elif isinstance(item, dict):
                # 字典格式
                if 'image' in item:
                    ocr_input = OCRInput(
                        source_id=item.get('id', f"dict_{i}"),
                        image=item['image'],
                        metadata=item
                    )
            
            elif isinstance(item, OCRInput):
                # 已经是OCRInput对象
                ocr_input = item
            
            if ocr_input:
                standardized.append(ocr_input)
            else:
                print(f"⚠️ 跳过无效输入: 索引{i}")
        
        return standardized
    
    def _process_large_batch(self, ocr_inputs: List[OCRInput]) -> Dict:
        """处理超过最大批次大小的输入"""
        
        print(f"🔄 大批次分割: {len(ocr_inputs)} 个输入 → "
              f"{(len(ocr_inputs) + self.max_batch_size - 1) // self.max_batch_size} 个子批次")
        
        all_results = []
        ocr_call_count = 0
        
        # 分批处理
        for batch_idx in range(0, len(ocr_inputs), self.max_batch_size):
            batch = ocr_inputs[batch_idx:batch_idx + self.max_batch_size]
            
            print(f"\n处理子批次 {batch_idx // self.max_batch_size + 1}: "
                  f"{len(batch)} 个输入")
            
            # 创建批处理图像
            batch_image, mappings = self._create_flexible_batch(batch)
            
            # 执行OCR
            batch_results = self._execute_batch_ocr(batch_image, mappings)
            all_results.extend(batch_results)
            ocr_call_count += 1
        
        return {
            'success': True,
            'total_inputs': len(ocr_inputs),
            'ocr_calls': ocr_call_count,
            'results': all_results,
            'batch_info': {
                'sub_batches': ocr_call_count,
                'max_batch_size': self.max_batch_size
            }
        }
    
    def _create_flexible_batch(self, ocr_inputs: List[OCRInput]) -> tuple:
        """创建灵活的批处理图像"""
        
        normalized_images = []
        mappings = []
        current_x = 0
        
        for ocr_input in ocr_inputs:
            img = ocr_input.image
            
            # 确保是3通道
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # 调整到目标高度
            h, w = img.shape[:2]
            scale = self.target_height / h
            new_w = int(w * scale)
            
            resized = cv2.resize(img, (new_w, self.target_height))
            normalized_images.append(resized)
            
            # 记录映射
            mappings.append({
                'source_id': ocr_input.source_id,
                'batch_position': (current_x, 0, current_x + new_w, self.target_height),
                'original_size': (h, w),
                'metadata': ocr_input.metadata
            })
            
            current_x += new_w + 5  # 5px间隔
        
        # 拼接批处理图像
        if normalized_images:
            # 计算总宽度
            total_width = current_x - 5
            batch_image = np.ones((self.target_height, total_width, 3), 
                                 dtype=np.uint8) * 255
            
            # 放置每个图像
            x = 0
            for img in normalized_images:
                w = img.shape[1]
                batch_image[:, x:x+w] = img
                x += w + 5
            
            return batch_image, mappings
        
        return np.array([]), []
    
    def _execute_batch_ocr(self, batch_image: np.ndarray, 
                          mappings: List[Dict]) -> List[Dict]:
        """执行批量OCR识别"""
        
        if batch_image.size == 0:
            return []
        
        print(f"🔍 执行OCR识别: 批次图像 {batch_image.shape}")
        
        # 模拟OCR调用
        start_time = time.time()
        
        results = []
        for mapping in mappings:
            results.append({
                'source_id': mapping['source_id'],
                'text': f"识别文字_{mapping['source_id']}",
                'confidence': 0.9,
                'metadata': mapping.get('metadata')
            })
        
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"✅ OCR完成: {len(results)} 个结果, 耗时 {elapsed_ms:.1f}ms")
        
        return results
    
    def process_from_directory(self, directory_path: str, 
                              extensions: List[str] = None) -> Dict:
        """处理目录中的所有图像文件"""
        
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        dir_path = Path(directory_path)
        if not dir_path.exists():
            return {'success': False, 'message': f'目录不存在: {directory_path}'}
        
        # 收集所有图像文件
        image_files = []
        for ext in extensions:
            image_files.extend(dir_path.glob(f'*{ext}'))
            image_files.extend(dir_path.glob(f'*{ext.upper()}'))
        
        print(f"📂 从目录 {directory_path} 找到 {len(image_files)} 个图像文件")
        
        # 转换为路径字符串列表
        file_paths = [str(f) for f in image_files]
        
        return self.process_dynamic_list(file_paths)


def demo_flexible_processing():
    """演示灵活的批处理功能"""
    
    print("🎮 灵活批处理OCR演示")
    print("="*60)
    
    processor = FlexibleBatchOCRProcessor(max_batch_size=10)
    
    # 测试1: 列表输入输出（新功能）
    print("\n📌 测试1: 列表输入 → 列表输出（保持顺序）")
    test_images = [
        np.ones((30, 100, 3), dtype=np.uint8) * 100,
        np.ones((35, 120, 3), dtype=np.uint8) * 150,
        np.ones((40, 110, 3), dtype=np.uint8) * 200,
        np.ones((25, 90, 3), dtype=np.uint8) * 120,
        np.ones((45, 130, 3), dtype=np.uint8) * 180,
    ]
    
    # 获取文字列表
    text_results = processor.process_to_text_list(test_images)
    print(f"输入: {len(test_images)} 个图片")
    print(f"输出: {len(text_results)} 个文字")
    print("结果列表:", text_results[:3], "...")  # 显示前3个结果
    
    # 获取带置信度的结果
    dict_results = processor.process_to_dict_list(test_images[:3])
    print("\n带置信度的结果:")
    for i, result in enumerate(dict_results):
        print(f"  图片{i}: text='{result['text']}', confidence={result['confidence']:.2f}")
    
    # 测试2: 不确定个数的图像数组
    print("\n📌 测试2: 处理随机数量的图像")
    import random
    num_images = random.randint(5, 25)
    random_images = [
        np.ones((30 + i*2, 100, 3), dtype=np.uint8) * (100 + i*10)
        for i in range(num_images)
    ]
    
    random_results = processor.process_to_text_list(random_images)
    print(f"• 输入数量: {len(random_images)}")
    print(f"• 输出数量: {len(random_results)}")
    print(f"• 验证: 数量一致 = {len(random_images) == len(random_results)}")
    
    # 测试3: 包含无效输入
    print("\n📌 测试3: 处理包含无效输入的列表")
    mixed_inputs = [
        np.ones((30, 100, 3), dtype=np.uint8) * 100,  # 有效
        None,                                           # 无效
        np.ones((35, 120, 3), dtype=np.uint8) * 150,  # 有效
        "non_existent.jpg",                            # 无效
        np.ones((40, 110, 3), dtype=np.uint8) * 200,  # 有效
    ]
    
    # 处理混合输入
    mixed_results = processor.process_to_text_list(mixed_inputs)
    print("输入输出对应:")
    for i, (inp, res) in enumerate(zip(mixed_inputs, mixed_results)):
        inp_type = "有效图片" if isinstance(inp, np.ndarray) else "无效输入"
        print(f"  索引{i} ({inp_type}): '{res}'")
    
    # 测试4: 超大批次
    print("\n📌 测试4: 超大批次（自动分割）")
    large_batch = [
        np.ones((30, 100, 3), dtype=np.uint8) * 100
        for _ in range(75)
    ]
    
    large_results = processor.process_to_text_list(large_batch)
    print(f"• 输入数量: {len(large_batch)}")
    print(f"• 输出数量: {len(large_results)}")
    print(f"• 处理成功: {len(large_results) == 75}")
    
    print("\n🏆 总结:")
    print("• 支持列表输入 → 列表输出（保持顺序）")
    print("• 自动处理无效输入（返回空字符串）")
    print("• 支持任意数量（1到∞）")
    print("• 保持输入输出一一对应")

if __name__ == "__main__":
    demo_flexible_processing()
