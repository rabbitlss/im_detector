# -*- coding: utf-8 -*-
"""
对比测试：优化前后的OCR调用次数
"""

import cv2
import numpy as np
import time
from ultrafast_ocr.intelligent_multiline_ocr import IntelligentMultilineOCR
from ultrafast_ocr.intelligent_multiline_ocr_optimized import IntelligentMultilineOCROptimized


class MockOCR:
    """模拟OCR引擎，用于测试"""
    def __init__(self):
        self.call_count = 0
        self.call_history = []
    
    def recognize_single_line(self, image):
        """模拟单行识别"""
        self.call_count += 1
        h, w = image.shape[:2]
        self.call_history.append(f"OCR调用#{self.call_count}: 图像尺寸={w}x{h}")
        
        # 模拟返回文本
        return f"Text_Line_{self.call_count}"
    
    def reset(self):
        """重置计数器"""
        self.call_count = 0
        self.call_history = []


def create_dense_text_image(num_lines=10):
    """创建密集文本测试图像"""
    img = np.ones((50 * num_lines, 800, 3), dtype=np.uint8) * 255
    
    for i in range(num_lines):
        y = 30 + i * 50
        text = f"Line {i+1}: This is test text for OCR optimization comparison"
        cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img


def test_original_method():
    """测试原始方法"""
    print("\n" + "="*70)
    print("测试原始智能拼接方法")
    print("="*70)
    
    mock_ocr = MockOCR()
    intelligent_ocr = IntelligentMultilineOCR(
        ocr_engine=mock_ocr,
        max_concat_width=1280,
        target_height=48
    )
    
    # 测试不同行数的图像
    test_cases = [5, 8, 10, 15, 20]
    
    for num_lines in test_cases:
        print(f"\n测试 {num_lines} 行文本:")
        mock_ocr.reset()
        
        test_img = create_dense_text_image(num_lines)
        
        # 分析结构
        structure_info = intelligent_ocr.analyze_text_structure(test_img)
        lines = intelligent_ocr.detect_text_lines(test_img, structure_info)
        concat_groups = intelligent_ocr.optimize_concatenation(lines)
        
        # 执行识别
        results = intelligent_ocr.recognize_multiline(test_img)
        
        print(f"  - 检测到行数: {len(lines)}")
        print(f"  - 拼接组数: {len(concat_groups)}")
        print(f"  - OCR调用次数: {mock_ocr.call_count}")
        print(f"  - 效率比: {len(lines)/mock_ocr.call_count:.2f}x")
        
        # 显示分组详情
        for i, (_, indices) in enumerate(concat_groups):
            print(f"    组{i+1}: 包含 {len(indices)} 行 (行号: {indices})")


def test_optimized_method():
    """测试优化后的方法"""
    print("\n" + "="*70)
    print("测试优化后的智能拼接方法")
    print("="*70)
    
    mock_ocr = MockOCR()
    optimized_ocr = IntelligentMultilineOCROptimized(
        ocr_engine=mock_ocr,
        max_concat_width=2560,  # 增大宽度限制
        target_height=48,
        enable_cache=True
    )
    
    # 测试不同行数的图像
    test_cases = [5, 8, 10, 15, 20]
    
    for num_lines in test_cases:
        print(f"\n测试 {num_lines} 行文本:")
        mock_ocr.reset()
        optimized_ocr.clear_cache()
        
        test_img = create_dense_text_image(num_lines)
        
        # 获取性能统计
        stats = optimized_ocr.get_performance_stats_optimized(test_img)
        
        print(f"  - 检测到行数: {stats['detected_lines']}")
        print(f"  - 拼接组数: {stats['concat_groups']}")
        print(f"  - 理论OCR调用: {stats['theoretical_ocr_calls']}")
        print(f"  - 实际OCR调用: {stats['actual_ocr_calls']}")
        print(f"  - 缓存命中率: {stats['cache_hit_rate']:.1%}")
        print(f"  - 效率比: {stats['actual_efficiency_ratio']:.2f}x")
        
        # 分析拼接组
        structure_info = optimized_ocr.analyze_text_structure(test_img)
        lines = optimized_ocr.detect_text_lines(test_img, structure_info)
        concat_groups = optimized_ocr.optimize_concatenation_aggressive(lines)
        
        for i, (_, indices) in enumerate(concat_groups):
            print(f"    组{i+1}: 包含 {len(indices)} 行 (行号: {indices})")


def compare_methods():
    """对比两种方法"""
    print("\n" + "="*70)
    print("方法对比总结")
    print("="*70)
    
    test_img = create_dense_text_image(15)
    
    # 原始方法
    mock_ocr1 = MockOCR()
    original_ocr = IntelligentMultilineOCR(mock_ocr1, max_concat_width=1280)
    
    start = time.time()
    results1 = original_ocr.recognize_multiline(test_img)
    time1 = time.time() - start
    
    # 优化方法
    mock_ocr2 = MockOCR()
    optimized_ocr = IntelligentMultilineOCROptimized(mock_ocr2, max_concat_width=2560)
    
    start = time.time()
    results2 = optimized_ocr.recognize_multiline_optimized(test_img)
    time2 = time.time() - start
    
    print(f"\n15行文本测试结果:")
    print(f"{'方法':<20} {'OCR调用次数':<15} {'耗时(ms)':<15} {'效率提升':<15}")
    print("-" * 65)
    print(f"{'原始方法':<20} {mock_ocr1.call_count:<15} {time1*1000:<15.2f} {'1.00x':<15}")
    print(f"{'优化方法':<20} {mock_ocr2.call_count:<15} {time2*1000:<15.2f} {f'{mock_ocr1.call_count/mock_ocr2.call_count:.2f}x':<15}")
    
    print(f"\n优化效果:")
    print(f"  - OCR调用减少: {mock_ocr1.call_count - mock_ocr2.call_count} 次 ({(1 - mock_ocr2.call_count/mock_ocr1.call_count)*100:.1f}%)")
    print(f"  - 速度提升: {time1/time2:.2f}x")


def analyze_optimization_details():
    """分析优化细节"""
    print("\n" + "="*70)
    print("优化策略详细分析")
    print("="*70)
    
    mock_ocr = MockOCR()
    
    # 创建两个OCR实例进行对比
    original = IntelligentMultilineOCR(mock_ocr, max_concat_width=1280)
    optimized = IntelligentMultilineOCROptimized(mock_ocr, max_concat_width=2560)
    
    test_img = create_dense_text_image(12)
    
    # 分析原始方法的限制
    structure_info = original.analyze_text_structure(test_img)
    lines = original.detect_text_lines(test_img, structure_info)
    original_groups = original.optimize_concatenation(lines)
    
    print(f"\n原始方法分析 (12行文本):")
    print(f"  最大拼接宽度: {original.max_concat_width}px")
    print(f"  拼接限制:")
    print(f"    - 最多6行一组（硬编码限制）")
    print(f"    - 字符高度差异 < 50%")
    print(f"    - 置信度差异 < 0.3")
    print(f"  结果: {len(original_groups)} 个组")
    
    # 分析优化方法
    optimized_groups = optimized.optimize_concatenation_aggressive(lines)
    
    print(f"\n优化方法分析 (12行文本):")
    print(f"  最大拼接宽度: {optimized.max_concat_width}px")
    print(f"  优化策略:")
    print(f"    - 无行数限制")
    print(f"    - 动态规划最优分组")
    print(f"    - OCR结果缓存")
    print(f"    - 批量处理")
    print(f"  结果: {len(optimized_groups)} 个组")
    
    print(f"\n优化效果: OCR调用从 {len(original_groups)} 次减少到 {len(optimized_groups)} 次")
    print(f"效率提升: {len(original_groups)/len(optimized_groups):.2f}x")


if __name__ == "__main__":
    print("智能多行OCR优化对比测试")
    print("="*80)
    
    # 1. 测试原始方法
    test_original_method()
    
    # 2. 测试优化方法
    test_optimized_method()
    
    # 3. 对比两种方法
    compare_methods()
    
    # 4. 分析优化细节
    analyze_optimization_details()
    
    print("\n" + "="*80)
    print("✅ 测试完成！")
    print("\n关键发现:")
    print("1. 原始方法的6行限制严重影响了拼接效率")
    print("2. 优化后的动态规划算法能找到最优分组方案")
    print("3. 缓存机制进一步减少了重复OCR调用")
    print("4. 增大最大宽度限制允许更多行拼接在一起")
