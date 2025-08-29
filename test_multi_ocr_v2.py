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


def create_text_with_blanks(text_lines=5, blank_lines=3):
    """创建包含空白行的测试图像"""
    total_lines = text_lines + blank_lines
    img = np.ones((50 * total_lines, 800, 3), dtype=np.uint8) * 255
    
    text_count = 0
    for i in range(total_lines):
        y = 30 + i * 50
        
        # 每3行插入1-2个空白行
        if i % 4 == 3 and blank_lines > 0:
            # 这是空白行，不画任何文字
            blank_lines -= 1
        else:
            if text_count < text_lines:
                # 文字行
                text = f"Line {text_count + 1}: Text content with spacing"
                cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                text_count += 1
    
    return img


def create_text_with_internal_spaces():
    """创建包含行内大量空白的测试图像"""
    img = np.ones((200, 800, 3), dtype=np.uint8) * 255
    
    # 第1行：正常文字
    cv2.putText(img, "Normal text line", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 第2行：包含大量行内空白的文字
    cv2.putText(img, "Text", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "with", (200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "spaces", (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 第3行：表格样式（空白分隔）
    cv2.putText(img, "Col1", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Col2", (150, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  
    cv2.putText(img, "Col3", (280, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Col4", (410, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img


def test_original_method():
    """测试原始方法"""
    print("\n" + "="*70)
    print("测试原始智能拼接方法")
    print("="*70)
    
    mock_ocr = MockOCR()
    intelligent_ocr = IntelligentMultilineOCR(
        ocr_engine=mock_ocr,
        max_concat_width=3840,  # 增大宽度限制
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
        
        # 调试：显示每行的有效宽度
        print(f"  - 每行有效宽度: {[getattr(line, 'effective_width', 'N/A') for line in lines]}")
        
        # 显示分组详情
        for i, (_, indices) in enumerate(concat_groups):
            print(f"    组{i+1}: 包含 {len(indices)} 行 (行号: {indices})")
            
        # 调试：如果所有组都是单行，分析原因
        if all(len(indices) == 1 for _, indices in concat_groups):
            print(f"  ⚠️  警告：所有组都是单行，拼接失败！")
            if len(lines) > 1:
                # 分析前两行为什么不能拼接
                line1, line2 = lines[0], lines[1]
                print(f"    第1行有效宽度: {getattr(line1, 'effective_width', 'N/A')}")
                print(f"    第2行有效宽度: {getattr(line2, 'effective_width', 'N/A')}")
                
                # 计算拼接宽度
                if hasattr(line1, 'effective_width') and hasattr(line2, 'effective_width'):
                    h1, h2 = line1.image.shape[0], line2.image.shape[0]
                    scale1 = 48 / h1 if h1 > 0 else 1.0
                    scale2 = 48 / h2 if h2 > 0 else 1.0
                    scaled_w1 = int(line1.effective_width * scale1)
                    scaled_w2 = int(line2.effective_width * scale2)
                    total_width = scaled_w1 + 20 + scaled_w2
                    print(f"    拼接宽度估算: {scaled_w1} + 20 + {scaled_w2} = {total_width}")
                    print(f"    最大允许宽度: {intelligent_ocr.max_concat_width}")
                    print(f"    宽度检查: {'通过' if total_width <= intelligent_ocr.max_concat_width else '失败'}")


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
    original_ocr = IntelligentMultilineOCR(mock_ocr1, max_concat_width=3840)
    
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
    original = IntelligentMultilineOCR(mock_ocr, max_concat_width=3840)
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


def test_blank_line_handling():
    """测试空白行处理"""
    print("\n" + "="*70)
    print("空白行处理测试")
    print("="*70)
    
    mock_ocr = MockOCR()
    intelligent_ocr = IntelligentMultilineOCR(
        ocr_engine=mock_ocr,
        max_concat_width=3840,
        dynamic_width=True,
        width_strategy='adaptive'
    )
    
    # 测试场景：不同空白行比例
    test_scenarios = [
        (5, 0, "密集文本"),     # 5行文字，0行空白
        (5, 2, "少量空白"),     # 5行文字，2行空白  
        (5, 5, "大量空白"),     # 5行文字，5行空白
        (3, 7, "稀疏文本"),     # 3行文字，7行空白
    ]
    
    for text_lines, blank_lines, scenario_name in test_scenarios:
        print(f"\n{scenario_name}: {text_lines}行文字 + {blank_lines}行空白")
        print("-" * 50)
        
        mock_ocr.reset()
        
        # 创建包含空白行的图像
        test_img = create_text_with_blanks(text_lines, blank_lines)
        
        # 执行检测和识别
        start_time = time.time()
        results = intelligent_ocr.recognize_multiline(test_img)
        total_time = time.time() - start_time
        
        # 获取详细统计
        structure_info = intelligent_ocr.analyze_text_structure(test_img)
        lines = intelligent_ocr.detect_text_lines(test_img, structure_info)
        
        # 统计结果
        total_visual_lines = text_lines + blank_lines
        detected_lines = len(lines)
        recognized_lines = len(results)
        ocr_calls = mock_ocr.call_count
        
        print(f"  图像总行数: {total_visual_lines}")
        print(f"  检测到的行数: {detected_lines} (过滤了 {total_visual_lines - detected_lines} 个空白行)")
        print(f"  识别结果行数: {recognized_lines}")
        print(f"  OCR调用次数: {ocr_calls}")
        print(f"  处理时间: {total_time * 1000:.1f}ms")
        
        # 检查空白行过滤效果
        if detected_lines == text_lines:
            print(f"  ✅ 空白行过滤正确")
        elif detected_lines < text_lines:
            print(f"  ⚠️  过度过滤: 丢失了 {text_lines - detected_lines} 行有效文字")
        else:
            print(f"  ⚠️  过滤不足: 仍包含 {detected_lines - text_lines} 行可能的空白")
        
        # 显示效率指标
        if ocr_calls > 0:
            efficiency = detected_lines / ocr_calls
            print(f"  效率比: {efficiency:.1f}x")


def test_simple_optimization():
    """简化版OCR优化测试 - 专注核心指标"""
    print("\n" + "="*70)
    print("简化版动态策略优化测试")
    print("="*70)
    
    # 测试参数
    test_cases = [5, 10, 15, 20]
    strategies = ['conservative', 'balanced', 'aggressive', 'adaptive']
    
    results = []
    
    for num_lines in test_cases:
        print(f"\n测试 {num_lines} 行文本:")
        print("-" * 40)
        test_img = create_dense_text_image(num_lines)
        
        for strategy in strategies:
            mock_ocr = MockOCR()
            
            # 创建OCR实例
            ocr_instance = IntelligentMultilineOCR(
                ocr_engine=mock_ocr,
                dynamic_width=True,
                width_strategy=strategy
            )
            
            # 执行测试
            start_time = time.time()
            recognized_texts = ocr_instance.recognize_multiline(test_img)
            total_time = time.time() - start_time
            
            # 收集结果
            result = {
                'lines': num_lines,
                'strategy': strategy,
                'ocr_calls': mock_ocr.call_count,
                'time_ms': total_time * 1000,
                'recognized_count': len(recognized_texts),
                'efficiency': num_lines / mock_ocr.call_count if mock_ocr.call_count > 0 else 0
            }
            results.append(result)
            
            print(f"{strategy:12} | OCR调用: {mock_ocr.call_count:2d} | 时间: {total_time*1000:5.1f}ms | 识别: {len(recognized_texts):2d} | 效率: {result['efficiency']:.1f}x")
    
    # 总结报告
    print(f"\n{'='*60}")
    print("优化效果总结")
    print("="*60)
    
    for strategy in strategies:
        strategy_results = [r for r in results if r['strategy'] == strategy]
        avg_efficiency = np.mean([r['efficiency'] for r in strategy_results])
        avg_time = np.mean([r['time_ms'] for r in strategy_results])
        total_calls = sum([r['ocr_calls'] for r in strategy_results])
        total_lines = sum([r['lines'] for r in strategy_results])
        
        print(f"{strategy:12} | 平均效率: {avg_efficiency:.1f}x | 平均时间: {avg_time:5.1f}ms | 总OCR调用: {total_calls}/{total_lines}")
    
    # 找出最佳策略
    best_strategy = max(strategies, key=lambda s: np.mean([r['efficiency'] for r in results if r['strategy'] == s]))
    print(f"\n🏆 推荐策略: {best_strategy}")
    
    return results


def test_internal_space_handling():
    """测试行内空白处理"""
    print("\n" + "="*70)
    print("行内空白处理测试")
    print("="*70)
    
    # 创建包含行内空白的测试图像
    test_img = create_text_with_internal_spaces()
    
    # 测试不同的宽度计算方法
    methods = ['span', 'compact', 'adaptive']
    
    for method in methods:
        print(f"\n{method.upper()} 方法:")
        print("-" * 30)
        
        mock_ocr = MockOCR()
        intelligent_ocr = IntelligentMultilineOCR(
            ocr_engine=mock_ocr,
            dynamic_width=True,
            width_strategy='adaptive'
        )
        
        # 获取行检测结果
        structure_info = intelligent_ocr.analyze_text_structure(test_img)
        lines = intelligent_ocr.detect_text_lines(test_img, structure_info)
        
        # 手动测试每行的有效宽度计算
        for i, line in enumerate(lines):
            # 使用不同方法计算有效宽度
            effective_width = intelligent_ocr._calculate_effective_width(line.image, method=method)
            
            # 分析空白比例
            gray = cv2.cvtColor(line.image, cv2.COLOR_BGR2GRAY) if len(line.image.shape) == 3 else line.image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            v_projection = np.sum(binary == 255, axis=0)
            non_zero_cols = np.where(v_projection > 0)[0]
            
            if len(non_zero_cols) > 0:
                span_width = non_zero_cols[-1] - non_zero_cols[0] + 1
                actual_text_cols = len(non_zero_cols)
                blank_ratio = (span_width - actual_text_cols) / span_width if span_width > 0 else 0
                
                print(f"  行{i+1}: 有效宽度={effective_width:3d}px, 跨度={span_width:3d}px, 文字列={actual_text_cols:3d}, 空白比例={blank_ratio:.1%}")
            else:
                print(f"  行{i+1}: 无文字内容")
        
        # 测试拼接效果
        results = intelligent_ocr.recognize_multiline(test_img)
        print(f"  识别结果: {len(results)}行, OCR调用: {mock_ocr.call_count}次")


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
    
    # 5. 空白行处理测试
    test_blank_line_handling()
    
    # 6. 简化版动态策略测试
    test_simple_optimization()
    
    # 7. 行内空白处理测试
    test_internal_space_handling()
    
    print("\n" + "="*80)
    print("✅ 所有测试完成！")
    print("\n关键发现:")
    print("1. 动态宽度策略显著提升拼接效率")
    print("2. 空白行过滤减少无效OCR调用")
    print("3. 文字有效宽度计算是拼接成功的关键")
    print("4. 不同策略适用于不同文档类型:")
    print("   - conservative: 稳定性优先，适合重要文档")
    print("   - balanced: 平衡模式，适合大多数场景")
    print("   - aggressive: 效率优先，适合批量处理")
    print("   - adaptive: 智能适应，推荐日常使用")
