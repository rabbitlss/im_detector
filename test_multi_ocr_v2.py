# -*- coding: utf-8 -*-
"""
IntelligentMultilineOCR 使用示例和测试
展示完整的多行文字识别流程
"""

import cv2
import numpy as np
import time
from ultrafast_ocr import UltraFastOCR
from ultrafast_ocr.intelligent_multiline_ocr import IntelligentMultilineOCR


def create_test_images():
    """创建不同类型的测试图片"""
    test_cases = {}
    
    # 测试1: 标准多行文本
    img1 = np.ones((300, 600, 3), dtype=np.uint8) * 255
    texts1 = [
        "第一行：这是测试文字内容",
        "第二行：Hello World Test",
        "第三行：OCR识别测试123",
        "第四行：多行文字智能识别"
    ]
    y = 50
    for text in texts1:
        cv2.putText(img1, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        y += 60
    test_cases["标准多行"] = img1
    
    # 测试2: 不同字体大小
    img2 = np.ones((250, 700, 3), dtype=np.uint8) * 255
    cv2.putText(img2, "大标题文字", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(img2, "中等字体内容测试", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img2, "小字体说明文字", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img2, "更小的备注信息", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    test_cases["不同字体大小"] = img2
    
    # 测试3: 密集文本
    img3 = np.ones((400, 500, 3), dtype=np.uint8) * 255
    dense_texts = [
        "密集文本行1",
        "密集文本行2", 
        "密集文本行3",
        "密集文本行4",
        "密集文本行5",
        "密集文本行6",
        "密集文本行7",
        "密集文本行8"
    ]
    y = 30
    for text in dense_texts:
        cv2.putText(img3, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y += 45
    test_cases["密集文本"] = img3
    
    # 测试4: 稀疏文本（行间距大）
    img4 = np.ones((400, 600, 3), dtype=np.uint8) * 255
    sparse_texts = ["第一行文字", "第二行文字", "第三行文字"]
    y = 80
    for text in sparse_texts:
        cv2.putText(img4, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        y += 120
    test_cases["稀疏文本"] = img4
    
    return test_cases


def demo_basic_usage():
    """基础使用示例"""
    print("="*70)
    print("基础使用示例")
    print("="*70)
    
    # 1. 初始化OCR引擎
    print("1. 初始化OCR引擎...")
    ocr_engine = UltraFastOCR(use_gpu=True)  # 可以设置use_gpu=False
    
    # 2. 初始化智能多行OCR
    intelligent_ocr = IntelligentMultilineOCR(
        ocr_engine=ocr_engine,
        max_concat_width=1280,  # 最大拼接宽度
        target_height=48        # OCR模型目标高度
    )
    
    # 3. 创建测试图片
    test_img = create_test_images()["标准多行"]
    
    # 4. 执行识别
    print("\n2. 执行多行识别...")
    start_time = time.time()
    results = intelligent_ocr.recognize_multiline(test_img)
    end_time = time.time()
    
    # 5. 显示结果
    print(f"\n识别完成，耗时: {(end_time - start_time)*1000:.1f}ms")
    print(f"识别到 {len(results)} 行文字:")
    for i, text in enumerate(results, 1):
        print(f"  行{i}: {text}")


def demo_detailed_analysis():
    """详细分析示例"""
    print("\n" + "="*70)
    print("详细分析示例")
    print("="*70)
    
    # 初始化
    ocr_engine = UltraFastOCR(use_gpu=True)
    intelligent_ocr = IntelligentMultilineOCR(ocr_engine, max_concat_width=1280)
    
    # 使用不同类型的测试图片
    test_images = create_test_images()
    
    for name, image in test_images.items():
        print(f"\n{'='*50}")
        print(f"测试图片: {name}")
        print(f"{'='*50}")
        
        # 步骤1: 文字结构分析
        print("1. 文字结构分析...")
        structure_info = intelligent_ocr.analyze_text_structure(image)
        print(f"   检测到字符高度: {structure_info['char_height']}px")
        print(f"   估算行高: {structure_info['line_height']}px")
        print(f"   估算行间距: {structure_info['line_spacing']}px")
        print(f"   检测到字符数: {structure_info['num_chars']}")
        
        # 步骤2: 文本行检测
        print("\n2. 文本行检测...")
        lines = intelligent_ocr.detect_text_lines(image, structure_info)
        print(f"   检测到行数: {len(lines)}")
        for i, line in enumerate(lines):
            print(f"   行{i+1}: 高度={line.bbox[3]}px, 估算字符数={line.estimated_chars}, 置信度={line.confidence:.2f}")
        
        # 步骤3: 拼接优化
        print("\n3. 拼接优化...")
        concat_groups = intelligent_ocr.optimize_concatenation(lines)
        print(f"   拼接成 {len(concat_groups)} 组")
        for i, (_, indices) in enumerate(concat_groups):
            print(f"   组{i+1}: 包含行 {[idx+1 for idx in indices]}")
        
        # 步骤4: OCR识别
        print("\n4. OCR识别结果:")
        results = intelligent_ocr.recognize_multiline(image)
        for i, text in enumerate(results, 1):
            print(f"   行{i}: '{text}'")
        
        # 性能统计
        print("\n5. 性能统计:")
        stats = intelligent_ocr.get_performance_stats(image)
        print(f"   总耗时: {stats['total_time_ms']:.1f}ms")
        print(f"   - 结构分析: {stats['analysis_time_ms']:.1f}ms")
        print(f"   - 行检测: {stats['detection_time_ms']:.1f}ms") 
        print(f"   - 拼接优化: {stats['concat_time_ms']:.1f}ms")
        print(f"   - OCR识别: {stats['ocr_time_ms']:.1f}ms")
        print(f"   OCR调用次数: {stats['ocr_calls']} (原本需要{stats['detected_lines']}次)")
        print(f"   效率提升: {stats['efficiency_ratio']:.1f}x")


def demo_performance_comparison():
    """性能对比示例"""
    print("\n" + "="*70)
    print("性能对比示例")
    print("="*70)
    
    # 初始化
    ocr_engine = UltraFastOCR(use_gpu=True)
    intelligent_ocr = IntelligentMultilineOCR(ocr_engine, max_concat_width=1280)
    
    # 使用密集文本进行测试
    test_img = create_test_images()["密集文本"]
    
    print("方法1: 传统逐行识别")
    # 模拟传统方法：先切割，然后逐行识别
    start_time = time.time()
    structure_info = intelligent_ocr.analyze_text_structure(test_img)
    lines = intelligent_ocr.detect_text_lines(test_img, structure_info)
    
    traditional_results = []
    for line in lines:
        text = ocr_engine.recognize_single_line(line.image)
        traditional_results.append(text)
    traditional_time = (time.time() - start_time) * 1000
    
    print(f"   耗时: {traditional_time:.1f}ms")
    print(f"   OCR调用次数: {len(lines)}")
    
    print("\n方法2: 智能拼接识别")
    start_time = time.time()
    intelligent_results = intelligent_ocr.recognize_multiline(test_img)
    intelligent_time = (time.time() - start_time) * 1000
    
    concat_groups = intelligent_ocr.optimize_concatenation(lines)
    print(f"   耗时: {intelligent_time:.1f}ms")
    print(f"   OCR调用次数: {len(concat_groups)}")
    
    print(f"\n性能提升:")
    print(f"   加速比: {traditional_time/intelligent_time:.1f}x")
    print(f"   OCR调用减少: {len(lines) - len(concat_groups)} 次")
    
    # 结果准确性对比
    print(f"\n结果对比:")
    print(f"   传统方法识别行数: {len([r for r in traditional_results if r])}")
    print(f"   智能方法识别行数: {len(intelligent_results)}")


def demo_real_world_usage():
    """实际应用示例"""
    print("\n" + "="*70)
    print("实际应用示例")
    print("="*70)
    
    # 初始化
    ocr_engine = UltraFastOCR(use_gpu=True)
    
    # 不同配置的智能OCR
    configs = [
        ("速度优先", {"max_concat_width": 640, "target_height": 32}),
        ("平衡模式", {"max_concat_width": 1280, "target_height": 48}),
        ("质量优先", {"max_concat_width": 1920, "target_height": 64})
    ]
    
    test_img = create_test_images()["标准多行"]
    
    for config_name, config in configs:
        print(f"\n{config_name} 配置:")
        intelligent_ocr = IntelligentMultilineOCR(ocr_engine, **config)
        
        start_time = time.time()
        results = intelligent_ocr.recognize_multiline(test_img)
        end_time = time.time()
        
        print(f"   耗时: {(end_time - start_time)*1000:.1f}ms")
        print(f"   识别行数: {len(results)}")
        print(f"   配置参数: {config}")


def save_debug_images():
    """保存调试图片"""
    print("\n" + "="*70)
    print("保存调试图片")
    print("="*70)
    
    test_images = create_test_images()
    for name, image in test_images.items():
        filename = f"test_{name.replace(' ', '_')}.jpg"
        cv2.imwrite(filename, image)
        print(f"保存测试图片: {filename}")


if __name__ == "__main__":
    print("IntelligentMultilineOCR 完整使用示例")
    print("="*80)
    
    try:
        # 1. 基础使用示例
        demo_basic_usage()
        
        # 2. 详细分析示例
        demo_detailed_analysis()
        
        # 3. 性能对比示例
        demo_performance_comparison()
        
        # 4. 实际应用示例
        demo_real_world_usage()
        
        # 5. 保存调试图片
        save_debug_images()
        
        print("\n" + "="*80)
        print("✅ 所有示例运行完成！")
        print("\n📝 使用总结:")
        print("1. 基础用法: intelligent_ocr.recognize_multiline(image)")
        print("2. 结构分析: intelligent_ocr.analyze_text_structure(image)")
        print("3. 性能统计: intelligent_ocr.get_performance_stats(image)")
        print("4. 配置调优: 调整max_concat_width和target_height参数")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("\n可能的原因:")
        print("1. UltraFastOCR未正确初始化")
        print("2. 缺少必要的依赖包")
        print("3. 模型文件未下载")
        print("\n解决方案:")
        print("1. 确保已下载OCR模型文件")
        print("2. 检查GPU/CPU设置")
        print("3. 安装所需依赖: opencv-python, numpy")
