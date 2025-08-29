# -*- coding: utf-8 -*-
"""
真实OCR性能测试 - 完整图片 vs 局部区域
支持自定义坐标区域测试
"""

import cv2
import numpy as np
import time
from ultrafast_ocr.intelligent_multiline_ocr import IntelligentMultilineOCR
from ultrafast_ocr import UltraFastOCR


class RealOCRWrapper:
    """真实OCR引擎包装器，记录详细调用信息"""
    def __init__(self, use_gpu=True):
        self.real_ocr = UltraFastOCR(use_gpu=use_gpu)
        self.call_count = 0
        self.call_details = []
        self.total_time = 0
    
    def recognize_single_line(self, image):
        """真实单行识别"""
        start_time = time.time()
        
        self.call_count += 1
        h, w = image.shape[:2]
        
        # 调用真实的OCR
        try:
            result = self.real_ocr.recognize_single_line(image)
            if not result or not result.strip():
                result = ""  # 空白结果
        except Exception as e:
            result = f"[Error: {str(e)[:30]}]"
            print(f"OCR错误 #{self.call_count}: {e}")
        
        actual_time = time.time() - start_time
        self.total_time += actual_time
        
        # 记录调用详情
        self.call_details.append({
            'call_id': self.call_count,
            'image_size': (w, h),
            'pixels': w * h,
            'time_ms': actual_time * 1000,
            'result_length': len(result) if result else 0,
            'is_empty': not bool(result.strip()) if result else True
        })
        
        return result
    
    def reset(self):
        """重置计数器"""
        self.call_count = 0
        self.call_details = []
        self.total_time = 0
    
    def get_stats(self):
        """获取统计信息"""
        if not self.call_details:
            return {}
        
        times = [call['time_ms'] for call in self.call_details]
        pixels = [call['pixels'] for call in self.call_details]
        empty_count = sum(1 for call in self.call_details if call['is_empty'])
        
        return {
            'total_calls': self.call_count,
            'total_time_ms': self.total_time * 1000,
            'avg_time_ms': np.mean(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'avg_pixels': np.mean(pixels),
            'total_pixels': sum(pixels),
            'empty_results': empty_count,
            'success_rate': (self.call_count - empty_count) / self.call_count if self.call_count > 0 else 0
        }


def create_test_document():
    """创建测试文档"""
    # 创建一个中等大小的测试文档
    img = np.ones((600, 900, 3), dtype=np.uint8) * 255
    
    # 标题
    cv2.putText(img, "OCR Performance Test Document", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # 第一段落
    paragraph1 = [
        "This is the first paragraph for testing.",
        "It contains multiple lines of text content.",
        "Each line will be processed by the OCR system."
    ]
    
    y_pos = 120
    for line in paragraph1:
        cv2.putText(img, line, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += 40
    
    # 表格
    cv2.putText(img, "Data Table:", (50, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    y_pos += 60
    
    table_rows = [
        ["ID", "Name", "Score", "Grade"],
        ["001", "Alice", "95", "A"],
        ["002", "Bob", "87", "B"],
        ["003", "Carol", "92", "A"]
    ]
    
    for row in table_rows:
        x_pos = 70
        for cell in row:
            cv2.putText(img, cell, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            x_pos += 120
        y_pos += 30
    
    # 第二段落
    paragraph2 = [
        "Second paragraph with additional content.",
        "Testing the OCR performance on mixed layouts.",
        "This includes both regular text and tabular data."
    ]
    
    y_pos += 30
    for line in paragraph2:
        cv2.putText(img, line, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += 40
    
    return img


def extract_region(image, x1, y1, x2, y2, region_name=None):
    """提取图像的指定区域"""
    h, w = image.shape[:2]
    
    # 确保坐标在有效范围内
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"无效的区域坐标: ({x1},{y1}) -> ({x2},{y2})")
    
    region = image[y1:y2, x1:x2]
    
    if region_name:
        print(f"提取区域 '{region_name}': ({x1},{y1}) -> ({x2},{y2}), 尺寸: {x2-x1}x{y2-y1}")
    
    return region


def run_ocr_test(image, region_name, ocr_instance):
    """执行OCR测试"""
    real_ocr = ocr_instance.ocr
    real_ocr.reset()
    
    print(f"\n{'='*60}")
    print(f"测试区域: {region_name}")
    print(f"{'='*60}")
    
    # 获取图像信息
    h, w = image.shape[:2]
    print(f"图像尺寸: {w} x {h} ({w*h:,} 像素)")
    
    # 分步计时执行
    total_start = time.time()
    
    # 步骤1: 结构分析
    step_start = time.time()
    structure_info = ocr_instance.analyze_text_structure(image)
    analysis_time = time.time() - step_start
    
    # 步骤2: 行检测
    step_start = time.time()
    lines = ocr_instance.detect_text_lines(image, structure_info)
    detection_time = time.time() - step_start
    
    # 步骤3: 拼接优化
    step_start = time.time()
    concat_groups = ocr_instance.optimize_concatenation(lines)
    concat_time = time.time() - step_start
    
    # 步骤4: OCR识别
    step_start = time.time()
    results = ocr_instance.recognize_multiline(image)
    recognition_time = time.time() - step_start
    
    total_time = time.time() - total_start
    
    # 获取OCR统计
    ocr_stats = real_ocr.get_stats()
    
    # 输出识别结果
    print(f"\n📝 识别结果 ({len(results)} 行):")
    for i, text in enumerate(results[:8], 1):  # 最多显示8行
        display_text = text[:50] + "..." if len(text) > 50 else text
        print(f"  {i:2d}. {display_text}")
    if len(results) > 8:
        print(f"  ... (总共 {len(results)} 行)")
    
    # 输出性能统计
    print(f"\n⏱️  性能统计:")
    print(f"  总处理时间: {total_time*1000:6.1f}ms")
    print(f"  - 结构分析: {analysis_time*1000:6.1f}ms ({analysis_time/total_time*100:4.1f}%)")
    print(f"  - 行检测:   {detection_time*1000:6.1f}ms ({detection_time/total_time*100:4.1f}%)")
    print(f"  - 拼接优化: {concat_time*1000:6.1f}ms ({concat_time/total_time*100:4.1f}%)")
    print(f"  - OCR识别:  {recognition_time*1000:6.1f}ms ({recognition_time/total_time*100:4.1f}%)")
    
    print(f"\n🔢 OCR调用统计:")
    print(f"  检测行数:     {len(lines):3d}")
    print(f"  拼接组数:     {len(concat_groups):3d}")
    print(f"  OCR调用次数:  {ocr_stats['total_calls']:3d}")
    print(f"  成功识别:     {ocr_stats['total_calls'] - ocr_stats['empty_results']:3d}")
    print(f"  空白结果:     {ocr_stats['empty_results']:3d}")
    print(f"  成功率:       {ocr_stats['success_rate']*100:5.1f}%")
    print(f"  效率提升:     {len(lines)/ocr_stats['total_calls']:5.1f}x")
    print(f"  平均OCR时间:  {ocr_stats['avg_time_ms']:6.1f}ms")
    
    # 显示拼接详情
    print(f"\n🔗 拼接分组详情:")
    for i, (_, indices) in enumerate(concat_groups[:5], 1):  # 最多显示5组
        print(f"  组{i}: {len(indices)}行 (行号: {[idx+1 for idx in indices]})")
    if len(concat_groups) > 5:
        print(f"  ... (总共 {len(concat_groups)} 组)")
    
    return {
        'region_name': region_name,
        'image_size': (w, h),
        'total_pixels': w * h,
        'detected_lines': len(lines),
        'concat_groups': len(concat_groups),
        'recognized_lines': len(results),
        'total_time_ms': total_time * 1000,
        'analysis_time_ms': analysis_time * 1000,
        'detection_time_ms': detection_time * 1000,
        'concat_time_ms': concat_time * 1000,
        'recognition_time_ms': recognition_time * 1000,
        'ocr_calls': ocr_stats['total_calls'],
        'ocr_time_ms': ocr_stats['total_time_ms'],
        'success_rate': ocr_stats['success_rate'],
        'efficiency': len(lines)/ocr_stats['total_calls'] if ocr_stats['total_calls'] > 0 else 0,
        'results': results
    }


def main():
    """主测试函数"""
    print("真实OCR性能测试")
    print("="*80)
    
    # 初始化OCR引擎
    print("🚀 初始化OCR引擎...")
    try:
        real_ocr = RealOCRWrapper(use_gpu=True)  # 先尝试GPU
        print("✅ OCR引擎初始化成功 (GPU)")
    except Exception as e:
        print(f"⚠️  GPU初始化失败，切换到CPU: {e}")
        try:
            real_ocr = RealOCRWrapper(use_gpu=False)
            print("✅ OCR引擎初始化成功 (CPU)")
        except Exception as e:
            print(f"❌ OCR引擎初始化失败: {e}")
            return None
    
    # 创建智能多行OCR实例
    ocr_instance = IntelligentMultilineOCR(
        ocr_engine=real_ocr,
        dynamic_width=True,
        width_strategy='adaptive'
    )
    
    # 创建测试文档
    print("📄 创建测试文档...")
    full_image = create_test_document()
    
    # 保存测试图像（可选）
    cv2.imwrite("test_document.jpg", full_image)
    print("💾 测试图像已保存为: test_document.jpg")
    
    # 测试结果存储
    all_results = []
    
    # 测试1: 完整图像
    result = run_ocr_test(full_image, "完整文档", ocr_instance)
    all_results.append(result)
    
    # 测试2: 预定义区域
    predefined_regions = [
        (50, 0, 900, 100, "标题区域"),          # 标题部分
        (50, 100, 900, 280, "第一段落"),        # 第一段落
        (50, 280, 900, 420, "表格区域"),        # 表格部分
        (50, 420, 900, 550, "第二段落"),        # 第二段落
        (0, 0, 450, 600, "左半部分"),           # 左半边
        (450, 0, 900, 600, "右半部分"),         # 右半边
    ]
    
    print(f"\n🎯 测试预定义区域 ({len(predefined_regions)} 个区域):")
    for x1, y1, x2, y2, name in predefined_regions:
        try:
            region_img = extract_region(full_image, x1, y1, x2, y2, name)
            result = run_ocr_test(region_img, name, ocr_instance)
            all_results.append(result)
        except Exception as e:
            print(f"❌ 区域 '{name}' 测试失败: {e}")
    
    # 测试3: 自定义区域（用户可以修改这些坐标）
    custom_regions = [
        (100, 150, 800, 250, "自定义区域1"),    # 用户可自定义
        (200, 300, 700, 400, "自定义区域2"),    # 用户可自定义
    ]
    
    print(f"\n✏️  测试自定义区域 ({len(custom_regions)} 个区域):")
    for x1, y1, x2, y2, name in custom_regions:
        try:
            region_img = extract_region(full_image, x1, y1, x2, y2, name)
            result = run_ocr_test(region_img, name, ocr_instance)
            all_results.append(result)
        except Exception as e:
            print(f"❌ 区域 '{name}' 测试失败: {e}")
    
    # 性能对比分析
    print(f"\n📊 性能对比分析")
    print("="*80)
    
    # 创建对比表格
    print(f"{'区域名称':15} {'尺寸':12} {'识别':4} {'OCR':4} {'时间':7} {'效率':6} {'成功率':7}")
    print("-" * 75)
    
    total_region_time = 0
    total_region_calls = 0
    total_region_lines = 0
    
    for i, result in enumerate(all_results):
        name = result['region_name'][:14]
        size = f"{result['image_size'][0]}x{result['image_size'][1]}"
        lines = result['recognized_lines']
        calls = result['ocr_calls']
        time_ms = result['total_time_ms']
        eff = result['efficiency']
        success = result['success_rate']
        
        print(f"{name:15} {size:12} {lines:4d} {calls:4d} {time_ms:6.0f}ms {eff:6.1f}x {success*100:6.1f}%")
        
        if i > 0:  # 跳过完整文档，只统计区域
            total_region_time += time_ms
            total_region_calls += calls
            total_region_lines += lines
    
    # 完整文档 vs 区域总计对比
    full_doc = all_results[0]
    print(f"\n🔍 整体对比:")
    print(f"完整文档:   时间={full_doc['total_time_ms']:6.0f}ms, OCR调用={full_doc['ocr_calls']:3d}, 识别={full_doc['recognized_lines']:3d}行")
    print(f"区域总计:   时间={total_region_time:6.0f}ms, OCR调用={total_region_calls:3d}, 识别={total_region_lines:3d}行")
    
    time_diff = total_region_time - full_doc['total_time_ms']
    calls_diff = total_region_calls - full_doc['ocr_calls']
    
    print(f"差异:       时间={time_diff:+6.0f}ms ({time_diff/full_doc['total_time_ms']*100:+5.1f}%), OCR={calls_diff:+3d}次")
    
    if time_diff < 0:
        print("🏆 完整文档处理更高效！")
    else:
        print("📋 分区域处理在某些情况下可能有优势")
    
    print(f"\n✅ 测试完成！共测试了 {len(all_results)} 个区域")
    return all_results


def test_custom_region():
    """自定义区域测试函数 - 用户可以调用此函数测试特定坐标"""
    print("🎯 自定义区域测试")
    print("="*50)
    
    # 用户可以修改这些参数
    image_path = "test_document.jpg"  # 图像路径
    x1, y1, x2, y2 = 100, 200, 800, 400  # 自定义坐标
    region_name = "用户自定义区域"
    
    try:
        # 加载图像
        if image_path == "test_document.jpg":
            # 如果是测试文档，重新生成
            image = create_test_document()
        else:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
        
        # 提取区域
        region = extract_region(image, x1, y1, x2, y2, region_name)
        
        # 初始化OCR
        real_ocr = RealOCRWrapper(use_gpu=False)
        ocr_instance = IntelligentMultilineOCR(
            ocr_engine=real_ocr,
            dynamic_width=True,
            width_strategy='adaptive'
        )
        
        # 执行测试
        result = run_ocr_test(region, region_name, ocr_instance)
        
        print(f"✅ 自定义区域测试完成")
        return result
        
    except Exception as e:
        print(f"❌ 自定义区域测试失败: {e}")
        return None


if __name__ == "__main__":
    try:
        # 运行主测试
        results = main()
        
        print(f"\n💡 使用提示:")
        print("1. 修改 test_custom_region() 函数中的坐标来测试特定区域")
        print("2. 修改 custom_regions 列表来添加更多自定义区域") 
        print("3. 查看生成的 test_document.jpg 来了解测试图像")
        print("4. 根据测试结果调整 width_strategy 参数优化性能")
        
        # 如果需要，也可以运行自定义区域测试
        # test_custom_region()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
