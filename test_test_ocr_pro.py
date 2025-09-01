# -*- coding: utf-8 -*-
"""
多行OCR调试脚本 - 逐步分析问题
"""

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from ultrafast_ocr.intelligent_multiline_ocr import IntelligentMultilineOCR
from ultrafast_ocr import UltraFastOCR


class DebugOCR:
    """调试版OCR包装器"""
    def __init__(self):
        try:
            self.real_ocr = UltraFastOCR(use_gpu=False)
            self.is_available = True
            print("✅ OCR引擎初始化成功")
        except Exception as e:
            print(f"❌ OCR引擎初始化失败: {e}")
            self.is_available = False
        
        self.call_count = 0
        self.call_results = []
    
    def recognize_single_line(self, image):
        """调试版单行识别"""
        self.call_count += 1
        
        if not self.is_available:
            return f"MockResult_{self.call_count}"
        
        try:
            result = self.real_ocr.recognize_single_line(image)
            
            # 记录详细信息
            h, w = image.shape[:2]
            self.call_results.append({
                'call_id': self.call_count,
                'image_size': (w, h),
                'result': result,
                'result_length': len(result) if result else 0,
                'is_empty': not bool(result.strip()) if result else True
            })
            
            print(f"  📞 OCR调用 #{self.call_count}: {w}x{h}px -> '{result}'")
            return result if result else ""
            
        except Exception as e:
            error_msg = f"Error_{self.call_count}"
            print(f"  ❌ OCR调用 #{self.call_count} 失败: {e}")
            return error_msg


def load_image_from_pixels(pixel_data, height, width, channels=3):
    """从像素数据创建图像"""
    if isinstance(pixel_data, str):
        # 如果是字符串格式的像素数据，需要解析
        pixel_array = eval(pixel_data)
    else:
        pixel_array = pixel_data
    
    # 转换为numpy数组
    img_array = np.array(pixel_array, dtype=np.uint8)
    
    # 重新整形为图像
    if channels == 3:
        img = img_array.reshape((height, width, channels))
    else:
        img = img_array.reshape((height, width))
    
    return img


def save_debug_images(image, lines, concat_groups, stage_name="debug"):
    """保存调试图像"""
    try:
        # 保存原图
        cv2.imwrite(f"{stage_name}_original.jpg", image)
        print(f"💾 保存原图: {stage_name}_original.jpg")
        
        # 保存检测到的行
        debug_img = image.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, line in enumerate(lines):
            color = colors[i % len(colors)]
            bbox = line.bbox
            cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
            cv2.putText(debug_img, f"L{i+1}", (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite(f"{stage_name}_lines_detected.jpg", debug_img)
        print(f"💾 保存行检测结果: {stage_name}_lines_detected.jpg")
        
        # 保存每行的单独图像
        for i, line in enumerate(lines[:10]):  # 最多保存10行
            line_filename = f"{stage_name}_line_{i+1}.jpg"
            cv2.imwrite(line_filename, line.image)
            print(f"💾 保存第{i+1}行: {line_filename}")
            
    except Exception as e:
        print(f"⚠️  保存调试图像失败: {e}")


def debug_step_by_step(image, save_debug=True):
    """逐步调试多行OCR"""
    print("🔍 开始逐步调试")
    print("="*60)
    
    # 初始化
    debug_ocr = DebugOCR()
    intelligent_ocr = IntelligentMultilineOCR(
        ocr_engine=debug_ocr,
        dynamic_width=True,
        width_strategy='adaptive'
    )
    
    h, w = image.shape[:2]
    print(f"📸 输入图像: {w}x{h} 像素")
    
    # 步骤1: 文字结构分析
    print(f"\n{'='*20} 步骤1: 文字结构分析 {'='*20}")
    structure_info = intelligent_ocr.analyze_text_structure(image)
    
    print(f"结构分析结果:")
    print(f"  字符高度: {structure_info['char_height']}px")
    print(f"  行高: {structure_info['line_height']}px")
    print(f"  行间距: {structure_info['line_spacing']}px")
    print(f"  字符数量: {structure_info['num_chars']}")
    print(f"  文字区域数: {len(structure_info['text_regions'])}")
    
    # 分析文字区域
    if structure_info['text_regions']:
        heights = [r.height for r in structure_info['text_regions']]
        widths = [r.width for r in structure_info['text_regions']]
        print(f"  区域高度范围: {min(heights)} - {max(heights)}px")
        print(f"  区域宽度范围: {min(widths)} - {max(widths)}px")
    else:
        print("  ⚠️  未检测到文字区域！")
    
    # 步骤2: 行检测
    print(f"\n{'='*20} 步骤2: 行检测 {'='*20}")
    lines = intelligent_ocr.detect_text_lines(image, structure_info)
    print(f"检测到 {len(lines)} 行文字")
    
    if len(lines) == 0:
        print("❌ 没有检测到任何行！问题可能在行检测阶段")
        return None
    
    # 详细分析每一行
    for i, line in enumerate(lines):
        h_line, w_line = line.image.shape[:2]
        print(f"  行{i+1}: 尺寸={w_line}x{h_line}px, 有效宽度={line.effective_width}px, "
              f"置信度={line.confidence:.2f}, 密度={getattr(line, 'text_density', 'N/A')}")
    
    # 步骤3: 拼接优化
    print(f"\n{'='*20} 步骤3: 拼接优化 {'='*20}")
    concat_groups = intelligent_ocr.optimize_concatenation(lines)
    print(f"拼接成 {len(concat_groups)} 组")
    
    for i, (concat_img, indices) in enumerate(concat_groups):
        h_concat, w_concat = concat_img.shape[:2]
        print(f"  组{i+1}: 包含{len(indices)}行 (行号: {[idx+1 for idx in indices]}), "
              f"拼接图像尺寸: {w_concat}x{h_concat}px")
        
        # 保存拼接后的图像
        if save_debug:
            cv2.imwrite(f"debug_concat_group_{i+1}.jpg", concat_img)
            print(f"    💾 保存拼接图像: debug_concat_group_{i+1}.jpg")
    
    # 步骤4: OCR识别
    print(f"\n{'='*20} 步骤4: OCR识别 {'='*20}")
    debug_ocr.call_count = 0  # 重置计数器
    debug_ocr.call_results = []
    
    print("开始OCR识别...")
    results = intelligent_ocr.recognize_multiline(image)
    
    print(f"\n最终识别结果 ({len(results)} 行):")
    for i, text in enumerate(results, 1):
        print(f"  {i}. '{text}'")
    
    # OCR调用统计
    print(f"\nOCR调用统计:")
    print(f"  总调用次数: {debug_ocr.call_count}")
    print(f"  成功识别: {sum(1 for r in debug_ocr.call_results if not r['is_empty'])}")
    print(f"  空白结果: {sum(1 for r in debug_ocr.call_results if r['is_empty'])}")
    
    # 保存调试图像
    if save_debug:
        save_debug_images(image, lines, concat_groups, "debug")
    
    return {
        'structure_info': structure_info,
        'lines': lines,
        'concat_groups': concat_groups,
        'results': results,
        'ocr_calls': debug_ocr.call_results
    }


def analyze_problem(debug_result):
    """分析问题所在"""
    print(f"\n🔍 问题分析")
    print("="*40)
    
    if debug_result is None:
        print("❌ 调试失败，无法分析")
        return
    
    structure_info = debug_result['structure_info']
    lines = debug_result['lines']
    concat_groups = debug_result['concat_groups']
    results = debug_result['results']
    ocr_calls = debug_result['ocr_calls']
    
    # 分析各阶段
    print("各阶段分析:")
    
    # 1. 结构分析
    if structure_info['num_chars'] == 0:
        print("❌ 结构分析阶段: 未检测到文字区域 -> 可能是图像预处理问题")
    else:
        print(f"✅ 结构分析阶段: 检测到 {structure_info['num_chars']} 个文字区域")
    
    # 2. 行检测
    if len(lines) == 0:
        print("❌ 行检测阶段: 未检测到文字行 -> 可能是投影分析或阈值问题")
    elif len(lines) == 1:
        print("⚠️  行检测阶段: 只检测到1行 -> 可能是行分割问题")
    else:
        print(f"✅ 行检测阶段: 检测到 {len(lines)} 行")
    
    # 3. 拼接优化
    if len(concat_groups) == len(lines):
        print("⚠️  拼接优化阶段: 没有进行拼接 -> 可能是宽度限制或相似性检查问题")
    else:
        efficiency = len(lines) / len(concat_groups)
        print(f"✅ 拼接优化阶段: {len(lines)}行 -> {len(concat_groups)}组, 效率提升 {efficiency:.1f}x")
    
    # 4. OCR识别
    successful_ocr = sum(1 for call in ocr_calls if not call['is_empty'])
    if successful_ocr == 0:
        print("❌ OCR识别阶段: 所有OCR调用都返回空结果 -> 可能是图像质量或OCR引擎问题")
    elif len(results) < len(lines):
        print(f"⚠️  OCR识别阶段: 识别结果少于检测行数 ({len(results)} < {len(lines)}) -> 可能是结果分割问题")
    else:
        print(f"✅ OCR识别阶段: 成功识别 {len(results)} 行")
    
    # 建议
    print(f"\n💡 建议:")
    if structure_info['num_chars'] == 0:
        print("1. 检查图像是否包含清晰的文字")
        print("2. 尝试调整二值化参数")
        print("3. 检查图像尺寸和分辨率")
    elif len(lines) <= 1:
        print("1. 检查水平投影分析参数")
        print("2. 调整最小行高和合并阈值")
        print("3. 验证行间距检测逻辑")
    elif len(results) == 0:
        print("1. 检查OCR引擎是否正常工作")
        print("2. 验证拼接后的图像质量")
        print("3. 尝试单独测试每行图像的OCR")


def test_with_user_image(image_path=None):
    """使用用户提供的图像进行测试
    
    Args:
        image_path: 本地图像文件路径（支持jpg, png等格式）
                   如果为None，则使用默认测试图像
    """
    print("📥 用户图像测试")
    print("="*40)
    
    # 方法1: 读取本地图像文件
    if image_path:
        print(f"📂 读取本地图像: {image_path}")
        test_img = cv2.imread(image_path)
        if test_img is None:
            print(f"❌ 无法读取图像: {image_path}")
            print("请确认:")
            print("1. 文件路径是否正确")
            print("2. 文件格式是否支持（jpg, png, bmp等）")
            return None
        
        # 获取图像信息
        if len(test_img.shape) == 2:
            # 灰度图像，转换为3通道
            test_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
        
        height, width = test_img.shape[:2]
        print(f"✅ 成功读取图像: {width}x{height}像素")
    
    # 方法2: 创建默认测试图像
    else:
        print("使用默认测试图像")
        width, height = 400, 200
        test_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 添加一些测试文字
        cv2.putText(test_img, "Line 1: First line of text", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(test_img, "Line 2: Second line of text", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(test_img, "Line 3: Third line of text", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 执行调试
    debug_result = debug_step_by_step(test_img, save_debug=True)
    analyze_problem(debug_result)
    
    return debug_result


if __name__ == "__main__":
    print("🔧 多行OCR调试工具")
    print("="*50)
    
    # ========== 用户配置区域 ==========
    # 方式1: 直接指定本地图像文件路径
    image_path = None  # 修改为您的图像路径，例如: "/path/to/your/image.jpg"
    
    # 方式2: 使用命令行参数
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"使用命令行参数图像: {image_path}")
    
    # 方式3: 交互式输入
    if not image_path:
        user_input = input("请输入图像路径（直接回车使用默认测试图像）: ").strip()
        if user_input:
            image_path = user_input
    # ====================================
    
    try:
        # 运行测试
        result = test_with_user_image(image_path)
        
        if result:
            print(f"\n✅ 调试完成！")
            print("请检查生成的调试图像文件：")
            print("- debug_original.jpg: 原始图像")
            print("- debug_lines_detected.jpg: 行检测结果")
            print("- debug_line_X.jpg: 各行单独图像")
            print("- debug_concat_group_X.jpg: 拼接后图像")
            
            print(f"\n📋 调试信息摘要:")
            if 'lines' in result:
                print(f"- 检测到 {len(result['lines'])} 行")
            if 'concat_groups' in result:
                print(f"- 拼接成 {len(result['concat_groups'])} 组")
            if 'results' in result:
                print(f"- 识别出 {len(result['results'])} 行文字")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
