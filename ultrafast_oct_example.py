# -*- coding: utf-8 -*-
"""
UltraFast OCR 使用示例
完整的使用说明和演示
"""

import cv2
import numpy as np
import time
from pathlib import Path

# 导入自定义OCR包
from . import UltraFastOCR, OptimizedOCR, quick_ocr
from .utils import create_test_image, batch_create_test_images, download_models, check_models


def basic_usage_example():
    """基础使用示例"""
    print("\n" + "=" * 60)
    print("1. 基础使用示例")
    print("=" * 60)
    
    # 方法1：使用包级别函数（最简单）
    test_img = create_test_image("Hello World")
    text = quick_ocr(test_img)
    print(f"快速OCR结果: {text}")
    
    # 方法2：创建OCR实例
    ocr = UltraFastOCR()
    
    # 识别单行文字
    text = ocr.recognize_single_line(test_img)
    print(f"单行识别: {text}")
    
    # 带置信度和耗时
    text, confidence, time_ms = ocr.recognize_single_line(
        test_img, 
        return_confidence=True, 
        return_time=True
    )
    print(f"详细结果: {text} (置信度: {confidence:.2f}, 耗时: {time_ms:.1f}ms)")


def optimized_usage_example():
    """优化版使用示例"""
    print("\n" + "=" * 60)
    print("2. 优化版使用示例")
    print("=" * 60)
    
    # 使用优化版OCR（带缓存）
    ocr = OptimizedOCR(cache_size=1000, use_template_matching=True)
    
    # 创建测试图片
    test_images = batch_create_test_images([
        "发送", "Hello", "微信", "Send", "发送"  # 有重复
    ])
    
    print("首次识别（建立缓存）:")
    for i, img in enumerate(test_images):
        text, hit_cache, time_ms = ocr.recognize_with_cache(img)
        cache_status = "缓存命中" if hit_cache else "OCR识别"
        print(f"  图片{i+1}: {text} ({cache_status}, {time_ms:.1f}ms)")
    
    print("\n第二次识别（缓存命中）:")
    for i, img in enumerate(test_images):
        text, hit_cache, time_ms = ocr.recognize_with_cache(img)
        cache_status = "缓存命中" if hit_cache else "OCR识别"
        print(f"  图片{i+1}: {text} ({cache_status}, {time_ms:.1f}ms)")
    
    # 性能统计
    stats = ocr.get_statistics()
    print(f"\n性能统计:")
    print(f"  总调用: {stats['total_calls']}")
    print(f"  缓存命中率: {stats['cache_hit_rate']*100:.1f}%")
    print(f"  平均耗时: {stats['avg_time_ms']:.2f}ms")


def batch_processing_example():
    """批量处理示例"""
    print("\n" + "=" * 60)
    print("3. 批量处理示例")
    print("=" * 60)
    
    ocr = OptimizedOCR()
    
    # 创建大量测试图片
    texts = ["用户1", "用户2", "消息内容", "发送", "Hello", "World"] * 10
    test_images = batch_create_test_images(texts[:20])  # 20张图片
    
    # 串行处理
    print("串行处理:")
    start_time = time.time()
    results_serial = [ocr.recognize(img) for img in test_images]
    serial_time = time.time() - start_time
    print(f"  耗时: {serial_time*1000:.1f}ms")
    print(f"  平均: {serial_time*1000/len(test_images):.1f}ms/图片")
    
    # 并行处理
    print("\n并行处理:")
    start_time = time.time()
    results_parallel = ocr.batch_recognize(test_images, use_parallel=True)
    parallel_time = time.time() - start_time
    print(f"  耗时: {parallel_time*1000:.1f}ms")
    print(f"  平均: {parallel_time*1000/len(test_images):.1f}ms/图片")
    print(f"  加速: {serial_time/parallel_time:.1f}x")


def im_interface_example():
    """IM界面识别示例"""
    print("\n" + "=" * 60)
    print("4. IM界面识别示例")
    print("=" * 60)
    
    ocr = OptimizedOCR()
    
    # 模拟创建一个IM界面
    im_interface = np.ones((600, 400, 3), dtype=np.uint8) * 240
    
    # 绘制各种元素
    elements = [
        {"type": "receiver_name", "text": "张三", "bbox": [150, 20, 220, 50]},
        {"type": "chat_message", "text": "你好", "bbox": [50, 100, 150, 130]},
        {"type": "chat_message", "text": "在吗？", "bbox": [50, 140, 180, 170]},
        {"type": "input_box", "text": "请输入消息", "bbox": [50, 520, 300, 550]},
        {"type": "send_button", "text": "发送", "bbox": [320, 525, 380, 545]},
    ]
    
    # 在图片上绘制元素
    for element in elements:
        x1, y1, x2, y2 = element["bbox"]
        # 绘制白色背景
        cv2.rectangle(im_interface, (x1, y1), (x2, y2), (255, 255, 255), -1)
        # 绘制文字
        cv2.putText(im_interface, element["text"], (x1+5, y1+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # 识别各个元素
    print("IM界面元素识别:")
    total_time = 0
    
    for element in elements:
        start_time = time.time()
        text = ocr.recognize_im_element(
            im_interface, 
            element["type"], 
            element["bbox"]
        )
        elapsed = (time.time() - start_time) * 1000
        total_time += elapsed
        
        expected = element["text"]
        status = "✅" if text == expected else "❌"
        print(f"  {element['type']:15s}: '{text}' (期望: '{expected}') {status} ({elapsed:.1f}ms)")
    
    print(f"\n总耗时: {total_time:.1f}ms")
    print(f"平均: {total_time/len(elements):.1f}ms/元素")


def performance_benchmark():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("5. 性能基准测试")
    print("=" * 60)
    
    # 创建测试图片
    test_texts = ["Hello", "World", "OCR", "Fast", "Test"]
    test_images = batch_create_test_images(test_texts)
    
    # 测试基础OCR
    print("基础OCR性能:")
    basic_ocr = UltraFastOCR()
    basic_stats = basic_ocr.benchmark(test_images, rounds=50)
    
    print(f"  平均耗时: {basic_stats['avg_time_ms']:.2f}ms")
    print(f"  最快: {basic_stats['min_time_ms']:.2f}ms")
    print(f"  最慢: {basic_stats['max_time_ms']:.2f}ms")
    print(f"  FPS: {basic_stats['fps']:.1f}")
    
    # 测试优化OCR
    print("\n优化OCR性能:")
    optimized_ocr = OptimizedOCR()
    optimized_stats = optimized_ocr.benchmark(test_images, rounds=50)
    
    print(f"无缓存:")
    print(f"  平均耗时: {optimized_stats['no_cache']['avg_time_ms']:.2f}ms")
    print(f"  FPS: {optimized_stats['no_cache']['fps']:.1f}")
    
    print(f"有缓存:")
    print(f"  平均耗时: {optimized_stats['with_cache']['avg_time_ms']:.2f}ms")
    print(f"  FPS: {optimized_stats['with_cache']['fps']:.1f}")
    print(f"  缓存命中率: {optimized_stats['with_cache']['hit_rate']*100:.1f}%")
    print(f"  加速比: {optimized_stats['speedup']:.1f}x")


def real_image_example(image_path: str):
    """真实图片识别示例"""
    print("\n" + "=" * 60)
    print("6. 真实图片识别示例")
    print("=" * 60)
    
    if not Path(image_path).exists():
        print(f"❌ 图片不存在: {image_path}")
        return
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    
    print(f"📸 处理图片: {image_path}")
    print(f"图片尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 初始化OCR
    ocr = OptimizedOCR()
    
    # 方式1: 识别整张图片
    print("\n方式1: 整图识别")
    start_time = time.time()
    results = ocr.ocr.recognize_multiline(image, return_boxes=True)
    elapsed = (time.time() - start_time) * 1000
    
    print(f"识别到 {len(results)} 个文字区域 (耗时: {elapsed:.1f}ms)")
    for i, (text, conf, box) in enumerate(results[:10]):  # 只显示前10个
        print(f"  {i+1:2d}: {text} (置信度: {conf:.2f})")
    
    # 方式2: 识别指定区域
    print("\n方式2: 区域识别")
    h, w = image.shape[:2]
    
    # 定义一些感兴趣的区域
    regions = [
        {"name": "顶部区域", "bbox": [0, 0, w, h//4]},
        {"name": "中间区域", "bbox": [0, h//4, w, 3*h//4]},
        {"name": "底部区域", "bbox": [0, 3*h//4, w, h]},
    ]
    
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        roi = image[y1:y2, x1:x2]
        
        start_time = time.time()
        text = ocr.recognize(roi)
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {region['name']}: '{text}' ({elapsed:.1f}ms)")


def model_management_example():
    """模型管理示例"""
    print("\n" + "=" * 60)
    print("7. 模型管理示例")
    print("=" * 60)
    
    # 检查模型状态
    print("检查模型状态:")
    status = check_models()
    for model_type, exists in status.items():
        status_text = "✅ 存在" if exists else "❌ 不存在"
        print(f"  {model_type}: {status_text}")
    
    # 如果模型不存在，提示下载
    missing_models = [k for k, v in status.items() if not v]
    if missing_models:
        print(f"\n缺少模型: {missing_models}")
        print("运行以下命令下载:")
        print("  from ultrafast_ocr.utils import download_models")
        print("  download_models()")
    else:
        print("\n✅ 所有模型已就绪!")


def error_handling_example():
    """错误处理示例"""
    print("\n" + "=" * 60)
    print("8. 错误处理示例")
    print("=" * 60)
    
    ocr = OptimizedOCR()
    
    # 测试各种边界情况
    test_cases = [
        ("空图片", np.array([])),
        ("过小图片", np.ones((5, 5, 3), dtype=np.uint8)),
        ("过大图片", np.ones((5000, 5000, 3), dtype=np.uint8)),
        ("None输入", None),
        ("正常图片", create_test_image("Normal")),
    ]
    
    for case_name, test_input in test_cases:
        try:
            if test_input is not None and test_input.size > 0:
                result = ocr.recognize(test_input)
                print(f"  {case_name}: '{result}' ✅")
            else:
                print(f"  {case_name}: 跳过无效输入 ⚠️")
        except Exception as e:
            print(f"  {case_name}: 错误 - {str(e)[:50]}... ❌")


def main():
    """主演示函数"""
    print("🚀 UltraFast OCR 完整示例")
    print("=" * 80)
    print("基于ONNX直接推理的超快速OCR引擎")
    print("识别速度: 3-10ms (GPU) / 8-15ms (CPU)")
    print("=" * 80)
    
    try:
        # 1. 基础使用
        basic_usage_example()
        
        # 2. 优化版使用
        optimized_usage_example()
        
        # 3. 批量处理
        batch_processing_example()
        
        # 4. IM界面识别
        im_interface_example()
        
        # 5. 性能测试
        performance_benchmark()
        
        # 6. 真实图片示例（如果有的话）
        # real_image_example("path/to/your/image.jpg")
        
        # 7. 模型管理
        model_management_example()
        
        # 8. 错误处理
        error_handling_example()
        
        print("\n" + "=" * 80)
        print("🎉 所有示例运行完成！")
        print("\n📚 使用说明:")
        print("  1. 基础用法: from ultrafast_ocr import quick_ocr; text = quick_ocr(image)")
        print("  2. 高级用法: from ultrafast_ocr import OptimizedOCR; ocr = OptimizedOCR()")
        print("  3. 模型下载: from ultrafast_ocr.utils import download_models; download_models()")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        print("\n可能的原因:")
        print("  1. 缺少模型文件 - 运行 download_models()")
        print("  2. 缺少依赖包 - pip install onnxruntime opencv-python numpy")
        print("  3. ONNX Runtime问题 - 检查CUDA/GPU设置")


if __name__ == "__main__":
    main()
