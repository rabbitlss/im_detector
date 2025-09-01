# -*- coding: utf-8 -*-
"""
测试UltraFastOCR的多行文字识别功能
验证检测模型和识别模型的协作
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from ultrafast_ocr import UltraFastOCR


def create_multiline_test_image():
    """创建多行文字测试图像"""
    img = np.ones((500, 800, 3), dtype=np.uint8) * 255
    
    # 添加多行文字（OpenCV只支持英文）
    texts = [
        ("Line 1: Hello World", (50, 50)),
        ("Line 2: This is a test", (50, 120)),
        ("Line 3: OCR Detection Test", (50, 190)),
        ("Line 4: Multiple Lines", (50, 260)),
        ("Line 5: Recognition Demo", (50, 330)),
        ("Line 6: Fast Processing", (50, 400)),
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    for text, pos in texts:
        cv2.putText(img, text, pos, font, 1.0, (0, 0, 0), 2)
    
    return img


def test_single_vs_multiline():
    """测试单行和多行识别的对比"""
    print("="*70)
    print("🧪 测试1: 单行 vs 多行识别对比")
    print("="*70)
    
    # 初始化OCR（带检测模型）
    print("\n1. 初始化OCR引擎（启用检测模型）...")
    ocr = UltraFastOCR(
        det_model_path="models/ocr/ch_PP-OCRv4_det.onnx",
        rec_model_path="models/ocr/ch_PP-OCRv4_rec.onnx",
        dict_path="models/ocr/ppocr_keys_v1.txt",
        use_gpu=False,
        enable_detection=True  # 启用检测
    )
    
    # 创建测试图像
    test_img = create_multiline_test_image()
    cv2.imwrite("test_multiline_input.jpg", test_img)
    print("📝 测试图像已保存: test_multiline_input.jpg")
    
    # 测试单行识别（整图作为单行）
    print("\n2. 测试单行识别模式...")
    start = time.time()
    single_result = ocr.recognize_single_line(test_img)
    single_time = (time.time() - start) * 1000
    print(f"   结果: '{single_result[:50]}...'")
    print(f"   耗时: {single_time:.1f}ms")
    
    # 测试多行识别
    print("\n3. 测试多行识别模式...")
    start = time.time()
    multi_results = ocr.recognize_multiline(test_img)
    multi_time = (time.time() - start) * 1000
    
    print(f"\n   识别到 {len(multi_results)} 行:")
    for i, text in enumerate(multi_results, 1):
        print(f"   行{i}: {text}")
    print(f"   总耗时: {multi_time:.1f}ms")
    
    # 性能对比
    print("\n4. 性能对比:")
    print(f"   单行模式: {single_time:.1f}ms (可能丢失信息)")
    print(f"   多行模式: {multi_time:.1f}ms (完整识别)")
    print(f"   速度差异: {multi_time/single_time:.1f}x")


def test_with_confidence():
    """测试带置信度的多行识别"""
    print("\n" + "="*70)
    print("🧪 测试2: 带置信度的多行识别")
    print("="*70)
    
    # 初始化OCR
    ocr = UltraFastOCR(
        det_model_path="models/ocr/ch_PP-OCRv4_det.onnx",
        rec_model_path="models/ocr/ch_PP-OCRv4_rec.onnx",
        dict_path="models/ocr/ppocr_keys_v1.txt",
        use_gpu=False
    )
    
    # 创建测试图像
    test_img = create_multiline_test_image()
    
    # 识别并返回置信度
    print("\n执行多行识别（返回置信度）...")
    results = ocr.recognize_multiline(test_img, return_confidence=True)
    
    print(f"\n识别结果（共{len(results)}行）:")
    total_conf = 0
    for i, (text, conf) in enumerate(results, 1):
        print(f"  行{i}: '{text}' (置信度: {conf:.3f})")
        total_conf += conf
    
    if results:
        avg_conf = total_conf / len(results)
        print(f"\n平均置信度: {avg_conf:.3f}")


def test_with_boxes():
    """测试返回检测框的多行识别"""
    print("\n" + "="*70)
    print("🧪 测试3: 返回检测框坐标")
    print("="*70)
    
    # 初始化OCR
    ocr = UltraFastOCR(
        det_model_path="models/ocr/ch_PP-OCRv4_det.onnx",
        rec_model_path="models/ocr/ch_PP-OCRv4_rec.onnx",
        dict_path="models/ocr/ppocr_keys_v1.txt",
        use_gpu=False
    )
    
    # 创建测试图像
    test_img = create_multiline_test_image()
    
    # 识别并返回框坐标
    print("\n执行多行识别（返回检测框）...")
    results = ocr.recognize_multiline(test_img, return_boxes=True)
    
    # 可视化结果
    vis_img = test_img.copy()
    
    print(f"\n识别结果（共{len(results)}行）:")
    for i, (text, conf, box) in enumerate(results, 1):
        print(f"  行{i}: '{text}'")
        print(f"       置信度: {conf:.3f}")
        print(f"       坐标: {box[0]} -> {box[2]}")
        
        # 绘制检测框
        box_array = np.array(box, dtype=np.int32)
        cv2.polylines(vis_img, [box_array], True, (0, 255, 0), 2)
        
        # 添加行号
        cv2.putText(vis_img, f"L{i}", tuple(box[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # 保存可视化结果
    cv2.imwrite("test_multiline_boxes.jpg", vis_img)
    print("\n📸 可视化结果已保存: test_multiline_boxes.jpg")


def test_real_image(image_path):
    """测试真实图像"""
    print("\n" + "="*70)
    print("🧪 测试4: 真实图像识别")
    print("="*70)
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    print(f"\n图像信息:")
    print(f"  路径: {image_path}")
    print(f"  尺寸: {img.shape[1]}x{img.shape[0]}")
    
    # 初始化OCR
    print("\n初始化OCR引擎...")
    ocr = UltraFastOCR(
        det_model_path="models/ocr/ch_PP-OCRv4_det.onnx",
        rec_model_path="models/ocr/ch_PP-OCRv4_rec.onnx",
        dict_path="models/ocr/ppocr_keys_v1.txt",
        use_gpu=False
    )
    
    # 多行识别
    print("\n执行多行识别...")
    start = time.time()
    results = ocr.recognize_multiline(img, return_confidence=True, min_confidence=0.3)
    elapsed = (time.time() - start) * 1000
    
    print(f"\n识别结果（共{len(results)}行，耗时{elapsed:.1f}ms）:")
    for i, (text, conf) in enumerate(results, 1):
        print(f"  行{i}: {text} (置信度: {conf:.3f})")


def test_performance():
    """性能基准测试"""
    print("\n" + "="*70)
    print("⚡ 性能基准测试")
    print("="*70)
    
    # 初始化OCR
    ocr = UltraFastOCR(
        det_model_path="models/ocr/ch_PP-OCRv4_det.onnx",
        rec_model_path="models/ocr/ch_PP-OCRv4_rec.onnx",
        dict_path="models/ocr/ppocr_keys_v1.txt",
        use_gpu=False
    )
    
    # 创建不同复杂度的测试图像
    test_cases = [
        ("单行文字", create_single_line_image()),
        ("3行文字", create_lines_image(3)),
        ("6行文字", create_lines_image(6)),
        ("10行文字", create_lines_image(10)),
    ]
    
    print("\n测试配置:")
    print("  测试轮数: 10")
    print("  使用GPU: 否")
    
    for name, img in test_cases:
        times = []
        for _ in range(10):
            start = time.time()
            _ = ocr.recognize_multiline(img)
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"\n{name}:")
        print(f"  平均耗时: {avg_time:.1f}ms")
        print(f"  最快: {min_time:.1f}ms")
        print(f"  最慢: {max_time:.1f}ms")
        print(f"  FPS: {1000/avg_time:.1f}")


def create_single_line_image():
    """创建单行文字图像"""
    img = np.ones((60, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Single Line Text", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


def create_lines_image(num_lines):
    """创建多行文字图像"""
    height = 60 * num_lines + 40
    img = np.ones((height, 600, 3), dtype=np.uint8) * 255
    
    for i in range(num_lines):
        y = 40 + i * 60
        text = f"Line {i+1}: Test Text Content"
        cv2.putText(img, text, (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return img


def main():
    """主函数"""
    print("="*70)
    print("UltraFastOCR 多行文字识别测试")
    print("="*70)
    
    import sys
    
    # 运行各项测试
    test_single_vs_multiline()
    test_with_confidence()
    test_with_boxes()
    
    # 如果提供了图像路径，测试真实图像
    if len(sys.argv) > 1:
        test_real_image(sys.argv[1])
    
    # 性能测试
    try:
        global np
        import numpy as np
        test_performance()
    except ImportError:
        print("\n⚠️ 跳过性能测试（需要numpy）")
    
    print("\n" + "="*70)
    print("✅ 所有测试完成！")
    print("="*70)
    
    print("\n💡 提示:")
    print("1. 测试真实图像: python test_ultrafast_multiline.py your_image.jpg")
    print("2. 查看生成的可视化结果: test_multiline_boxes.jpg")
    print("3. 确保模型文件存在: models/ocr/ch_PP-OCRv4_*.onnx")


if __name__ == "__main__":
    main()
