# -*- coding: utf-8 -*-
"""
测试多行文字OCR识别
"""

import cv2
import numpy as np
from ultrafast_ocr import UltraFastOCR

def create_multiline_test_image():
    """创建多行文字测试图片"""
    # 创建白色背景
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # 添加多行文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    texts = [
        "第一行：这是测试文字",
        "第二行：Hello World",  
        "第三行：OCR测试123",
        "第四行：多行文字识别"
    ]
    
    y_offset = 50
    for text in texts:
        cv2.putText(img, text, (30, y_offset), font, 0.8, (0, 0, 0), 2)
        y_offset += 80
    
    return img

def test_ocr_modes():
    """测试不同的OCR模式"""
    print("="*60)
    print("多行文字OCR测试")
    print("="*60)
    
    # 创建测试图片
    test_img = create_multiline_test_image()
    cv2.imwrite('test_multiline.jpg', test_img)
    print("✅ 测试图片已保存: test_multiline.jpg")
    
    # 初始化OCR（尝试加载检测模型）
    print("\n初始化OCR引擎...")
    ocr = UltraFastOCR()
    
    # 检查是否有检测模型
    if ocr.det_session is not None:
        print("✅ 检测模型已加载，支持多行文字")
    else:
        print("⚠️ 检测模型未加载，仅支持单行文字")
        print("请先下载并转换检测模型：")
        print("  1. 下载: wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar")
        print("  2. 解压: tar -xf ch_PP-OCRv4_det_infer.tar")
        print("  3. 转换: paddle2onnx --model_dir ch_PP-OCRv4_det_infer --save_file ch_PP-OCRv4_det.onnx")
    
    print("\n" + "="*40)
    print("测试1: 单行模式（recognize_single_line）")
    print("="*40)
    result = ocr.recognize_single_line(test_img)
    print(f"结果: {result}")
    print("说明: 单行模式会把整张图当作一行处理，结果可能混乱")
    
    print("\n" + "="*40)
    print("测试2: 多行模式（recognize_multiline）")
    print("="*40)
    results = ocr.recognize_multiline(test_img, return_boxes=True)
    
    if results:
        print(f"检测到 {len(results)} 个文本区域：")
        for i, item in enumerate(results, 1):
            if isinstance(item, tuple) and len(item) >= 2:
                text, confidence = item[0], item[1]
                print(f"  行{i}: {text} (置信度: {confidence:.2f})")
            else:
                print(f"  行{i}: {item}")
    else:
        print("未检测到文字")
    
    print("\n" + "="*40)
    print("测试3: 通用识别模式（recognize）")
    print("="*40)
    # 单行模式
    result_single = ocr.recognize(test_img, single_line=True)
    print(f"单行: {result_single[:50]}..." if len(result_single) > 50 else f"单行: {result_single}")
    
    # 多行模式
    result_multi = ocr.recognize(test_img, single_line=False)
    print(f"多行: {result_multi}")

def test_real_image(image_path: str):
    """测试真实图片"""
    print("\n" + "="*60)
    print(f"测试真实图片: {image_path}")
    print("="*60)
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    
    print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")
    
    # 初始化OCR
    ocr = UltraFastOCR()
    
    # 多行识别
    import time
    start = time.time()
    results = ocr.recognize_multiline(img, return_boxes=True)
    elapsed = (time.time() - start) * 1000
    
    print(f"\n识别结果（耗时: {elapsed:.1f}ms）：")
    if results:
        for i, (text, conf, box) in enumerate(results[:20], 1):  # 最多显示20行
            print(f"{i:2d}. {text[:50]}..." if len(text) > 50 else f"{i:2d}. {text}")
    else:
        print("未检测到文字")

if __name__ == "__main__":
    # 测试基本功能
    test_ocr_modes()
    
    # 如果有真实图片，也可以测试
    # test_real_image("path/to/your/image.jpg")
