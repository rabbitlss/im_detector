# -*- coding: utf-8 -*-
"""
超快速OCR简单使用示例
输入：图片区域
输出：识别的文字
"""

import cv2
import numpy as np
from ultra_fast_ocr import UltraFastOCR
import time


def simple_ocr_demo():
    """最简单的使用示例"""
    
    # 1. 初始化（只需一次）
    print("🚀 初始化OCR引擎...")
    ocr = UltraFastOCR(
        det_model_path="models/ocr/ch_PP-OCRv4_det.onnx",
        rec_model_path="models/ocr/ch_PP-OCRv4_rec.onnx",
        dict_path="models/ocr/ppocr_keys_v1.txt",
        use_gpu=True  # 有GPU用GPU，没有自动用CPU
    )
    print("✅ OCR引擎初始化完成\n")
    
    # 2. 识别单行文字（最快，3-5ms）
    def recognize_text(image_region):
        """
        输入：图片区域（numpy array）
        输出：识别的文字
        """
        text, confidence, time_ms = ocr.recognize_single_line(image_region)
        print(f"识别结果: {text}")
        print(f"置信度: {confidence:.2f}")
        print(f"耗时: {time_ms:.1f}ms")
        return text
    
    # 3. 创建测试图片
    print("📝 创建测试图片...")
    
    # 测试图片1：白底黑字
    test_img1 = np.ones((60, 300, 3), dtype=np.uint8) * 255
    cv2.putText(test_img1, "Hello World", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # 测试图片2：中文
    test_img2 = np.ones((60, 300, 3), dtype=np.uint8) * 255
    cv2.putText(test_img2, "Test Chinese", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # 4. 识别测试
    print("\n🔍 开始识别测试...")
    print("-" * 40)
    
    print("测试1：英文")
    text1 = recognize_text(test_img1)
    print()
    
    print("测试2：混合文字")
    text2 = recognize_text(test_img2)
    print()
    
    # 5. 性能测试
    print("⚡ 性能测试（100次）...")
    times = []
    for _ in range(100):
        start = time.time()
        _, _, _ = ocr.recognize_single_line(test_img1)
        times.append((time.time() - start) * 1000)
    
    avg_time = np.mean(times)
    print(f"平均耗时: {avg_time:.2f}ms")
    print(f"最快: {np.min(times):.2f}ms")
    print(f"最慢: {np.max(times):.2f}ms")
    print(f"FPS: {1000/avg_time:.1f}")


def real_image_demo(image_path):
    """使用真实图片的示例"""
    
    print(f"\n📸 处理真实图片: {image_path}")
    print("-" * 40)
    
    # 初始化OCR
    ocr = UltraFastOCR(
        det_model_path="models/ocr/ch_PP-OCRv4_det.onnx",
        rec_model_path="models/ocr/ch_PP-OCRv4_rec.onnx",
        dict_path="models/ocr/ppocr_keys_v1.txt",
        use_gpu=True
    )
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    
    height, width = img.shape[:2]
    print(f"图片尺寸: {width}x{height}")
    
    # 示例1：识别顶部区域（可能是标题/昵称）
    print("\n识别顶部区域（昵称）:")
    top_region = img[10:60, 100:300]  # 调整坐标以适应你的图片
    text, conf, time_ms = ocr.recognize_single_line(top_region)
    print(f"  文字: {text}")
    print(f"  置信度: {conf:.2f}")
    print(f"  耗时: {time_ms:.1f}ms")
    
    # 示例2：识别中间区域（可能是消息）
    print("\n识别中间区域（消息）:")
    middle_region = img[height//2-30:height//2+30, 50:width-50]
    text, conf, time_ms = ocr.recognize_single_line(middle_region)
    print(f"  文字: {text}")
    print(f"  置信度: {conf:.2f}")
    print(f"  耗时: {time_ms:.1f}ms")
    
    # 示例3：识别底部区域（可能是输入框）
    print("\n识别底部区域（输入框）:")
    bottom_region = img[height-60:height-10, 50:width-100]
    text, conf, time_ms = ocr.recognize_single_line(bottom_region)
    print(f"  文字: {text}")
    print(f"  置信度: {conf:.2f}")
    print(f"  耗时: {time_ms:.1f}ms")
    
    # 示例4：识别多行文字
    print("\n识别整张图片（多行）:")
    results = ocr.recognize_multiline(img)
    for i, (text, conf, box) in enumerate(results[:5]):  # 只显示前5个
        print(f"  第{i+1}行: {text} (置信度: {conf:.2f})")


def im_screenshot_demo():
    """专门针对IM截图的示例"""
    
    print("\n💬 IM截图OCR示例")
    print("-" * 40)
    
    # 初始化OCR
    ocr = UltraFastOCR(
        det_model_path="models/ocr/ch_PP-OCRv4_det.onnx",
        rec_model_path="models/ocr/ch_PP-OCRv4_rec.onnx",
        dict_path="models/ocr/ppocr_keys_v1.txt",
        use_gpu=True
    )
    
    def extract_im_texts(image_path, yolo_detections):
        """
        从YOLO检测结果中提取文字
        
        Args:
            image_path: 图片路径
            yolo_detections: YOLO检测结果
                [{'class': 'receiver_name', 'bbox': [x1,y1,x2,y2]}, ...]
        """
        img = cv2.imread(image_path)
        
        results = {}
        
        for detection in yolo_detections:
            class_name = detection['class']
            bbox = detection['bbox']
            
            # 只对需要OCR的类别进行文字提取
            if class_name in ['receiver_name', 'chat_message', 'contact_item', 'input_box']:
                x1, y1, x2, y2 = bbox
                roi = img[y1:y2, x1:x2]
                
                # 识别文字
                text, conf, time_ms = ocr.recognize_single_line(roi)
                
                if class_name not in results:
                    results[class_name] = []
                
                results[class_name].append({
                    'text': text,
                    'confidence': conf,
                    'time_ms': time_ms,
                    'bbox': bbox
                })
                
                print(f"{class_name}: {text} ({time_ms:.1f}ms)")
        
        return results
    
    # 模拟YOLO检测结果
    mock_detections = [
        {'class': 'receiver_name', 'bbox': [150, 20, 250, 50]},
        {'class': 'chat_message', 'bbox': [50, 100, 400, 150]},
        {'class': 'chat_message', 'bbox': [50, 160, 400, 210]},
        {'class': 'input_box', 'bbox': [50, 500, 400, 540]},
    ]
    
    # 如果有真实的IM截图，替换这里的路径
    # texts = extract_im_texts("wechat_screenshot.jpg", mock_detections)
    
    print("\n✅ IM文字提取完成")


def batch_processing_demo():
    """批量处理示例"""
    
    print("\n📦 批量处理示例")
    print("-" * 40)
    
    ocr = UltraFastOCR(
        det_model_path="models/ocr/ch_PP-OCRv4_det.onnx",
        rec_model_path="models/ocr/ch_PP-OCRv4_rec.onnx",
        dict_path="models/ocr/ppocr_keys_v1.txt",
        use_gpu=True
    )
    
    # 创建多个测试图片
    test_images = []
    texts_to_recognize = ["Hello", "World", "OCR", "Test", "Fast"]
    
    for text in texts_to_recognize:
        img = np.ones((60, 200, 3), dtype=np.uint8) * 255
        cv2.putText(img, text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        test_images.append(img)
    
    # 批量识别
    print(f"批量识别 {len(test_images)} 张图片...")
    
    start_time = time.time()
    results = []
    for i, img in enumerate(test_images):
        text, conf, time_ms = ocr.recognize_single_line(img)
        results.append(text)
        print(f"  图片{i+1}: {text} ({time_ms:.1f}ms)")
    
    total_time = (time.time() - start_time) * 1000
    avg_time = total_time / len(test_images)
    
    print(f"\n总耗时: {total_time:.1f}ms")
    print(f"平均每张: {avg_time:.1f}ms")
    print(f"处理速度: {1000/avg_time:.1f} FPS")


if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("超快速OCR使用示例")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # 如果提供了图片路径，处理真实图片
        real_image_demo(sys.argv[1])
    else:
        # 否则运行所有演示
        
        # 1. 简单演示
        simple_ocr_demo()
        
        # 2. 批量处理演示
        batch_processing_demo()
        
        # 3. IM截图演示
        im_screenshot_demo()
        
        print("\n" + "=" * 50)
        print("💡 提示：")
        print("1. 使用真实图片: python simple_usage.py your_image.jpg")
        print("2. 模型下载: 运行 download_ocr_models.sh")
        print("3. 性能: GPU 3-5ms, CPU 8-12ms")
        print("=" * 50)
