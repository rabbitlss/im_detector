# -*- coding: utf-8 -*-
"""
测试批量OCR处理器的使用示例
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# 添加路径以导入模块
sys.path.insert(0, str(Path(__file__).parent))

from batch_ocr_processor import BatchOCRProcessor, TextRegion
from fast_text_line_splitter import ProjectionLineSplitter

def test_batch_ocr_with_real_image():
    """使用真实图片测试批量OCR"""
    
    print("🚀 批量OCR处理测试")
    print("=" * 60)
    
    # 初始化组件
    batch_processor = BatchOCRProcessor(
        max_batch_size=32,
        target_height=48,  # 目标高度
        padding=5  # 区域间隔
    )
    line_splitter = ProjectionLineSplitter()
    
    # 读取测试图片（请替换为你的实际图片路径）
    image_path = "your_test_image.jpg"  # 替换为实际路径
    
    # 如果没有真实图片，创建模拟图片
    if not Path(image_path).exists():
        print("创建模拟测试图片...")
        image = create_test_image()
    else:
        image = cv2.imread(image_path)
    
    # 模拟YOLO检测结果（实际使用时从YOLO获取）
    detections = [
        {
            'class': 'chat_area',
            'bbox': [50, 50, 400, 200],  # x1, y1, x2, y2
            'confidence': 0.95
        },
        {
            'class': 'input_area',
            'bbox': [50, 250, 400, 320],
            'confidence': 0.92
        },
        {
            'class': 'button_area',
            'bbox': [50, 350, 200, 400],
            'confidence': 0.88
        }
    ]
    
    # 执行批量OCR处理
    result = batch_processor.process_im_image_batch(
        image=image,
        yolo_detections=detections,
        line_splitter=line_splitter
    )
    
    # 显示结果
    if result['success']:
        print("\n✅ 批量OCR处理成功！")
        print(f"统计信息：")
        stats = result['stats']
        print(f"  - OCR调用次数: {stats['ocr_calls']} 次")
        print(f"  - 处理区域数: {stats['total_regions']}")
        print(f"  - 总耗时: {stats['total_time_ms']:.1f}ms")
        print(f"  - 平均耗时: {stats['avg_time_per_region']:.1f}ms/区域")
        
        print("\n识别结果：")
        for source_type, regions in result['results'].items():
            print(f"\n{source_type}:")
            for region in regions:
                print(f"  - {region['region_id']}: {region['text']}")
                print(f"    置信度: {region['confidence']:.3f}")
                print(f"    位置: {region['bbox']}")
        
        # 保存可视化结果
        batch_processor.save_batch_visualization(result, "batch_ocr_output")
        print("\n💾 结果已保存到 batch_ocr_output/ 目录")
    else:
        print(f"❌ 处理失败: {result.get('message', 'Unknown error')}")

def test_direct_text_regions():
    """直接提供文字区域进行批量OCR"""
    
    print("\n🔧 直接文字区域批量处理")
    print("=" * 60)
    
    batch_processor = BatchOCRProcessor()
    
    # 创建一些测试文字区域
    text_regions = []
    
    # 模拟多个文字行区域
    for i in range(5):
        # 创建模拟的文字图片
        text_img = np.ones((30, 150, 3), dtype=np.uint8) * 255
        cv2.putText(text_img, f"Text Line {i+1}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        region = TextRegion(
            region_id=f"region_{i+1}",
            bbox=(0, i*40, 150, i*40+30),
            image_crop=text_img,
            source_type="test",
            confidence=0.9
        )
        text_regions.append(region)
    
    # 创建批处理图像
    batch_image, region_mappings = batch_processor.create_batch_image(text_regions)
    
    print(f"批处理图像尺寸: {batch_image.shape}")
    print(f"包含区域数: {len(region_mappings)}")
    
    # 执行OCR识别
    results = batch_processor.simulate_ocr_recognition(batch_image, region_mappings)
    
    print("\n识别结果:")
    for result in results:
        print(f"  {result.region_id}: {result.text_content}")
        print(f"    置信度: {result.confidence:.3f}")
    
    # 显示性能优势
    print(f"\n🎯 性能优势:")
    print(f"  传统方法: {len(text_regions)} 次OCR调用")
    print(f"  批量方法: 1 次OCR调用")
    print(f"  理论加速: {len(text_regions)}x")

def create_test_image():
    """创建测试图片"""
    # 创建一个模拟的聊天界面图片
    img = np.ones((500, 450, 3), dtype=np.uint8) * 240
    
    # 添加一些文字区域
    texts = [
        ("聊天区域 - 消息1", (60, 80)),
        ("聊天区域 - 消息2", (60, 120)),
        ("聊天区域 - 消息3", (60, 160)),
        ("输入框文字", (60, 280)),
        ("发送按钮", (80, 370))
    ]
    
    for text, (x, y) in texts:
        cv2.putText(img, text, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img

if __name__ == "__main__":
    # 测试批量OCR处理
    test_batch_ocr_with_real_image()
    
    # 测试直接处理文字区域
    test_direct_text_regions()
    
    print("\n🎉 所有测试完成！")
