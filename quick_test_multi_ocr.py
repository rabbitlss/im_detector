# -*- coding: utf-8 -*-
"""
IntelligentMultilineOCR 快速入门
3分钟上手智能多行OCR
"""

import cv2
import numpy as np
from ultrafast_ocr import UltraFastOCR
from ultrafast_ocr.intelligent_multiline_ocr import IntelligentMultilineOCR


def quick_start_example():
    """最简单的使用示例"""
    print("🚀 智能多行OCR - 3分钟快速上手")
    print("="*50)
    
    # 步骤1: 初始化（只需要两行代码）
    print("📝 步骤1: 初始化OCR引擎")
    ocr_engine = UltraFastOCR()  # 自动检测GPU/CPU
    intelligent_ocr = IntelligentMultilineOCR(ocr_engine)
    print("✅ 初始化完成")
    
    # 步骤2: 创建测试图片（实际使用时替换为你的图片）
    print("\n📝 步骤2: 准备测试图片")
    test_image = create_simple_test()
    print("✅ 测试图片已创建")
    
    # 步骤3: 一行代码识别多行文字
    print("\n📝 步骤3: 执行多行识别")
    results = intelligent_ocr.recognize_multiline(test_image)
    print("✅ 识别完成")
    
    # 步骤4: 查看结果
    print("\n📋 识别结果:")
    for i, text in enumerate(results, 1):
        print(f"   第{i}行: {text}")
    
    return results


def create_simple_test():
    """创建简单测试图片"""
    img = np.ones((200, 500, 3), dtype=np.uint8) * 255
    texts = ["第一行文字", "第二行文字", "第三行文字"]
    
    y = 50
    for text in texts:
        cv2.putText(img, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        y += 60
    
    cv2.imwrite("quick_test.jpg", img)
    return img


def real_image_example():
    """真实图片使用示例"""
    print("\n" + "="*50)
    print("🖼️  真实图片使用示例")
    print("="*50)
    
    # 如果你有真实图片，替换这里的路径
    image_path = "your_image.jpg"  
    
    print(f"📂 尝试读取图片: {image_path}")
    
    try:
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️  图片不存在，使用测试图片代替")
            image = create_simple_test()
        else:
            print(f"✅ 成功读取图片，尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 初始化OCR
        ocr_engine = UltraFastOCR()
        intelligent_ocr = IntelligentMultilineOCR(ocr_engine)
        
        # 识别
        print("🔍 开始识别...")
        import time
        start = time.time()
        results = intelligent_ocr.recognize_multiline(image)
        elapsed = (time.time() - start) * 1000
        
        print(f"⚡ 识别完成，耗时: {elapsed:.1f}ms")
        print(f"📋 识别到 {len(results)} 行文字:")
        
        for i, text in enumerate(results, 1):
            print(f"   {i:2d}. {text}")
            
    except Exception as e:
        print(f"❌ 出错: {e}")


def advanced_config_example():
    """高级配置示例"""
    print("\n" + "="*50)
    print("⚙️  高级配置示例")
    print("="*50)
    
    # 不同场景的推荐配置
    configs = {
        "手机截图": {"max_concat_width": 960, "target_height": 48},
        "文档扫描": {"max_concat_width": 1920, "target_height": 64}, 
        "聊天记录": {"max_concat_width": 1280, "target_height": 40},
        "菜单识别": {"max_concat_width": 640, "target_height": 32}
    }
    
    test_image = create_simple_test()
    ocr_engine = UltraFastOCR()
    
    for scenario, config in configs.items():
        print(f"\n📱 {scenario} 配置:")
        print(f"   参数: {config}")
        
        # 使用特定配置
        intelligent_ocr = IntelligentMultilineOCR(ocr_engine, **config)
        
        # 测试性能
        import time
        start = time.time()
        results = intelligent_ocr.recognize_multiline(test_image)
        elapsed = (time.time() - start) * 1000
        
        print(f"   耗时: {elapsed:.1f}ms")
        print(f"   识别行数: {len(results)}")


def performance_tips():
    """性能优化提示"""
    print("\n" + "="*50)
    print("💡 性能优化提示")
    print("="*50)
    
    tips = [
        "1. GPU加速: UltraFastOCR(use_gpu=True) 可提速3-5倍",
        "2. 宽度设置: max_concat_width=1280 是速度和精度的最佳平衡点",
        "3. 图片预处理: 适当的对比度增强可提高识别率",
        "4. 批量处理: 多张图片时复用同一个OCR实例",
        "5. 内存优化: 处理大图片时可以先缩放到合适尺寸",
    ]
    
    for tip in tips:
        print(f"   {tip}")
    
    print(f"\n📊 典型性能指标:")
    print(f"   - 3-5行文字: 15-25ms")
    print(f"   - 8-10行文字: 25-40ms") 
    print(f"   - 15-20行文字: 40-60ms")


if __name__ == "__main__":
    print("IntelligentMultilineOCR 快速入门指南")
    print("="*60)
    
    try:
        # 1. 最简单的示例
        results = quick_start_example()
        
        # 2. 真实图片示例
        real_image_example()
        
        # 3. 高级配置示例
        advanced_config_example()
        
        # 4. 性能优化提示
        performance_tips()
        
        print(f"\n🎉 快速入门完成!")
        print(f"\n📚 更多示例请运行: python test_intelligent_multiline.py")
        print(f"🔧 如需调试请查看保存的图片: quick_test.jpg")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print(f"\n解决方案:")
        print(f"1. 确保已正确安装 ultrafast_ocr 包")
        print(f"2. 检查 Python 路径设置")
        print(f"3. 运行: pip install opencv-python numpy onnxruntime")
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        print(f"\n常见问题解决:")
        print(f"1. 模型文件缺失: 运行模型下载脚本")
        print(f"2. GPU不可用: 设置 use_gpu=False")
        print(f"3. 内存不足: 减小 max_concat_width 参数")
