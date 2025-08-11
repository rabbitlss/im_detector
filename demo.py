# -*- coding: utf-8 -*-
"""
演示脚本：展示IM检测器的完整功能
"""

import os
import cv2
import time
from pathlib import Path
from end2end_pipeline import End2EndIMDetector


def create_sample_data():
    """创建示例数据结构"""
    print("📁 创建示例数据结构...")
    
    dirs = [
        "./data/raw_images",
        "./data/test_images", 
        "./models",
        "./results",
        "./demo_results"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        
    print("✅ 目录结构创建完成")
    print("\n请将IM截图放入以下文件夹：")
    print("  - 训练图片：./data/raw_images/")
    print("  - 测试图片：./data/test_images/")
    print("  - 支持格式：.jpg, .jpeg, .png")


def check_requirements():
    """检查环境要求"""
    print("🔍 检查环境要求...")
    
    # 检查Python包
    required_packages = {
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics YOLO',
        'opencv-python': 'OpenCV',
        'openai': 'OpenAI API',
        'numpy': 'NumPy',
        'tqdm': 'TQDM'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n请安装缺失的包：")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # 检查OpenAI API Key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"  ✅ OpenAI API Key - 已设置")
    else:
        print(f"  ⚠️ OpenAI API Key - 未设置")
        print(f"     export OPENAI_API_KEY='your-key-here'")
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✅ CUDA - {gpu_count} 个GPU ({gpu_name})")
        else:
            print(f"  ⚠️ CUDA - 未检测到GPU，将使用CPU")
    except:
        print(f"  ❌ CUDA - 检查失败")
    
    return len(missing_packages) == 0


def demo_auto_labeling():
    """演示自动标注功能"""
    print("\n🏷️ 演示：GPT-4V自动标注")
    print("=" * 40)
    
    from auto_labeler import GPT4VAutoLabeler
    
    # 检查图片
    raw_images = list(Path("./data/raw_images").glob("*.jpg"))
    if len(raw_images) == 0:
        print("❌ 未找到原始图片")
        print("请将IM截图放入 ./data/raw_images/ 文件夹")
        return False
    
    print(f"📸 找到 {len(raw_images)} 张图片")
    
    # 只标注前3张做演示
    demo_images = raw_images[:3]
    
    labeler = GPT4VAutoLabeler()
    
    print(f"🤖 开始标注 {len(demo_images)} 张图片（演示模式）...")
    
    try:
        labeler.batch_labeling(
            "./data/raw_images",
            "./data/labeled_data/demo",
            max_images=len(demo_images)
        )
        
        # 检查标注结果
        labeled_images = list(Path("./data/labeled_data/demo/images").glob("*"))
        labeled_texts = list(Path("./data/labeled_data/demo/labels").glob("*.txt"))
        
        print(f"✅ 标注完成：{len(labeled_images)} 张图片，{len(labeled_texts)} 个标注文件")
        
        # 显示标注内容示例
        if labeled_texts:
            with open(labeled_texts[0], 'r') as f:
                lines = f.readlines()
            print(f"📄 标注示例 ({labeled_texts[0].name}):")
            for line in lines[:3]:  # 显示前3行
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    from config import CLASS_NAMES
                    class_name = CLASS_NAMES[class_id]
                    print(f"  - {class_name}: 坐标 {parts[1:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 标注失败：{e}")
        return False


def demo_training():
    """演示模型训练"""
    print("\n🚂 演示：YOLO模型训练")
    print("=" * 40)
    
    # 检查标注数据
    labeled_images = list(Path("./data/labeled_data/demo/images").glob("*"))
    if len(labeled_images) == 0:
        print("❌ 未找到标注数据，请先运行自动标注演示")
        return False
    
    from yolo_trainer import YOLOTrainer
    
    trainer = YOLOTrainer()
    
    try:
        # 准备数据集
        print("📊 准备训练数据集...")
        dataset_config = trainer.prepare_dataset("./data/labeled_data/demo")
        
        # 快速训练（仅用于演示）
        print("🏃‍♂️ 快速训练（10轮，仅用于演示）...")
        model = trainer.train_model(
            dataset_config=dataset_config,
            epochs=10,  # 演示用，实际需要100+轮
            batch_size=2,
            model_type='yolov8n'
        )
        
        # 评估
        print("📈 评估模型...")
        metrics = trainer.evaluate_model()
        
        print("📊 训练结果：")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # 保存模型
        model_path = trainer.save_model("./models/demo_model.pt")
        print(f"💾 演示模型已保存：{model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练失败：{e}")
        return False


def demo_detection():
    """演示检测功能"""
    print("\n🔍 演示：IM元素检测")
    print("=" * 40)
    
    # 检查模型
    model_path = "./models/demo_model.pt"
    if not os.path.exists(model_path):
        print("❌ 演示模型不存在，请先运行训练演示")
        return False
    
    # 检查测试图片
    test_images = list(Path("./data/test_images").glob("*.jpg"))
    if len(test_images) == 0:
        # 使用原始图片作为测试
        test_images = list(Path("./data/raw_images").glob("*.jpg"))[:2]
    
    if len(test_images) == 0:
        print("❌ 未找到测试图片")
        return False
    
    from im_detector import IMDetector
    
    # 初始化检测器
    detector = IMDetector(model_path, confidence=0.3)
    
    print(f"🎯 测试检测功能，共 {len(test_images)} 张图片...")
    
    for i, img_path in enumerate(test_images):
        print(f"\n📸 检测图片 {i+1}: {img_path.name}")
        
        # 检测
        start_time = time.time()
        result = detector.predict(str(img_path), return_image=True)
        inference_time = time.time() - start_time
        
        # 显示结果
        detection_count = 0
        for class_name, objects in result.items():
            if class_name not in ['metadata', 'annotated_image']:
                if len(objects) > 0:
                    print(f"  ✅ {class_name}: {len(objects)} 个")
                    detection_count += len(objects)
        
        print(f"  ⚡ 推理时间: {inference_time*1000:.2f} ms")
        print(f"  📊 检测总数: {detection_count} 个元素")
        
        # 保存标注图片
        if 'annotated_image' in result and result['annotated_image'] is not None:
            output_path = f"./demo_results/detected_{i+1}.jpg"
            cv2.imwrite(output_path, result['annotated_image'])
            print(f"  💾 标注图片已保存: {output_path}")
        
        # 提取IM信息
        im_info = detector.extract_im_info(str(img_path))
        
        print(f"  📋 IM信息提取:")
        print(f"    - 接收者: {'✅ 检测到' if im_info['receiver'] else '❌ 未检测到'}")
        print(f"    - 输入框: {'✅ 检测到' if im_info['input_text'] else '❌ 未检测到'}")
        print(f"    - 联系人: {len(im_info['contacts'])} 个")
        print(f"    - 消息: {len(im_info['messages'])} 条")
    
    return True


def demo_end2end():
    """演示端到端流程"""
    print("\n🚀 演示：端到端流程")
    print("=" * 40)
    
    # 检查图片
    raw_images = list(Path("./data/raw_images").glob("*.jpg"))
    if len(raw_images) < 5:
        print("❌ 需要至少5张IM截图进行端到端演示")
        print("请将更多截图放入 ./data/raw_images/ 文件夹")
        return False
    
    pipeline = End2EndIMDetector()
    
    try:
        print("🎬 开始端到端演示（使用前5张图片）...")
        
        # 快速构建
        model_path = pipeline.quick_build(
            "./data/raw_images", 
            num_images=5  # 只用5张图片演示
        )
        
        print(f"✅ 端到端构建完成！")
        print(f"📄 模型路径: {model_path}")
        
        # 测试检测
        test_images = raw_images[:2]
        print(f"\n🔬 测试新训练的模型...")
        
        for img_path in test_images:
            result = pipeline.predict(str(img_path))
            im_info = pipeline.extract_im_info(str(img_path))
            
            print(f"📸 {img_path.name}:")
            print(f"  接收者: {'✅' if im_info['receiver'] else '❌'}")
            print(f"  输入框: {'✅' if im_info['input_text'] else '❌'}")
        
        # 性能测试
        print(f"\n⚡ 性能测试...")
        metrics = pipeline.benchmark("./data/raw_images", num_images=3)
        print(f"  平均FPS: {metrics['avg_fps']:.2f}")
        print(f"  平均延迟: {metrics['avg_inference_time_ms']:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"❌ 端到端演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主演示函数"""
    print("🎭 IM检测器完整演示")
    print("=" * 50)
    
    # 1. 环境检查
    if not check_requirements():
        print("\n❌ 环境检查失败，请先安装依赖包")
        return
    
    # 2. 创建示例数据结构
    create_sample_data()
    
    print("\n🎯 演示菜单：")
    print("1. 🏷️  GPT-4V自动标注演示")
    print("2. 🚂 YOLO训练演示")
    print("3. 🔍 检测功能演示")
    print("4. 🚀 端到端流程演示")
    print("5. 🎬 完整演示（推荐）")
    
    while True:
        choice = input("\n请选择演示项目 (1-5, q退出): ").strip().lower()
        
        if choice == 'q':
            print("👋 演示结束")
            break
        elif choice == '1':
            demo_auto_labeling()
        elif choice == '2':
            demo_training()
        elif choice == '3':
            demo_detection()
        elif choice == '4':
            demo_end2end()
        elif choice == '5':
            print("\n🎬 开始完整演示流程...")
            success = True
            
            # 按顺序执行所有演示
            if success:
                success = demo_auto_labeling()
            if success:
                success = demo_training()
            if success:
                success = demo_detection()
            
            if success:
                print("\n🎉 完整演示成功完成！")
                print("\n📋 演示总结：")
                print("  ✅ GPT-4V自动标注 - 完成")
                print("  ✅ YOLO模型训练 - 完成")  
                print("  ✅ IM元素检测 - 完成")
                print("\n📁 生成的文件：")
                print("  - 标注数据: ./data/labeled_data/demo/")
                print("  - 训练模型: ./models/demo_model.pt")
                print("  - 检测结果: ./demo_results/")
            else:
                print("\n❌ 演示过程中出现错误")
        else:
            print("❌ 无效选择，请输入1-5或q")


if __name__ == "__main__":
    main()
