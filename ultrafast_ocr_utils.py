# -*- coding: utf-8 -*-
"""
工具函数模块
"""

import os
import cv2
import numpy as np
import subprocess
import urllib.request
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict


def validate_image(image: np.ndarray) -> bool:
    """
    验证图片输入
    
    Args:
        image: 输入图片
        
    Returns:
        是否有效
    """
    if image is None:
        return False
    
    if not isinstance(image, np.ndarray):
        return False
    
    if len(image.shape) < 2 or len(image.shape) > 3:
        return False
    
    if image.size == 0:
        return False
    
    # 检查尺寸合理性
    h, w = image.shape[:2]
    if h < 8 or w < 8 or h > 4096 or w > 4096:
        return False
    
    return True


def get_default_model_path(model_type: str) -> str:
    """
    获取默认模型路径
    
    Args:
        model_type: 模型类型 ('det', 'rec', 'dict')
        
    Returns:
        模型文件路径
    """
    base_path = Path(__file__).parent.parent / "models" / "ocr"
    
    paths = {
        'det': base_path / "ch_PP-OCRv4_det.onnx",
        'rec': base_path / "ch_PP-OCRv4_rec.onnx", 
        'dict': base_path / "ppocr_keys_v1.txt"
    }
    
    return str(paths.get(model_type, ""))


def check_models() -> Dict[str, bool]:
    """
    检查模型文件是否存在
    
    Returns:
        模型存在状态字典
    """
    status = {}
    
    for model_type in ['det', 'rec', 'dict']:
        path = get_default_model_path(model_type)
        status[model_type] = os.path.exists(path)
    
    return status


def download_models(force: bool = False) -> bool:
    """
    下载OCR模型
    
    Args:
        force: 是否强制重新下载
        
    Returns:
        是否下载成功
    """
    model_dir = Path(__file__).parent.parent / "models" / "ocr"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 开始下载OCR模型...")
    
    # 模型URLs
    model_urls = {
        'det_paddle': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar',
        'rec_paddle': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar',
        'dict': 'https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt'
    }
    
    success = True
    
    try:
        # 1. 下载字典文件
        dict_path = model_dir / "ppocr_keys_v1.txt"
        if force or not dict_path.exists():
            print("下载字符字典...")
            urllib.request.urlretrieve(model_urls['dict'], str(dict_path))
            print(f"✅ 字典下载完成: {dict_path}")
        
        # 2. 下载并转换检测模型
        det_onnx_path = model_dir / "ch_PP-OCRv4_det.onnx"
        if force or not det_onnx_path.exists():
            print("下载检测模型...")
            det_tar_path = model_dir / "ch_PP-OCRv4_det_infer.tar"
            
            # 下载tar文件
            urllib.request.urlretrieve(model_urls['det_paddle'], str(det_tar_path))
            
            # 解压
            subprocess.run(['tar', '-xf', str(det_tar_path), '-C', str(model_dir)], 
                         check=True, capture_output=True)
            
            # 转换为ONNX
            if _convert_to_onnx(model_dir / "ch_PP-OCRv4_det_infer", str(det_onnx_path)):
                print(f"✅ 检测模型转换完成: {det_onnx_path}")
                # 清理临时文件
                det_tar_path.unlink(missing_ok=True)
                _remove_directory(model_dir / "ch_PP-OCRv4_det_infer")
            else:
                success = False
        
        # 3. 下载并转换识别模型
        rec_onnx_path = model_dir / "ch_PP-OCRv4_rec.onnx"
        if force or not rec_onnx_path.exists():
            print("下载识别模型...")
            rec_tar_path = model_dir / "ch_PP-OCRv4_rec_infer.tar"
            
            # 下载tar文件
            urllib.request.urlretrieve(model_urls['rec_paddle'], str(rec_tar_path))
            
            # 解压
            subprocess.run(['tar', '-xf', str(rec_tar_path), '-C', str(model_dir)], 
                         check=True, capture_output=True)
            
            # 转换为ONNX
            if _convert_to_onnx(model_dir / "ch_PP-OCRv4_rec_infer", str(rec_onnx_path)):
                print(f"✅ 识别模型转换完成: {rec_onnx_path}")
                # 清理临时文件
                rec_tar_path.unlink(missing_ok=True)
                _remove_directory(model_dir / "ch_PP-OCRv4_rec_infer")
            else:
                success = False
                
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        success = False
    
    if success:
        print("🎉 所有模型下载完成!")
        # 验证模型
        status = check_models()
        print("模型状态:")
        for model_type, exists in status.items():
            print(f"  {model_type}: {'✅' if exists else '❌'}")
    
    return success


def _convert_to_onnx(paddle_model_dir: Path, onnx_path: str) -> bool:
    """
    转换PaddlePaddle模型为ONNX格式
    
    Args:
        paddle_model_dir: PaddlePaddle模型目录
        onnx_path: 输出ONNX路径
        
    Returns:
        是否转换成功
    """
    try:
        # 检查paddle2onnx是否安装
        try:
            import paddle2onnx
        except ImportError:
            print("安装paddle2onnx...")
            subprocess.run(['pip', 'install', 'paddle2onnx'], check=True)
            import paddle2onnx
        
        # 转换命令
        cmd = [
            'paddle2onnx',
            '--model_dir', str(paddle_model_dir),
            '--model_filename', 'inference.pdmodel',
            '--params_filename', 'inference.pdiparams',
            '--save_file', onnx_path,
            '--opset_version', '11'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            print(f"转换失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"ONNX转换错误: {e}")
        return False


def _remove_directory(path: Path):
    """安全删除目录"""
    try:
        import shutil
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
    except Exception:
        pass


def create_test_image(text: str, 
                     width: int = 200, 
                     height: int = 60,
                     font_scale: float = 1.0,
                     bg_color: Tuple[int, int, int] = (255, 255, 255),
                     text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    创建测试图片
    
    Args:
        text: 要绘制的文字
        width: 图片宽度
        height: 图片高度
        font_scale: 字体大小
        bg_color: 背景色 (B, G, R)
        text_color: 文字颜色 (B, G, R)
        
    Returns:
        生成的图片
    """
    # 创建背景
    image = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    # 设置字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # 计算文字尺寸
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # 居中位置
    x = (width - text_width) // 2
    y = (height + text_height) // 2
    
    # 绘制文字
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)
    
    return image


def batch_create_test_images(texts: list, **kwargs) -> list:
    """批量创建测试图片"""
    return [create_test_image(text, **kwargs) for text in texts]


def calculate_image_hash(image: np.ndarray, hash_size: int = 8) -> str:
    """
    计算图片感知哈希
    
    Args:
        image: 输入图片
        hash_size: 哈希尺寸
        
    Returns:
        哈希字符串
    """
    try:
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 缩放
        resized = cv2.resize(gray, (hash_size, hash_size))
        
        # 计算平均值
        avg = resized.mean()
        
        # 生成哈希
        hash_bits = []
        for i in range(hash_size):
            for j in range(hash_size):
                hash_bits.append('1' if resized[i, j] > avg else '0')
        
        return ''.join(hash_bits)
        
    except Exception as e:
        # 备用方案
        return hashlib.md5(image.tobytes()).hexdigest()[:16]


def hamming_distance(hash1: str, hash2: str) -> int:
    """计算汉明距离"""
    if len(hash1) != len(hash2):
        raise ValueError("哈希长度必须相同")
    
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def image_similarity(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    计算图片相似度
    
    Args:
        image1, image2: 要比较的图片
        
    Returns:
        相似度 (0-1, 1表示完全相同)
    """
    try:
        hash1 = calculate_image_hash(image1)
        hash2 = calculate_image_hash(image2)
        
        distance = hamming_distance(hash1, hash2)
        max_distance = len(hash1)
        
        similarity = 1 - (distance / max_distance)
        return similarity
        
    except Exception:
        return 0.0


def benchmark_ocr_engine(ocr_func, 
                        test_images: list,
                        rounds: int = 100,
                        warmup_rounds: int = 3) -> Dict:
    """
    OCR引擎基准测试
    
    Args:
        ocr_func: OCR函数
        test_images: 测试图片列表
        rounds: 测试轮数
        warmup_rounds: 预热轮数
        
    Returns:
        性能统计
    """
    import time
    
    if not test_images:
        return {}
    
    # 预热
    for _ in range(warmup_rounds):
        for img in test_images:
            try:
                _ = ocr_func(img)
            except:
                pass
    
    # 测试
    times = []
    successful_calls = 0
    
    for round_idx in range(rounds):
        for img in test_images:
            start = time.time()
            try:
                result = ocr_func(img)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
                successful_calls += 1
            except Exception as e:
                # 记录失败，但继续测试
                pass
    
    if not times:
        return {'error': 'All calls failed'}
    
    times = np.array(times)
    
    return {
        'total_calls': len(test_images) * rounds,
        'successful_calls': successful_calls,
        'success_rate': successful_calls / (len(test_images) * rounds),
        'avg_time_ms': float(np.mean(times)),
        'min_time_ms': float(np.min(times)),
        'max_time_ms': float(np.max(times)),
        'std_time_ms': float(np.std(times)),
        'median_time_ms': float(np.median(times)),
        'p95_time_ms': float(np.percentile(times, 95)),
        'p99_time_ms': float(np.percentile(times, 99)),
        'fps': float(1000 / np.mean(times)),
        'throughput_per_second': float(successful_calls / (np.sum(times) / 1000))
    }


def setup_logging(log_level: str = 'INFO'):
    """设置日志"""
    import logging
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ultrafast_ocr.log')
        ]
    )
    
    return logging.getLogger('ultrafast_ocr')


def get_system_info() -> Dict:
    """获取系统信息"""
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
    }
    
    # GPU信息
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        info['onnx_providers'] = providers
        info['gpu_available'] = 'CUDAExecutionProvider' in providers
    except:
        info['gpu_available'] = False
    
    return info


if __name__ == "__main__":
    # 测试工具函数
    print("🔧 工具函数测试")
    print("=" * 50)
    
    # 创建测试图片
    test_img = create_test_image("Hello OCR", width=200, height=60)
    print(f"创建测试图片: {test_img.shape}")
    
    # 计算哈希
    img_hash = calculate_image_hash(test_img)
    print(f"图片哈希: {img_hash}")
    
    # 检查模型
    model_status = check_models()
    print(f"模型状态: {model_status}")
    
    # 系统信息
    sys_info = get_system_info()
    print(f"系统信息: {sys_info}")
    
    print("✅ 工具函数测试完成")
