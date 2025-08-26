# -*- coding: utf-8 -*-
"""
UltraFast OCR Package
超快速OCR包 - 基于ONNX直接推理实现3-10ms识别速度
"""

__version__ = "1.0.0"
__author__ = "IM Detector Team"

from .core import UltraFastOCR
from .optimized import OptimizedOCR
from .utils import download_models, check_models

__all__ = [
    'UltraFastOCR',
    'OptimizedOCR',
    'download_models',
    'check_models'
]

# 包级别的便捷函数
def quick_ocr(image_region, model_path=None):
    """
    快速OCR识别
    
    Args:
        image_region: 图片区域(numpy array)
        model_path: 模型路径(可选)
    
    Returns:
        识别的文字
    """
    if not hasattr(quick_ocr, 'ocr_instance'):
        if model_path:
            quick_ocr.ocr_instance = UltraFastOCR(rec_model_path=model_path)
        else:
            quick_ocr.ocr_instance = UltraFastOCR()
    
    text, _, _ = quick_ocr.ocr_instance.recognize_single_line(image_region)
    return text
