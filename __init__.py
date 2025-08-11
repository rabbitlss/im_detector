# -*- coding: utf-8 -*-
"""
IM界面检测器包

基于GPT-4V自动标注和YOLO训练的IM聊天界面元素检测系统
"""

__version__ = "1.0.0"
__author__ = "IM Detector Team"
__email__ = "contact@example.com"

from .im_detector import IMDetector
from .auto_labeler import GPT4VAutoLabeler
from .yolo_trainer import YOLOTrainer
from .end2end_pipeline import End2EndIMDetector

__all__ = [
    'IMDetector',
    'GPT4VAutoLabeler', 
    'YOLOTrainer',
    'End2EndIMDetector'
]
