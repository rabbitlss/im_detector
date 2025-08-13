# -*- coding: utf-8 -*-
"""
IM检测器配置文件
"""

import os

# API配置

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-api-key-here')

# 类别映射
CLASS_MAPPING = {
    'receiver_avatar': 0,
    'receiver_name': 1,
    'input_box': 2,
    'send_button': 3,
    'chat_message': 4,
    'contact_item': 5,
    'user_avatar': 6
}

# 类别名称列表
CLASS_NAMES = [
    'receiver_avatar',
    'receiver_name', 
    'input_box',
    'send_button',
    'chat_message',
    'contact_item',
    'user_avatar'
]

# GPT-4V标注提示词
LABELING_PROMPT = """
你是一个专业的UI界面标注专家。请仔细分析这张IM聊天界面截图。

## 第一步：界面布局识别
先识别整体布局结构：
0、有可能对话框是在一个大的图片背景下的，它不能占据整个图片
1. 顶部栏（标题栏）：包含对话对象信息
2. 中间区域：消息显示区
3. 底部栏：输入区域
4. 左侧栏（如果有）：联系人列表

## 第二步：精确标注规则

### receiver_avatar（接收者头像）
- 位置：通常在顶部栏左侧或中间
- 特征：圆形或方形，尺寸30-60像素
- 注意：只标注当前对话对象的头像，不是消息中的头像

### receiver_name（接收者姓名）
- 位置：顶部栏中央或头像右侧
- 特征：较大的文字，通常是标题样式
- 注意：不要与消息中的名字混淆

### input_box（输入框）
- 位置：底部区域
- 特征：白色或浅色背景的矩形框
- 注意：可能包含提示文字如"请输入消息"

### send_button（发送按钮）
- 位置：输入框右侧或下方
- 特征：按钮样式，可能显示"发送"或箭头图标
- 注意：有时可能是图标而非文字

### chat_message（聊天消息）
- 位置：中间消息区域
- 特征：气泡状或卡片状
- 注意：每条消息单独标注，包括文字和背景气泡

### contact_item（联系人项）
- 位置：左侧列表区域
- 特征：包含头像+名字的横条
- 注意：每个联系人单独标注整个条目

### user_avatar（用户头像）
- 位置：用户发送的消息旁边
- 特征：通常在消息右侧
- 注意：与receiver_avatar区分

## 第三步：输出要求
1. 坐标必须是整数
2. 使用[x1, y1, x2, y2]格式（左上到右下）
3. 确保坐标不超出图片边界
4. 相同类型的元素按从上到下顺序标注

## 输出JSON格式：
{
    "analysis": "简要描述界面布局",
    "objects": [
        {"class": "类别名", "bbox": [x1, y1, x2, y2], "confidence": 0.95}
    ]
}
"""

# 训练配置
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 16,
    'img_size': 640,
    'patience': 20,
    'workers': 8,
    'device': 'cuda'  # 或 'cpu'
}

# 文件路径配置
PATHS = {
    'raw_images': './data/raw_images',
    'labeled_data': './data/labeled_data',
    'train_images': './data/labeled_data/train/images',
    'train_labels': './data/labeled_data/train/labels',
    'val_images': './data/labeled_data/val/images',
    'val_labels': './data/labeled_data/val/labels',
    'models': './models',
    'results': './results'
}
