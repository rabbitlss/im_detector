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
你是一个专业的计算机视觉标注专家。请分析这张聊天软件界面截图，精确标注每个UI元素。

**标注类别说明：**
1. receiver_avatar: 当前对话对象的头像（通常在顶部，圆形）
2. receiver_name: 当前对话对象的名字（通常在顶部中央）
3. input_box: 底部的文字输入框
4. send_button: 发送按钮（通常在输入框右侧）
5. chat_message: 聊天消息气泡（包括发送和接收的）
6. contact_item: 左侧联系人列表中的每一项
7. user_avatar: 用户自己的头像

**标注要求：**
- 坐标格式：[x1, y1, x2, y2]，左上角到右下角
- 必须区分receiver_avatar和user_avatar
- 必须区分receiver_name和contact_item中的名字
- 每个chat_message都要单独标注
- 坐标要精确到像素

**返回JSON格式：**
```json
{
    "objects": [
        {"class": "receiver_name", "bbox": [150, 20, 280, 45]},
        {"class": "receiver_avatar", "bbox": [100, 15, 140, 55]},
        {"class": "input_box", "bbox": [50, 700, 500, 740]},
        {"class": "send_button", "bbox": [510, 705, 550, 735]},
        {"class": "chat_message", "bbox": [60, 100, 300, 150]},
        {"class": "contact_item", "bbox": [10, 80, 200, 110]},
        {"class": "user_avatar", "bbox": [520, 200, 560, 240]}
    ]
}
```
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
