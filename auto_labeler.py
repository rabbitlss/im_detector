# -*- coding: utf-8 -*-
"""
GPT-4V自动标注器
"""

import base64
import json
import time
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import cv2
from tqdm import tqdm
from openai import OpenAI

from config import CLASS_MAPPING, LABELING_PROMPT, OPENAI_API_KEY


class GPT4VAutoLabeler:
    """GPT-4V自动标注器"""
    
    def __init__(self, api_key: str = None):
        """
        初始化标注器
        
        Args:
            api_key: OpenAI API密钥
        """
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        self.class_mapping = CLASS_MAPPING
        
    def batch_labeling(self, images_folder: str, output_folder: str, 
                      max_images: int = None) -> None:
        """
        批量自动标注
        
        Args:
            images_folder: 原始图片文件夹
            output_folder: 输出标注文件夹
            max_images: 最大处理图片数量
        """
        # 创建输出目录
        os.makedirs(f"{output_folder}/images", exist_ok=True)
        os.makedirs(f"{output_folder}/labels", exist_ok=True)
        
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(images_folder).glob(f"*{ext}"))
            image_files.extend(Path(images_folder).glob(f"*{ext.upper()}"))
        
        if max_images:
            image_files = image_files[:max_images]
            
        print(f"找到 {len(image_files)} 张图片，开始标注...")
        
        success_count = 0
        failed_count = 0
        
        for image_file in tqdm(image_files, desc="标注进度"):
            try:
                # GPT-4V标注
                annotations = self.label_single_image(str(image_file))
                
                if annotations and annotations.get('objects'):
                    # 复制图片到输出目录
                    import shutil
                    output_image = f"{output_folder}/images/{image_file.name}"
                    shutil.copy2(image_file, output_image)
                    
                    # 保存YOLO格式标注
                    self.save_yolo_format(
                        annotations, 
                        str(image_file),
                        f"{output_folder}/labels"
                    )
                    
                    success_count += 1
                    print(f"✅ 标注成功: {image_file.name}")
                else:
                    failed_count += 1
                    print(f"⚠️ 未检测到元素: {image_file.name}")
                
            except Exception as e:
                failed_count += 1
                print(f"❌ 标注失败: {image_file.name}, 错误: {e}")
            
            # 避免API限制
            time.sleep(0.5)
        
        print(f"\n标注完成！成功: {success_count}, 失败: {failed_count}")
    
    def label_single_image(self, image_path: str) -> Optional[Dict]:
        """
        标注单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            标注结果字典
        """
        try:
            # 读取并编码图片
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            
            # 调用GPT-4V API
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": LABELING_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }],
                max_tokens=2000,
                temperature=0.1  # 降低随机性，提高一致性
            )
            
            # 解析响应
            content = response.choices[0].message.content
            return self.parse_gpt4v_response(content)
            
        except Exception as e:
            print(f"API调用失败: {e}")
            return None
    
    def parse_gpt4v_response(self, response_text: str) -> Optional[Dict]:
        """
        解析GPT-4V的响应
        
        Args:
            response_text: GPT-4V的响应文本
            
        Returns:
            解析后的标注数据
        """
        try:
            # 方法1: 直接解析JSON
            if '```json' in response_text:
                json_part = response_text.split('```json')[1].split('```')[0].strip()
                return json.loads(json_part)
            elif '```' in response_text:
                json_part = response_text.split('```')[1].strip()
                return json.loads(json_part)
            
            # 方法2: 尝试从整个文本解析JSON
            json_match = re.search(r'\{.*"objects".*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # 方法3: 用正则表达式提取坐标信息
            return self.regex_extract_annotations(response_text)
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return self.regex_extract_annotations(response_text)
        except Exception as e:
            print(f"响应解析失败: {e}")
            return None
    
    def regex_extract_annotations(self, text: str) -> Dict:
        """
        用正则表达式从文本中提取标注信息
        
        Args:
            text: 响应文本
            
        Returns:
            提取的标注数据
        """
        objects = []
        
        # 匹配形如 "receiver_name": [150, 20, 280, 45] 的模式
        pattern = r'["\']?(\w+)["\']?\s*[:\s]\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        matches = re.findall(pattern, text)
        
        for match in matches:
            class_name, x1, y1, x2, y2 = match
            if class_name in self.class_mapping:
                objects.append({
                    "class": class_name,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
        
        return {"objects": objects}
    
    def save_yolo_format(self, annotations: Dict, image_path: str, 
                        output_folder: str) -> None:
        """
        保存为YOLO格式
        
        Args:
            annotations: 标注数据
            image_path: 原图片路径
            output_folder: 输出文件夹
        """
        # 获取图片尺寸
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            return
            
        h, w = img.shape[:2]
        
        # 转换坐标格式
        yolo_lines = []
        for obj in annotations.get('objects', []):
            class_name = obj['class']
            if class_name not in self.class_mapping:
                continue
                
            class_id = self.class_mapping[class_name]
            bbox = obj['bbox']
            
            # 验证坐标
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = bbox
            
            # 坐标范围检查
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(x1, min(w, x2))
            y2 = max(y1, min(h, y2))
            
            # 转换为YOLO格式（归一化的中心点坐标）
            center_x = (x1 + x2) / 2 / w
            center_y = (y1 + y2) / 2 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            # 过滤异常小的框
            if width > 0.01 and height > 0.01:
                yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        # 保存标注文件
        base_name = Path(image_path).stem
        label_path = os.path.join(output_folder, f"{base_name}.txt")
        
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
    
    def validate_annotations(self, image_path: str, label_path: str) -> bool:
        """
        验证标注文件的有效性
        
        Args:
            image_path: 图片路径
            label_path: 标注文件路径
            
        Returns:
            是否有效
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
                
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    return False
                
                class_id = int(parts[0])
                if class_id >= len(self.class_mapping):
                    return False
                
                # 检查坐标范围
                coords = [float(x) for x in parts[1:]]
                if not all(0 <= x <= 1 for x in coords):
                    return False
            
            return True
            
        except Exception:
            return False


def main():
    """测试自动标注功能"""
    
    # 初始化标注器
    labeler = GPT4VAutoLabeler()
    
    # 批量标注
    labeler.batch_labeling(
        images_folder='./data/raw_images',
        output_folder='./data/labeled_data/train',
        max_images=10  # 测试时只处理10张图片
    )


if __name__ == "__main__":
    main()
