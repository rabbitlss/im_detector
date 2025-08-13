# -*- coding: utf-8 -*-
"""
人工辅助验证器 - 完整实现
"""

import os
import cv2
import json
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime


class HumanAssistedValidator:
    """人工辅助验证器"""
    
    def __init__(self, interactive_mode: bool = True, auto_save: bool = True):
        """
        初始化验证器
        
        Args:
            interactive_mode: 是否启用交互式验证界面
            auto_save: 是否自动保存验证结果
        """
        self.interactive_mode = interactive_mode
        self.auto_save = auto_save
        
        # 类别颜色配置
        self.colors = {
            'receiver_name': (0, 255, 0),      # 绿色
            'receiver_avatar': (255, 0, 0),    # 蓝色  
            'input_box': (0, 0, 255),          # 红色
            'send_button': (255, 255, 0),      # 青色
            'chat_message': (255, 0, 255),     # 紫色
            'contact_item': (0, 255, 255),     # 黄色
            'user_avatar': (128, 0, 128)       # 深紫色
        }
        
        # 验证历史
        self.validation_history = []
        
        # 统计信息
        self.stats = {
            'total_validated': 0,
            'total_corrected': 0,
            'class_accuracy': {},
            'common_errors': []
        }
    
    def validate_with_visualization(self, image_path: str, labels: List[Dict], 
                                  output_dir: str = "./validation_results") -> Dict:
        """
        生成可视化结果供人工验证
        
        Args:
            image_path: 图片路径
            labels: 标注列表
            output_dir: 输出目录
            
        Returns:
            验证结果字典
        """
        print(f"🔍 开始验证图片: {os.path.basename(image_path)}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 1. 生成可视化图片
        viz_results = self._create_visualizations(image, labels, image_path, output_dir)
        
        # 2. 生成验证报告
        validation_report = self._generate_validation_report(labels, image.shape, image_path)
        
        # 3. 保存报告
        report_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, ensure_ascii=False, indent=2)
        
        # 4. 判断是否需要人工审查
        needs_review, review_reason = self._needs_review(labels)
        
        # 5. 交互式验证（如果启用）
        correction_result = None
        if self.interactive_mode and needs_review:
            print("⚠️ 检测到问题，启动交互式验证...")
            correction_result = self._launch_interactive_validation(image_path, labels)
        
        # 6. 汇总结果
        result = {
            'image_path': image_path,
            'visualization_paths': viz_results,
            'validation_report': validation_report,
            'report_path': report_path,
            'needs_human_review': needs_review,
            'review_reason': review_reason,
            'correction_result': correction_result,
            'timestamp': datetime.now().isoformat()
        }
        
        # 7. 保存到历史记录
        self.validation_history.append(result)
        
        # 8. 更新统计
        self._update_stats(labels, correction_result)
        
        if self.auto_save:
            self._save_validation_session(output_dir)
        
        return result
    
    def _create_visualizations(self, image: np.ndarray, labels: List[Dict], 
                             image_path: str, output_dir: str) -> Dict[str, str]:
        """创建多种可视化图片"""
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        viz_paths = {}
        
        # 1. 基础标注图
        labeled_image = self._draw_labels_opencv(image.copy(), labels)
        basic_path = os.path.join(output_dir, f"{base_name}_labeled.jpg")
        cv2.imwrite(basic_path, labeled_image)
        viz_paths['basic_labeled'] = basic_path
        
        # 2. 详细信息图
        detailed_image = self._draw_detailed_labels(image.copy(), labels)
        detailed_path = os.path.join(output_dir, f"{base_name}_detailed.jpg")
        cv2.imwrite(detailed_path, detailed_image)
        viz_paths['detailed_labeled'] = detailed_path
        
        # 3. 分类别显示图
        class_images = self._draw_class_separated(image, labels)
        class_paths = {}
        for class_name, class_image in class_images.items():
            class_path = os.path.join(output_dir, f"{base_name}_{class_name}.jpg")
            cv2.imwrite(class_path, class_image)
            class_paths[class_name] = class_path
        viz_paths['class_separated'] = class_paths
        
        # 4. 问题高亮图
        issues_image = self._highlight_issues(image.copy(), labels)
        issues_path = os.path.join(output_dir, f"{base_name}_issues.jpg")
        cv2.imwrite(issues_path, issues_image)
        viz_paths['issues_highlighted'] = issues_path
        
        # 5. Matplotlib版本（更美观）
        plt_path = self._create_matplotlib_visualization(image, labels, output_dir, base_name)
        viz_paths['matplotlib'] = plt_path
        
        return viz_paths
    
    def _draw_labels_opencv(self, image: np.ndarray, labels: List[Dict]) -> np.ndarray:
        """使用OpenCV绘制基础标注"""
        
        for i, label in enumerate(labels):
            class_name = label['class']
            bbox = [int(x) for x in label['bbox']]
            x1, y1, x2, y2 = bbox
            
            color = self.colors.get(class_name, (128, 128, 128))
            
            # 画边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 画标签背景
            label_text = f"{class_name}_{i+1}"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0] + 10, y1), color, -1)
            
            # 画标签文字
            cv2.putText(image, label_text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 画中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(image, (center_x, center_y), 3, color, -1)
        
        return image
    
    def _draw_detailed_labels(self, image: np.ndarray, labels: List[Dict]) -> np.ndarray:
        """绘制详细信息标注"""
        
        for i, label in enumerate(labels):
            class_name = label['class']
            bbox = [int(x) for x in label['bbox']]
            x1, y1, x2, y2 = bbox
            
            color = self.colors.get(class_name, (128, 128, 128))
            
            # 画边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 计算详细信息
            width = x2 - x1
            height = y2 - y1
            area = width * height
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # 详细标签信息
            details = [
                f"{class_name} #{i+1}",
                f"Size: {width}x{height}",
                f"Area: {area}px",
                f"Center: {center}"
            ]
            
            # 绘制详细信息
            y_offset = y1 - 80
            for j, detail in enumerate(details):
                text_y = y_offset + j * 20
                text_size = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # 背景
                cv2.rectangle(image, (x1, text_y - 15), 
                             (x1 + text_size[0] + 10, text_y + 5), color, -1)
                
                # 文字
                cv2.putText(image, detail, (x1 + 5, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def _draw_class_separated(self, image: np.ndarray, labels: List[Dict]) -> Dict[str, np.ndarray]:
        """为每个类别单独绘制图片"""
        
        class_images = {}
        
        # 按类别分组
        class_groups = {}
        for label in labels:
            class_name = label['class']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(label)
        
        # 为每个类别创建图片
        for class_name, class_labels in class_groups.items():
            class_image = image.copy()
            
            # 添加类别标题
            title = f"Class: {class_name} (Count: {len(class_labels)})"
            cv2.putText(class_image, title, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 绘制该类别的所有标注
            for i, label in enumerate(class_labels):
                bbox = [int(x) for x in label['bbox']]
                x1, y1, x2, y2 = bbox
                color = self.colors.get(class_name, (128, 128, 128))
                
                # 高亮显示
                cv2.rectangle(class_image, (x1, y1), (x2, y2), color, 3)
                
                # 编号
                cv2.putText(class_image, str(i+1), (x1+5, y1+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            class_images[class_name] = class_image
        
        return class_images
    
    def _highlight_issues(self, image: np.ndarray, labels: List[Dict]) -> np.ndarray:
        """高亮显示可能的问题"""
        
        issues_found = []
        
        # 检查重复元素
        class_counts = {}
        for label in labels:
            class_name = label['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        unique_classes = ['receiver_name', 'receiver_avatar', 'input_box', 'send_button']
        
        for i, label in enumerate(labels):
            class_name = label['class']
            bbox = [int(x) for x in label['bbox']]
            x1, y1, x2, y2 = bbox
            
            has_issue = False
            issue_reasons = []
            
            # 检查是否为重复的唯一元素
            if class_name in unique_classes and class_counts[class_name] > 1:
                has_issue = True
                issue_reasons.append("重复")
            
            # 检查尺寸异常
            width = x2 - x1
            height = y2 - y1
            
            if width <= 0 or height <= 0:
                has_issue = True
                issue_reasons.append("无效尺寸")
            elif width < 10 or height < 10:
                has_issue = True
                issue_reasons.append("过小")
            elif width > image.shape[1] * 0.8 or height > image.shape[0] * 0.8:
                has_issue = True
                issue_reasons.append("过大")
            
            # 检查坐标超界
            h, w = image.shape[:2]
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                has_issue = True
                issue_reasons.append("超界")
            
            if has_issue:
                # 用红色粗框标出问题
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
                
                # 标注问题原因
                issue_text = f"问题: {', '.join(issue_reasons)}"
                cv2.putText(image, issue_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                issues_found.append({
                    'class': class_name,
                    'bbox': bbox,
                    'issues': issue_reasons
                })
        
        # 添加总结信息
        summary = f"Found {len(issues_found)} issues"
        cv2.putText(image, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return image
    
    def _create_matplotlib_visualization(self, image: np.ndarray, labels: List[Dict], 
                                       output_dir: str, base_name: str) -> str:
        """创建Matplotlib版本的可视化（更美观）"""
        
        # 转换BGR到RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(rgb_image)
        
        # 颜色映射（转换为matplotlib格式）
        mpl_colors = {
            'receiver_name': 'green',
            'receiver_avatar': 'blue',
            'input_box': 'red',
            'send_button': 'cyan',
            'chat_message': 'magenta',
            'contact_item': 'yellow',
            'user_avatar': 'purple'
        }
        
        # 绘制边界框
        for i, label in enumerate(labels):
            class_name = label['class']
            bbox = label['bbox']
            x1, y1, x2, y2 = bbox
            
            width = x2 - x1
            height = y2 - y1
            
            color = mpl_colors.get(class_name, 'gray')
            
            # 创建矩形
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor=color, 
                                   facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # 添加标签
            ax.text(x1, y1-5, f"{class_name}_{i+1}", 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                   fontsize=10, color='white', weight='bold')
        
        ax.set_title(f"IM Elements Detection: {base_name}", fontsize=14, weight='bold')
        ax.axis('off')
        
        # 添加图例
        legend_elements = [patches.Patch(color=color, label=class_name) 
                          for class_name, color in mpl_colors.items() 
                          if any(label['class'] == class_name for label in labels)]
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        # 保存
        plt_path = os.path.join(output_dir, f"{base_name}_matplotlib.png")
        plt.savefig(plt_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plt_path
    
    def _generate_validation_report(self, labels: List[Dict], image_shape: Tuple, 
                                  image_path: str) -> Dict:
        """生成详细的验证报告"""
        
        h, w = image_shape[:2]
        
        report = {
            'image_info': {
                'path': image_path,
                'size': f"{w}x{h}",
                'total_labels': len(labels)
            },
            'class_distribution': {},
            'potential_issues': [],
            'statistics': {},
            'recommendations': []
        }
        
        # 统计类别分布
        for label in labels:
            class_name = label['class']
            report['class_distribution'][class_name] = report['class_distribution'].get(class_name, 0) + 1
        
        # 检查潜在问题
        unique_classes = ['receiver_name', 'receiver_avatar', 'input_box', 'send_button']
        
        for class_name, count in report['class_distribution'].items():
            if class_name in unique_classes and count > 1:
                report['potential_issues'].append({
                    'type': 'duplicate_unique_element',
                    'class': class_name,
                    'count': count,
                    'severity': 'high'
                })
        
        # 检查缺失的重要元素
        required_elements = ['receiver_name', 'input_box']
        for required in required_elements:
            if required not in report['class_distribution']:
                report['potential_issues'].append({
                    'type': 'missing_required_element',
                    'class': required,
                    'severity': 'medium'
                })
        
        # 统计信息
        if labels:
            areas = [(label['bbox'][2] - label['bbox'][0]) * (label['bbox'][3] - label['bbox'][1]) 
                    for label in labels]
            
            report['statistics'] = {
                'avg_area': np.mean(areas),
                'min_area': np.min(areas),
                'max_area': np.max(areas),
                'area_std': np.std(areas)
            }
        
        # 生成建议
        if len(report['potential_issues']) == 0:
            report['recommendations'].append("标注看起来正常，建议进行最终确认")
        else:
            report['recommendations'].append("发现潜在问题，建议人工验证")
            
            if any(issue['severity'] == 'high' for issue in report['potential_issues']):
                report['recommendations'].append("存在高严重性问题，强烈建议修正")
        
        return report
    
    def _needs_review(self, labels: List[Dict]) -> Tuple[bool, str]:
        """判断是否需要人工审查"""
        
        issues = []
        
        # 1. 检查重要元素缺失
        required_elements = ['receiver_name', 'input_box']
        detected_classes = [label['class'] for label in labels]
        missing_elements = [elem for elem in required_elements if elem not in detected_classes]
        
        if missing_elements:
            issues.append(f"缺失重要元素: {missing_elements}")
        
        # 2. 检查重复的唯一元素
        unique_elements = ['receiver_name', 'receiver_avatar', 'input_box', 'send_button']
        class_counts = {}
        for label in labels:
            class_name = label['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        duplicates = [k for k, v in class_counts.items() 
                     if k in unique_elements and v > 1]
        
        if duplicates:
            issues.append(f"重复的唯一元素: {duplicates}")
        
        # 3. 检查异常尺寸
        size_issues = []
        for label in labels:
            bbox = label['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            if width <= 0 or height <= 0:
                size_issues.append(f"{label['class']}: 无效尺寸")
            elif width < 5 or height < 5:
                size_issues.append(f"{label['class']}: 尺寸过小")
        
        if size_issues:
            issues.append(f"尺寸异常: {size_issues}")
        
        # 4. 检查标注数量异常
        if len(labels) == 0:
            issues.append("未检测到任何元素")
        elif len(labels) > 20:
            issues.append("检测到过多元素，可能存在误检")
        
        needs_review = len(issues) > 0
        reason = "; ".join(issues) if issues else "看起来正常"
        
        return needs_review, reason
    
    def _launch_interactive_validation(self, image_path: str, labels: List[Dict]) -> Optional[Dict]:
        """启动交互式验证界面"""
        
        try:
            validator_gui = InteractiveValidatorGUI(image_path, labels, self.colors)
            result = validator_gui.run_validation()
            return result
        except Exception as e:
            print(f"交互式验证启动失败: {e}")
            return None
    
    def _update_stats(self, original_labels: List[Dict], correction_result: Optional[Dict]) -> None:
        """更新统计信息"""
        
        self.stats['total_validated'] += 1
        
        if correction_result and correction_result.get('corrected'):
            self.stats['total_corrected'] += 1
            
            # 统计各类别的准确率
            corrected_classes = correction_result.get('corrected_classes', [])
            for class_name in corrected_classes:
                if class_name not in self.stats['class_accuracy']:
                    self.stats['class_accuracy'][class_name] = {'total': 0, 'errors': 0}
                
                self.stats['class_accuracy'][class_name]['total'] += 1
                self.stats['class_accuracy'][class_name]['errors'] += 1
    
    def _save_validation_session(self, output_dir: str) -> None:
        """保存验证会话"""
        
        session_data = {
            'validation_history': self.validation_history,
            'statistics': self.stats,
            'session_info': {
                'total_images': len(self.validation_history),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        session_path = os.path.join(output_dir, 'validation_session.json')
        with open(session_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        print(f"📊 验证会话已保存: {session_path}")
    
    def get_validation_statistics(self) -> Dict:
        """获取验证统计信息"""
        
        if self.stats['total_validated'] == 0:
            return {'message': '暂无验证数据'}
        
        # 计算准确率
        accuracy_by_class = {}
        for class_name, data in self.stats['class_accuracy'].items():
            if data['total'] > 0:
                accuracy = (data['total'] - data['errors']) / data['total']
                accuracy_by_class[class_name] = f"{accuracy:.2%}"
        
        overall_accuracy = 1 - (self.stats['total_corrected'] / self.stats['total_validated'])
        
        return {
            'total_validated': self.stats['total_validated'],
            'total_corrected': self.stats['total_corrected'],
            'overall_accuracy': f"{overall_accuracy:.2%}",
            'accuracy_by_class': accuracy_by_class,
            'correction_rate': f"{self.stats['total_corrected'] / self.stats['total_validated']:.2%}"
        }
    
    def batch_validate(self, image_folder: str, labels_folder: str) -> Dict:
        """批量验证"""
        
        print(f"🔄 开始批量验证...")
        print(f"图片文件夹: {image_folder}")
        print(f"标注文件夹: {labels_folder}")
        
        # 获取所有图片文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([f for f in os.listdir(image_folder) 
                              if f.lower().endswith(ext)])
        
        if not image_files:
            return {'error': '未找到图片文件'}
        
        batch_results = []
        
        for image_file in image_files:
            try:
                image_path = os.path.join(image_folder, image_file)
                
                # 查找对应的标注文件
                base_name = os.path.splitext(image_file)[0]
                label_file = os.path.join(labels_folder, f"{base_name}.json")
                
                if not os.path.exists(label_file):
                    print(f"⚠️ 未找到标注文件: {label_file}")
                    continue
                
                # 读取标注
                with open(label_file, 'r', encoding='utf-8') as f:
                    labels_data = json.load(f)
                
                labels = labels_data.get('objects', [])
                
                # 验证
                result = self.validate_with_visualization(image_path, labels)
                batch_results.append(result)
                
                print(f"✅ 已验证: {image_file}")
                
            except Exception as e:
                print(f"❌ 验证失败 {image_file}: {e}")
                batch_results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        # 生成批量报告
        batch_summary = {
            'total_processed': len(batch_results),
            'successful': len([r for r in batch_results if 'error' not in r]),
            'failed': len([r for r in batch_results if 'error' in r]),
            'needs_review': len([r for r in batch_results if r.get('needs_human_review', False)]),
            'results': batch_results,
            'statistics': self.get_validation_statistics()
        }
        
        return batch_summary


class InteractiveValidatorGUI:
    """交互式验证GUI界面"""
    
    def __init__(self, image_path: str, labels: List[Dict], colors: Dict):
        """
        初始化GUI
        
        Args:
            image_path: 图片路径
            labels: 原始标注
            colors: 颜色配置
        """
        self.image_path = image_path
        self.original_labels = labels.copy()
        self.current_labels = labels.copy()
        self.colors = colors
        
        # GUI组件
        self.root = None
        self.canvas = None
        self.image_label = None
        self.image_tk = None
        
        # 交互状态
        self.selected_label_index = None
        self.editing_mode = False
        self.result = {'corrected': False, 'final_labels': labels}
        
        # 缩放参数
        self.scale_factor = 1.0
        self.max_display_size = (800, 600)
    
    def run_validation(self) -> Dict:
        """运行交互式验证"""
        
        self.root = tk.Tk()
        self.root.title(f"人工验证 - {os.path.basename(self.image_path)}")
        self.root.geometry("1200x800")
        
        try:
            self._setup_gui()
            self._load_and_display_image()
            
            # 运行GUI主循环
            self.root.mainloop()
            
        except Exception as e:
            print(f"GUI运行错误: {e}")
            if self.root:
                self.root.destroy()
        
        return self.result
    
    def _setup_gui(self):
        """设置GUI界面"""
        
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padding=10)
        
        # 左侧：图片显示区
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='both', expand=True)
        
        # 图片标签
        self.image_label = ttk.Label(left_frame)
        self.image_label.pack(pady=10)
        
        # 右侧：控制面板
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side='right', fill='y', padx=(10, 0))
        right_frame.pack_propagate(False)
        
        self._setup_control_panel(right_frame)
        
        # 底部：状态栏
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', side='bottom')
        
        self.status_label = ttk.Label(status_frame, text="准备就绪")
        self.status_label.pack(side='left', padx=10, pady=5)
    
    def _setup_control_panel(self, parent):
        """设置控制面板"""
        
        # 标题
        title_label = ttk.Label(parent, text="验证控制面板", font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # 标注列表
        list_frame = ttk.LabelFrame(parent, text="检测到的元素")
        list_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # 创建Treeview
        columns = ('ID', '类别', '坐标')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # 绑定选择事件
        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)
        
        # 操作按钮
        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(fill='x', pady=10)
        
        ttk.Button(buttons_frame, text="删除选中", command=self._delete_selected).pack(fill='x', pady=2)
        ttk.Button(buttons_frame, text="编辑选中", command=self._edit_selected).pack(fill='x', pady=2)
        ttk.Button(buttons_frame, text="添加元素", command=self._add_element).pack(fill='x', pady=2)
        
        # 分隔线
        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=10)
        
        # 最终操作
        final_frame = ttk.Frame(parent)
        final_frame.pack(fill='x')
        
        ttk.Button(final_frame, text="确认标注", command=self._confirm_labels, 
                  style='Accent.TButton').pack(fill='x', pady=2)
        ttk.Button(final_frame, text="重置", command=self._reset_labels).pack(fill='x', pady=2)
        ttk.Button(final_frame, text="取消", command=self._cancel).pack(fill='x', pady=2)
    
    def _load_and_display_image(self):
        """加载并显示图片"""
        
        # 读取图片
        image = cv2.imread(self.image_path)
        if image is None:
            messagebox.showerror("错误", f"无法读取图片: {self.image_path}")
            return
        
        # 计算缩放比例
        h, w = image.shape[:2]
        max_w, max_h = self.max_display_size
        
        if w > max_w or h > max_h:
            scale_w = max_w / w
            scale_h = max_h / h
            self.scale_factor = min(scale_w, scale_h)
        else:
            self.scale_factor = 1.0
        
        # 缩放图片
        if self.scale_factor != 1.0:
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            image = cv2.resize(image, (new_w, new_h))
        
        # 绘制标注
        self._draw_labels_on_image(image)
        
        # 转换为Tkinter格式
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        self.image_tk = ImageTk.PhotoImage(pil_image)
        
        # 显示图片
        self.image_label.configure(image=self.image_tk)
        
        # 更新标注列表
        self._update_labels_tree()
    
    def _draw_labels_on_image(self, image):
        """在图片上绘制标注"""
        
        for i, label in enumerate(self.current_labels):
            class_name = label['class']
            bbox = [int(x * self.scale_factor) for x in label['bbox']]
            x1, y1, x2, y2 = bbox
            
            # 颜色
            bgr_color = self.colors.get(class_name, (128, 128, 128))
            
            # 高亮选中的标注
            thickness = 3 if i == self.selected_label_index else 2
            
            # 画边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), bgr_color, thickness)
            
            # 画标签
            label_text = f"{class_name}_{i+1}"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (x1, y1 - text_size[1] - 8), 
                         (x1 + text_size[0] + 6, y1), bgr_color, -1)
            cv2.putText(image, label_text, (x1 + 3, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _update_labels_tree(self):
        """更新标注列表"""
        
        # 清空现有项目
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 添加新项目
        for i, label in enumerate(self.current_labels):
            bbox = label['bbox']
            coord_str = f"({int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])})"
            
            self.tree.insert('', 'end', values=(i+1, label['class'], coord_str))
    
    def _on_tree_select(self, event):
        """处理列表选择事件"""
        
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            index = int(self.tree.item(item)['values'][0]) - 1
            self.selected_label_index = index
            
            # 重新绘制图片（高亮选中项）
            self._refresh_image()
            
            self.status_label.configure(text=f"已选择: {self.current_labels[index]['class']}")
    
    def _refresh_image(self):
        """刷新图片显示"""
        
        # 重新读取原图
        image = cv2.imread(self.image_path)
        
        # 缩放
        if self.scale_factor != 1.0:
            h, w = image.shape[:2]
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            image = cv2.resize(image, (new_w, new_h))
        
        # 绘制标注
        self._draw_labels_on_image(image)
        
        # 更新显示
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        self.image_tk = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=self.image_tk)
    
    def _delete_selected(self):
        """删除选中的标注"""
        
        if self.selected_label_index is not None:
            deleted_label = self.current_labels.pop(self.selected_label_index)
            self.selected_label_index = None
            
            self._refresh_image()
            self._update_labels_tree()
            
            self.status_label.configure(text=f"已删除: {deleted_label['class']}")
            self.result['corrected'] = True
    
    def _edit_selected(self):
        """编辑选中的标注"""
        
        if self.selected_label_index is not None:
            label = self.current_labels[self.selected_label_index]
            
            # 创建编辑对话框
            dialog = EditLabelDialog(self.root, label)
            new_label = dialog.get_result()
            
            if new_label:
                self.current_labels[self.selected_label_index] = new_label
                self._refresh_image()
                self._update_labels_tree()
                self.status_label.configure(text="标注已更新")
                self.result['corrected'] = True
    
    def _add_element(self):
        """添加新元素"""
        
        # 创建添加对话框
        dialog = AddLabelDialog(self.root)
        new_label = dialog.get_result()
        
        if new_label:
            self.current_labels.append(new_label)
            self._refresh_image()
            self._update_labels_tree()
            self.status_label.configure(text=f"已添加: {new_label['class']}")
            self.result['corrected'] = True
    
    def _confirm_labels(self):
        """确认标注"""
        
        self.result = {
            'corrected': len(self.current_labels) != len(self.original_labels) or 
                        self.current_labels != self.original_labels,
            'final_labels': self.current_labels.copy(),
            'original_labels': self.original_labels.copy(),
            'corrected_classes': list(set(label['class'] for label in self.current_labels) - 
                                    set(label['class'] for label in self.original_labels))
        }
        
        self.root.quit()
        self.root.destroy()
    
    def _reset_labels(self):
        """重置到原始标注"""
        
        self.current_labels = self.original_labels.copy()
        self.selected_label_index = None
        
        self._refresh_image()
        self._update_labels_tree()
        self.status_label.configure(text="已重置到原始标注")
    
    def _cancel(self):
        """取消验证"""
        
        self.result = {
            'corrected': False,
            'final_labels': self.original_labels,
            'cancelled': True
        }
        
        self.root.quit()
        self.root.destroy()


class EditLabelDialog:
    """编辑标注对话框"""
    
    def __init__(self, parent, label: Dict):
        self.parent = parent
        self.original_label = label.copy()
        self.result = None
        
        # 可选择的类别
        self.class_options = [
            'receiver_name', 'receiver_avatar', 'input_box', 
            'send_button', 'chat_message', 'contact_item', 'user_avatar'
        ]
        
        self._create_dialog()
    
    def _create_dialog(self):
        """创建对话框"""
        
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("编辑标注")
        self.dialog.geometry("400x300")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # 居中显示
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (300 // 2)
        self.dialog.geometry(f"400x300+{x}+{y}")
        
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill='both', expand=True)
        
        # 类别选择
        ttk.Label(main_frame, text="类别:").grid(row=0, column=0, sticky='w', pady=5)
        self.class_var = tk.StringVar(value=self.original_label['class'])
        class_combo = ttk.Combobox(main_frame, textvariable=self.class_var, 
                                  values=self.class_options, state='readonly')
        class_combo.grid(row=0, column=1, sticky='ew', padx=(10, 0), pady=5)
        
        # 坐标输入
        bbox = self.original_label['bbox']
        
        ttk.Label(main_frame, text="X1:").grid(row=1, column=0, sticky='w', pady=5)
        self.x1_var = tk.StringVar(value=str(int(bbox[0])))
        ttk.Entry(main_frame, textvariable=self.x1_var).grid(row=1, column=1, sticky='ew', padx=(10, 0), pady=5)
        
        ttk.Label(main_frame, text="Y1:").grid(row=2, column=0, sticky='w', pady=5)
        self.y1_var = tk.StringVar(value=str(int(bbox[1])))
        ttk.Entry(main_frame, textvariable=self.y1_var).grid(row=2, column=1, sticky='ew', padx=(10, 0), pady=5)
        
        ttk.Label(main_frame, text="X2:").grid(row=3, column=0, sticky='w', pady=5)
        self.x2_var = tk.StringVar(value=str(int(bbox[2])))
        ttk.Entry(main_frame, textvariable=self.x2_var).grid(row=3, column=1, sticky='ew', padx=(10, 0), pady=5)
        
        ttk.Label(main_frame, text="Y2:").grid(row=4, column=0, sticky='w', pady=5)
        self.y2_var = tk.StringVar(value=str(int(bbox[3])))
        ttk.Entry(main_frame, textvariable=self.y2_var).grid(row=4, column=1, sticky='ew', padx=(10, 0), pady=5)
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="确定", command=self._ok_clicked).pack(side='left', padx=5)
        ttk.Button(button_frame, text="取消", command=self._cancel_clicked).pack(side='left', padx=5)
        
        # 配置列权重
        main_frame.columnconfigure(1, weight=1)
    
    def _ok_clicked(self):
        """确定按钮点击"""
        
        try:
            new_label = {
                'class': self.class_var.get(),
                'bbox': [
                    float(self.x1_var.get()),
                    float(self.y1_var.get()),
                    float(self.x2_var.get()),
                    float(self.y2_var.get())
                ]
            }
            
            # 简单验证
            if new_label['bbox'][0] >= new_label['bbox'][2] or new_label['bbox'][1] >= new_label['bbox'][3]:
                messagebox.showerror("错误", "坐标不合法：x1应小于x2，y1应小于y2")
                return
            
            self.result = new_label
            self.dialog.destroy()
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")
    
    def _cancel_clicked(self):
        """取消按钮点击"""
        self.result = None
        self.dialog.destroy()
    
    def get_result(self):
        """获取结果"""
        self.dialog.wait_window()
        return self.result


class AddLabelDialog:
    """添加标注对话框"""
    
    def __init__(self, parent):
        self.parent = parent
        self.result = None
        
        # 可选择的类别
        self.class_options = [
            'receiver_name', 'receiver_avatar', 'input_box', 
            'send_button', 'chat_message', 'contact_item', 'user_avatar'
        ]
        
        self._create_dialog()
    
    def _create_dialog(self):
        """创建对话框"""
        
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("添加标注")
        self.dialog.geometry("400x300")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # 居中显示
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (300 // 2)
        self.dialog.geometry(f"400x300+{x}+{y}")
        
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill='both', expand=True)
        
        # 类别选择
        ttk.Label(main_frame, text="类别:").grid(row=0, column=0, sticky='w', pady=5)
        self.class_var = tk.StringVar(value=self.class_options[0])
        class_combo = ttk.Combobox(main_frame, textvariable=self.class_var, 
                                  values=self.class_options, state='readonly')
        class_combo.grid(row=0, column=1, sticky='ew', padx=(10, 0), pady=5)
        
        # 坐标输入
        ttk.Label(main_frame, text="X1:").grid(row=1, column=0, sticky='w', pady=5)
        self.x1_var = tk.StringVar(value="0")
        ttk.Entry(main_frame, textvariable=self.x1_var).grid(row=1, column=1, sticky='ew', padx=(10, 0), pady=5)
        
        ttk.Label(main_frame, text="Y1:").grid(row=2, column=0, sticky='w', pady=5)
        self.y1_var = tk.StringVar(value="0")
        ttk.Entry(main_frame, textvariable=self.y1_var).grid(row=2, column=1, sticky='ew', padx=(10, 0), pady=5)
        
        ttk.Label(main_frame, text="X2:").grid(row=3, column=0, sticky='w', pady=5)
        self.x2_var = tk.StringVar(value="100")
        ttk.Entry(main_frame, textvariable=self.x2_var).grid(row=3, column=1, sticky='ew', padx=(10, 0), pady=5)
        
        ttk.Label(main_frame, text="Y2:").grid(row=4, column=0, sticky='w', pady=5)
        self.y2_var = tk.StringVar(value="100")
        ttk.Entry(main_frame, textvariable=self.y2_var).grid(row=4, column=1, sticky='ew', padx=(10, 0), pady=5)
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="确定", command=self._ok_clicked).pack(side='left', padx=5)
        ttk.Button(button_frame, text="取消", command=self._cancel_clicked).pack(side='left', padx=5)
        
        # 配置列权重
        main_frame.columnconfigure(1, weight=1)
    
    def _ok_clicked(self):
        """确定按钮点击"""
        
        try:
            new_label = {
                'class': self.class_var.get(),
                'bbox': [
                    float(self.x1_var.get()),
                    float(self.y1_var.get()),
                    float(self.x2_var.get()),
                    float(self.y2_var.get())
                ]
            }
            
            # 简单验证
            if new_label['bbox'][0] >= new_label['bbox'][2] or new_label['bbox'][1] >= new_label['bbox'][3]:
                messagebox.showerror("错误", "坐标不合法：x1应小于x2，y1应小于y2")
                return
            
            self.result = new_label
            self.dialog.destroy()
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")
    
    def _cancel_clicked(self):
        """取消按钮点击"""
        self.result = None
        self.dialog.destroy()
    
    def get_result(self):
        """获取结果"""
        self.dialog.wait_window()
        return self.result


def main():
    """测试人工验证功能"""
    
    # 示例标注数据
    test_labels = [
        {'class': 'receiver_name', 'bbox': [150, 20, 280, 45]},
        {'class': 'receiver_avatar', 'bbox': [100, 15, 140, 55]},
        {'class': 'input_box', 'bbox': [50, 700, 500, 740]},
        {'class': 'send_button', 'bbox': [510, 705, 550, 735]},
        {'class': 'chat_message', 'bbox': [60, 100, 300, 150]},
        {'class': 'chat_message', 'bbox': [350, 200, 590, 250]},
        {'class': 'contact_item', 'bbox': [10, 80, 200, 110]}
    ]
    
    # 创建验证器
    validator = HumanAssistedValidator(interactive_mode=True, auto_save=True)
    
    # 测试图片路径（请替换为实际路径）
    test_image = "./data/test_images/test_screenshot.jpg"
    
    if os.path.exists(test_image):
        print("🔍 开始人工验证测试...")
        
        # 执行验证
        result = validator.validate_with_visualization(test_image, test_labels)
        
        print("✅ 验证完成!")
        print(f"可视化结果: {result['visualization_paths']}")
        print(f"需要人工审查: {result['needs_human_review']}")
        
        if result['correction_result']:
            print(f"用户修正: {result['correction_result']['corrected']}")
        
        # 显示统计信息
        stats = validator.get_validation_statistics()
        print(f"验证统计: {stats}")
        
    else:
        print(f"测试图片不存在: {test_image}")
        print("请创建测试图片或修改路径")


if __name__ == "__main__":
    main()
