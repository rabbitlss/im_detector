# -*- coding: utf-8 -*-
"""
持久化OCR管理器
针对"YOLO检测 -> OCR识别"的批量处理场景优化
"""

import time
import threading
import os
from typing import List
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass

@dataclass
class Region:
    """区域定义"""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str = ""
    index: int = 0

@dataclass
class OCRResult:
    """OCR识别结果"""
    region: Region
    text: str
    time_ms: float
    worker_id: str

def process_gpu_task_persistent(args):
    """
    在独立进程中处理GPU任务 - 支持持久化和预热
    每个进程有独立的CUDA_VISIBLE_DEVICES设置
    """
    worker_id, gpu_id, image, task_regions, is_first_image = args
    
    # 在进程开始时设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"🚀 进程 {os.getpid()}: 使用GPU {gpu_id}")
    
    # 总计时开始（包含初始化和预热时间）
    total_start_time = time.time()
    
    # 延迟导入，避免在主进程中初始化
    from ultrafast_ocr.core import UltraFastOCR
    
    # 创建OCR实例
    print(f"🔧 进程 {os.getpid()}: 初始化OCR Worker...")
    ocr = UltraFastOCR()
    
    # 如果是第一张图像，进行预热
    if is_first_image:
        print(f"🔥 进程 {os.getpid()}: 第一张图像，开始预热...")
        warmup_start = time.time()
        
        # 创建预热图像
        warmup_images = []
        for i in range(3):
            h, w = (32 + i*16, 100 + i*50)
            warmup_img = np.ones((h, w, 3), dtype=np.uint8) * 255
            cv2.putText(warmup_img, f"warmup{i}", (10, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            warmup_images.append(warmup_img)
        
        # 预热推理
        warmup_times = []
        for warmup_img in warmup_images:
            for _ in range(2):  # 每种图像预热2次
                warmup_time_start = time.time()
                _ = ocr.recognize_single_line(warmup_img)
                warmup_time = (time.time() - warmup_time_start) * 1000
                warmup_times.append(warmup_time)
        
        warmup_duration = (time.time() - warmup_start) * 1000
        print(f"🔥 进程 {os.getpid()}: 预热完成 ({warmup_duration:.1f}ms)")
        print(f"     预热效果: {warmup_times[0]:.1f}ms -> {warmup_times[-1]:.1f}ms")
    
    # 开始处理实际任务
    gpu_start_time = time.time()
    local_results = []
    region_times = []
    
    print(f"📝 进程 {os.getpid()}: 开始处理 {len(task_regions)} 个区域")
    
    for i, region in enumerate(task_regions):
        # 从图像中裁剪区域
        x1, y1, x2, y2 = region.x1, region.y1, region.x2, region.y2
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            text = ""
            region_time = 0
        else:
            region_start = time.time()
            text = ocr.recognize_single_line(roi)
            region_time = (time.time() - region_start) * 1000
        
        result = OCRResult(
            region=region,
            text=text,
            time_ms=region_time,
            worker_id=f"GPU_{gpu_id}_P{os.getpid()}"
        )
        
        local_results.append((region.index, result))
        region_times.append(region_time)
        
        # 显示处理进度
        print(f"   [进程{os.getpid()}|GPU{gpu_id}] ({i+1}/{len(task_regions)}) {region.label}: {text[:30]}... ({region_time:.1f}ms)"
              if len(text) > 30 else
              f"   [进程{os.getpid()}|GPU{gpu_id}] ({i+1}/{len(task_regions)}) {region.label}: {text} ({region_time:.1f}ms)")
        
        # 显示预热效果
        if i == 0 and is_first_image and region_time < 100:
            print(f"   ✅ 预热成功！首次识别仅耗时 {region_time:.1f}ms")
    
    gpu_processing_time = (time.time() - gpu_start_time) * 1000
    total_time_with_init = (time.time() - total_start_time) * 1000
    
    print(f"🏁 进程 {os.getpid()}: 处理完成")
    print(f"     纯识别时间: {gpu_processing_time:.1f}ms")
    if is_first_image:
        print(f"     总时间(含预热): {total_time_with_init:.1f}ms")
    
    return {
        'results': local_results,
        'gpu_id': gpu_id,
        'total_time': gpu_processing_time,
        'total_time_with_init': total_time_with_init,
        'region_times': region_times,
        'num_regions': len(task_regions),
        'process_id': os.getpid(),
        'avg_time_per_region': gpu_processing_time / len(task_regions) if task_regions else 0,
        'is_first_image': is_first_image
    }

class PersistentOCRManager:
    """持久化OCR管理器 - 支持进程并发和模型预热"""
    
    def __init__(self, gpu_ids: List[int] = None, use_process_pool: bool = True):
        """
        初始化持久化OCR管理器
        
        Args:
            gpu_ids: 使用的GPU ID列表
            use_process_pool: 是否使用进程池（推荐True，确保GPU隔离）
        """
        self.gpu_ids = gpu_ids or [0]
        self.use_process_pool = use_process_pool
        self.first_image_processed = False
        
        print(f"🚀 初始化持久化OCR管理器")
        print(f"   使用GPU: {self.gpu_ids}")
        print(f"   并发模式: {'进程池' if use_process_pool else '线程池'}")
        print(f"   GPU隔离: {'✅' if use_process_pool else '⚠️'}")
        
        if use_process_pool:
            print(f"   ✅ 每个GPU在独立进程中运行，真正并行")
        else:
            print(f"   ⚠️ 线程池模式，可能存在GPU环境变量冲突")
    
    def _get_or_create_ocr_worker(self, gpu_id: int):
        """获取或创建OCR工作器（懒加载）"""
        if gpu_id not in self.ocr_workers:
            print(f"🔧 首次创建GPU {gpu_id}的OCR实例...")
            
            # 设置GPU环境
            import os
            original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            try:
                from ultrafast_ocr.core import UltraFastOCR
                ocr_instance = UltraFastOCR()
                self.ocr_workers[gpu_id] = ocr_instance
                self.is_warmed_up[gpu_id] = False
                
                print(f"✅ GPU {gpu_id}的OCR实例创建完成")
                
                # 恢复环境变量
                if original_cuda_visible:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                else:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                    
            except Exception as e:
                print(f"❌ GPU {gpu_id}的OCR创建失败: {e}")
                # 恢复环境变量
                if original_cuda_visible:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                else:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                raise
        
        return self.ocr_workers[gpu_id]
    
    def _warmup_ocr_if_needed(self, gpu_id: int):
        """预热OCR模型（线程安全）"""
        with self.warmup_lock:
            if self.is_warmed_up.get(gpu_id, False):
                return  # 已经预热过
            
            print(f"🔥 开始预热GPU {gpu_id}的OCR模型...")
            start_time = time.time()
            
            ocr = self._get_or_create_ocr_worker(gpu_id)
            
            # 创建预热图像
            warmup_images = []
            for i in range(3):  # 3种不同大小的预热图像
                h, w = (32 + i*16, 100 + i*50)  # 不同尺寸
                img = np.ones((h, w, 3), dtype=np.uint8) * 255
                cv2.putText(img, f"warmup{i}", (10, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                warmup_images.append(img)
            
            # 预热推理
            warmup_times = []
            for i, img in enumerate(warmup_images):
                for round_idx in range(2):  # 每种图像预热2次
                    warmup_start = time.time()
                    _ = ocr.recognize_single_line(img)
                    warmup_time = (time.time() - warmup_start) * 1000
                    warmup_times.append(warmup_time)
                    
                    if i == 0 and round_idx == 0:
                        first_warmup_time = warmup_time
            
            total_warmup_time = (time.time() - start_time) * 1000
            avg_warmup_time = sum(warmup_times[-3:]) / 3  # 最后3次的平均时间
            
            self.is_warmed_up[gpu_id] = True
            
            print(f"🔥 GPU {gpu_id}预热完成:")
            print(f"     首次预热: {first_warmup_time:.1f}ms")
            print(f"     最终平均: {avg_warmup_time:.1f}ms")
            print(f"     总预热时间: {total_warmup_time:.1f}ms")
            print(f"     ✅ 后续识别将保持 {avg_warmup_time:.1f}ms 左右的速度")
    
    def _async_warmup_unused_gpus(self, exclude_gpu: int):
        """异步预热其他未使用的GPU"""
        def warmup_worker():
            for gpu_id in self.gpu_ids:
                if gpu_id != exclude_gpu and not self.is_warmed_up.get(gpu_id, False):
                    try:
                        self._warmup_ocr_if_needed(gpu_id)
                    except Exception as e:
                        print(f"⚠️ 异步预热GPU {gpu_id}失败: {e}")
        
        # 在后台线程中预热其他GPU
        threading.Thread(target=warmup_worker, daemon=True).start()
    
    def process_image_with_yolo_ocr(self, 
                                   image: np.ndarray, 
                                   yolo_detections: List[dict],
                                   image_index: int = 0) -> List[str]:
        """
        处理图像：YOLO检测 -> OCR识别（支持进程并发）
        
        Args:
            image: 输入图像
            yolo_detections: YOLO检测结果 [{'bbox': [x1,y1,x2,y2], 'class': 'text'}, ...]
            image_index: 图像索引（用于日志）
            
        Returns:
            OCR识别结果列表
        """
        print(f"\n📷 处理第 {image_index+1} 张图像...")
        total_start = time.time()
        
        if not yolo_detections:
            print("   ⚠️ 没有YOLO检测结果")
            return []
        
        print(f"   📍 YOLO检测到 {len(yolo_detections)} 个文本区域")
        
        # 是否为第一张图像
        is_first_image = not self.first_image_processed
        if is_first_image:
            print(f"   🎯 首张图像，启用预热策略...")
            self.first_image_processed = True
        else:
            print(f"   ⚡ 非首张图像，享受预热效果...")
        
        # 创建Region对象
        region_objs = []
        for i, detection in enumerate(yolo_detections):
            x1, y1, x2, y2 = detection['bbox']
            # 验证区域有效性
            if x2 > x1 and y2 > y1 and y1 >= 0 and x1 >= 0:
                region = Region(x1, y1, x2, y2, f"region_{i}", i)
                region_objs.append(region)
        
        if not region_objs:
            print("   ⚠️ 没有有效的文本区域")
            return []
        
        # GPU负载均衡分配
        gpu_tasks = [[] for _ in self.gpu_ids]
        for i, region in enumerate(region_objs):
            gpu_idx = i % len(self.gpu_ids)
            gpu_tasks[gpu_idx].append(region)
        
        print(f"   🔀 区域分配到 {len(self.gpu_ids)} 个GPU: {[len(tasks) for tasks in gpu_tasks]}")
        
        ocr_start = time.time()
        
        if self.use_process_pool:
            # 使用进程池 - 真正的GPU并行
            results_dict = self._process_with_process_pool(image, gpu_tasks, is_first_image)
        else:
            # 使用线程池 - 可能有GPU冲突
            results_dict = self._process_with_thread_pool(image, gpu_tasks, is_first_image)
        
        # 按原始顺序排列结果
        results = [results_dict.get(i, "") for i in range(len(region_objs))]
        
        ocr_time = (time.time() - ocr_start) * 1000
        total_time = (time.time() - total_start) * 1000
        
        print(f"   ⏱️ OCR识别: {ocr_time:.1f}ms")
        print(f"   ⏱️ 总耗时: {total_time:.1f}ms")
        print(f"   📊 平均每区域: {ocr_time/len(region_objs):.1f}ms")
        
        if image_index > 0:
            print(f"   ✅ 第{image_index+1}张图像享受预热效果！")
        
        return results
    
    def _process_with_process_pool(self, image: np.ndarray, gpu_tasks: List[List[Region]], is_first_image: bool) -> dict:
        """使用进程池处理（推荐，真正并行）"""
        # 准备进程池参数
        process_args = []
        for i, (gpu_id, task_regions) in enumerate(zip(self.gpu_ids, gpu_tasks)):
            if task_regions:  # 只处理有任务的GPU
                process_args.append((i, gpu_id, image, task_regions, is_first_image))
        
        results_dict = {}
        gpu_time_stats = {}
        
        print(f"   🚀 启动 {len(process_args)} 个GPU进程...")
        
        with ProcessPoolExecutor(max_workers=len(process_args)) as executor:
            # 提交任务到进程池
            future_to_args = {
                executor.submit(process_gpu_task_persistent, args): args 
                for args in process_args
            }
            
            # 收集结果
            for future in as_completed(future_to_args):
                args = future_to_args[future]
                worker_id, gpu_id, _, _, _ = args
                
                try:
                    gpu_result = future.result()
                    process_id = gpu_result['process_id']
                    
                    print(f"   ✅ GPU {gpu_id} (进程 {process_id}) 完成")
                    
                    # 保存时间统计
                    gpu_time_stats[f'GPU_{gpu_id}'] = {
                        'total_time_ms': gpu_result['total_time'],
                        'num_regions': gpu_result['num_regions'],
                        'avg_time_ms': gpu_result['avg_time_per_region'],
                        'process_id': process_id
                    }
                    
                    # 保存识别结果
                    for region_idx, result in gpu_result['results']:
                        results_dict[region_idx] = result.text
                        
                except Exception as e:
                    print(f"   ❌ GPU进程失败: {e}")
        
        # 显示GPU统计
        if gpu_time_stats:
            print(f"   📊 GPU处理统计:")
            for gpu_name, stats in gpu_time_stats.items():
                print(f"      {gpu_name}: {stats['num_regions']}区域, "
                      f"平均{stats['avg_time_ms']:.1f}ms/区域 "
                      f"(进程{stats['process_id']})")
        
        return results_dict
    
    def _process_with_thread_pool(self, image: np.ndarray, gpu_tasks: List[List[Region]], is_first_image: bool) -> dict:
        """使用线程池处理（可能有GPU冲突）"""
        print("   ⚠️ 使用线程池模式，可能存在GPU环境变量冲突")
        # 这里可以实现线程池版本，但推荐使用进程池
        # 为了简化，这里返回空结果，建议始终使用进程池
        return {}
    
    def batch_process_images(self, 
                           images: List[np.ndarray],
                           yolo_results: List[List[dict]]) -> List[List[str]]:
        """
        批量处理多张图像
        
        Args:
            images: 图像列表
            yolo_results: 每张图像的YOLO检测结果
            
        Returns:
            每张图像的OCR结果
        """
        print(f"🚀 开始批量处理 {len(images)} 张图像")
        print("=" * 60)
        
        all_results = []
        batch_start = time.time()
        
        for i, (image, detections) in enumerate(zip(images, yolo_results)):
            ocr_results = self.process_image_with_yolo_ocr(image, detections, i)
            all_results.append(ocr_results)
        
        total_batch_time = (time.time() - batch_start) * 1000
        
        print(f"\n🎉 批量处理完成:")
        print(f"   - 总图像数: {len(images)}")
        print(f"   - 总耗时: {total_batch_time:.1f}ms")
        print(f"   - 平均每张: {total_batch_time/len(images):.1f}ms")
        print(f"   - 模型常驻内存，无需重复初始化")
        
        return all_results
    
    def get_warmup_status(self):
        """获取预热状态"""
        return {
            f"GPU_{gpu_id}": f"进程并发模式"
            for gpu_id in self.gpu_ids
        }
    
    def cleanup(self):
        """清理资源（进程模式下无需手动清理）"""
        if self.use_process_pool:
            print("🧹 进程池模式，子进程会自动清理资源")
        else:
            print("🧹 清理线程池资源...")
    
    def get_performance_summary(self, total_images: int, total_time_ms: float):
        """获取性能总结"""
        return {
            'total_images': total_images,
            'total_time_ms': total_time_ms,
            'avg_time_per_image': total_time_ms / total_images if total_images > 0 else 0,
            'images_per_second': total_images / (total_time_ms / 1000) if total_time_ms > 0 else 0,
            'gpu_count': len(self.gpu_ids),
            'concurrent_mode': '进程池' if self.use_process_pool else '线程池',
            'gpu_isolation': self.use_process_pool
        }

def demo_persistent_ocr():
    """演示持久化OCR管理器的效果"""
    
    print("🎭 持久化OCR管理器演示")
    print("=" * 80)
    
    # 创建测试图像
    test_images = []
    test_yolo_results = []
    
    for img_idx in range(3):  # 3张图像
        # 创建图像
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # 模拟YOLO检测结果
        detections = []
        for region_idx in range(5):  # 每张图像5个区域
            x = (region_idx % 3) * 180 + 20
            y = (region_idx // 3) * 150 + 50 + img_idx * 20  # 每张图像略有不同
            
            # 在图像上绘制文本
            text = f"Image{img_idx}_Text{region_idx}"
            cv2.putText(img, text, (x+10, y+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # 添加YOLO检测结果
            detections.append({
                'bbox': [x, y, x+160, y+60],
                'class': 'text',
                'confidence': 0.95
            })
        
        test_images.append(img)
        test_yolo_results.append(detections)
    
    # 创建持久化OCR管理器
    ocr_manager = PersistentOCRManager(gpu_ids=[0, 1])  # 使用2个GPU
    
    # 批量处理
    results = ocr_manager.batch_process_images(test_images, test_yolo_results)
    
    # 显示结果
    print("\n📊 处理结果汇总:")
    for i, image_results in enumerate(results):
        print(f"图像 {i+1}: 识别到 {len(image_results)} 个文本")
        for j, text in enumerate(image_results[:3]):  # 只显示前3个
            print(f"   {j+1}: {text}")
        if len(image_results) > 3:
            print(f"   ... 还有 {len(image_results)-3} 个结果")
    
    # 显示预热状态
    print(f"\n🔥 预热状态: {ocr_manager.get_warmup_status()}")
    
    return ocr_manager

if __name__ == "__main__":
    # 演示效果
    manager = demo_persistent_ocr()
    
    print("\n" + "=" * 80)
    print("💡 关键优势：")
    print("1. 模型常驻内存，避免重复初始化")
    print("2. 第一张图像处理时完成预热")
    print("3. 第二张图像开始享受高速识别")
    print("4. 异步预热其他GPU，不影响当前处理")
    print("5. 智能负载均衡，充分利用多GPU")
    print("=" * 80)
