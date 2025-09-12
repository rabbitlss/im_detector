# -*- coding: utf-8 -*-
"""
持久化OCR管理器
针对"YOLO检测 -> OCR识别"的批量处理场景优化
"""

import time
import os
from typing import List
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing

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

# 全局变量用于维护OCR实例
_ocr_instances = {}
_process_initialized = {}

def init_worker_process(gpu_id):
    """
    初始化工作进程，创建OCR实例
    这个函数在进程启动时只执行一次
    """
    global _ocr_instances, _process_initialized
    
    process_id = os.getpid()
    if process_id in _process_initialized:
        return  # 已经初始化过
    
    # 设置GPU环境
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"🚀 进程 {process_id}: 初始化GPU {gpu_id}")
    
    # 创建OCR实例（只创建一次）
    try:
        from ultrafast_ocr.core import UltraFastOCR
        print(f"🔧 进程 {process_id}: 初始化OCR模型...")
        
        init_start = time.time()
        ocr = UltraFastOCR()
        init_time = (time.time() - init_start) * 1000
        
        _ocr_instances[process_id] = ocr
        _process_initialized[process_id] = True
        
        print(f"✅ 进程 {process_id}: OCR模型初始化完成 ({init_time:.1f}ms)")
        print(f"💾 进程 {process_id}: 模型已常驻内存，后续调用无需重新初始化")
        
    except Exception as e:
        print(f"❌ 进程 {process_id}: OCR模型初始化失败: {e}")
        raise

def get_ocr_instance():
    """
    获取当前进程的OCR实例
    """
    global _ocr_instances
    process_id = os.getpid()
    return _ocr_instances.get(process_id)

def process_gpu_task_persistent(args):
    """
    在独立进程中处理GPU任务 - 真正的模型持久化
    模型只在进程首次调用时初始化，后续调用直接复用
    """
    worker_id, gpu_id, image, task_regions = args
    
    process_id = os.getpid()
    print(f"📝 进程 {process_id}: 处理 {len(task_regions)} 个区域")
    
    # 获取已初始化的OCR实例
    ocr = get_ocr_instance()
    if ocr is None:
        print(f"❌ 进程 {process_id}: OCR实例未初始化")
        raise RuntimeError(f"进程 {process_id} 中的OCR实例未初始化")
    
    print(f"✅ 进程 {process_id}: 使用已初始化的OCR实例（无需重新加载模型）")
    
    # 开始处理实际任务
    gpu_start_time = time.time()
    local_results = []
    region_times = []
    
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
        
    
    gpu_processing_time = (time.time() - gpu_start_time) * 1000
    
    print(f"🏁 进稌 {process_id}: 处理完成")
    print(f"     识别时间: {gpu_processing_time:.1f}ms （无模型初始化开销）")
    
    return {
        'results': local_results,
        'gpu_id': gpu_id,
        'total_time': gpu_processing_time,
        'total_time_with_init': gpu_processing_time,  # 现在没有初始化开销
        'region_times': region_times,
        'num_regions': len(task_regions),
        'process_id': os.getpid(),
        'avg_time_per_region': gpu_processing_time / len(task_regions) if task_regions else 0
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
        self.process_pool = None
        self.pool_initialized = False
        
        print(f"🚀 初始化持久化OCR管理器")
        print(f"   使用GPU: {self.gpu_ids}")
        print(f"   并发模式: {'进程池' if use_process_pool else '线程池'}")
        print(f"   GPU隔离: {'✅' if use_process_pool else '⚠️'}")
        
        if use_process_pool:
            print(f"   ✅ 每个GPU在独立进程中运行，真正并行")
            print(f"   💾 模型将在进程首次调用时初始化，后续复用")
        else:
            print(f"   ⚠️ 线程池模式，可能存在GPU环境变量冲突")
    
    
    
    
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
            results_dict = self._process_with_process_pool(image, gpu_tasks)
        else:
            # 使用线程池 - 可能有GPU冲突
            results_dict = self._process_with_thread_pool(image, gpu_tasks)
        
        # 按原始顺序排列结果
        results = [results_dict.get(i, "") for i in range(len(region_objs))]
        
        ocr_time = (time.time() - ocr_start) * 1000
        total_time = (time.time() - total_start) * 1000
        
        print(f"   ⏱️ OCR识别: {ocr_time:.1f}ms")
        print(f"   ⏱️ 总耗时: {total_time:.1f}ms")
        print(f"   📊 平均每区域: {ocr_time/len(region_objs):.1f}ms")
        
        return results
    
    def _get_or_create_process_pool(self):
        """获取或创建持久化进程池"""
        if self.process_pool is None or not self.pool_initialized:
            print(f"   🚀 创建持久化进程池...")
            
            # 为每个GPU创建一个进程，并初始化OCR实例
            max_workers = len(self.gpu_ids)
            self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
            
            # 提前初始化所有进程的OCR实例
            print(f"   🔧 预初始化 {len(self.gpu_ids)} 个GPU进程...")
            init_futures = []
            for gpu_id in self.gpu_ids:
                future = self.process_pool.submit(init_worker_process, gpu_id)
                init_futures.append(future)
            
            # 等待所有初始化完成
            for future in init_futures:
                try:
                    future.result(timeout=30)  # 30秒超时
                except Exception as e:
                    print(f"   ❌ 进程初始化失败: {e}")
                    raise
            
            self.pool_initialized = True
            print(f"   ✅ 所有GPU进程初始化完成，模型已常驻内存")
        
        return self.process_pool
    
    def _process_with_process_pool(self, image: np.ndarray, gpu_tasks: List[List[Region]]) -> dict:
        """使用持久化进程池处理（真正持久化）"""
        # 准备进程池参数
        process_args = []
        for i, (gpu_id, task_regions) in enumerate(zip(self.gpu_ids, gpu_tasks)):
            if task_regions:  # 只处理有任务的GPU
                process_args.append((i, gpu_id, image, task_regions))
        
        results_dict = {}
        gpu_time_stats = {}
        
        # 使用持久化进程池
        executor = self._get_or_create_process_pool()
        print(f"   ⚡ 使用已初始化的进程池处理 {len(process_args)} 个任务...")
        # 提交任务到持久化进程池
        future_to_args = {
            executor.submit(process_gpu_task_persistent, args): args 
            for args in process_args
        }
            
        # 收集结果
        for future in as_completed(future_to_args):
            args = future_to_args[future]
            _, gpu_id, _, _ = args
            
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
                      f"(持久化进程{stats['process_id']})")
        
        return results_dict
    
    def _process_with_thread_pool(self, image: np.ndarray, gpu_tasks: List[List[Region]]) -> dict:
        """使用线程池处理（可能有GPU冲突）"""
        # 避免未使用参数警告
        _ = image, gpu_tasks
        print("   ⚠️ 使用线程池模式，可能存在GPU环境变量冲突")
        print("   🚧 线程池功能未实现，请使用 use_process_pool=True")
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
    
    def get_status(self):
        """获取管理器状态"""
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
    
    print("🎭 持久化OCR管理器演示 - 真正的模型持久化")
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
    
    # 显示状态
    print(f"\n📈 管理器状态: {ocr_manager.get_status()}")
    
    return ocr_manager

if __name__ == "__main__":
    # 演示效果
    manager = demo_persistent_ocr()
    
    print("\n" + "=" * 80)
    print("💡 关键优势：")
    print("1. 各进程中的OCR模型独立初始化")
    print("2. 真正的多GPU并行处理")
    print("3. 简化的工作流程，无首张图像判断")
    print("4. GPU资源隔离，避免冲突")
    print("5. 智能负载均衡，充分利用多GPU")
    print("=" * 80)
