# -*- coding: utf-8 -*-
"""
并行区域OCR识别脚本
支持多GPU/多进程并发执行OCR识别
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from queue import Queue
import threading

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
class RegionOCRResult:
    """区域OCR结果"""
    region: Region
    text: str
    time_ms: float
    worker_id: str  # 标识哪个worker处理的

class OCRWorker:
    """OCR工作器（每个GPU一个实例）"""
    
    def __init__(self, worker_id: int, device: str = 'cpu'):
        """
        初始化OCR工作器
        
        Args:
            worker_id: 工作器ID
            device: 设备类型 ('cpu', 'cuda:0', 'cuda:1', etc.)
        """
        self.worker_id = worker_id
        self.device = device
        
        # 延迟导入，避免在主进程中初始化
        from ultrafast_ocr.core import UltraFastOCR
        
        # 根据设备初始化OCR
        if 'cuda' in device:
            import os
            # 设置CUDA设备
            gpu_id = int(device.split(':')[1]) if ':' in device else 0
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
        self.ocr = UltraFastOCR()
        print(f"✅ Worker {worker_id} 初始化完成 (设备: {device})")
    
    def process_region(self, image: np.ndarray, region: Region) -> RegionOCRResult:
        """处理单个区域"""
        x1, y1, x2, y2 = region.x1, region.y1, region.x2, region.y2
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return RegionOCRResult(
                region=region,
                text="",
                time_ms=0,
                worker_id=f"worker_{self.worker_id}"
            )
        
        start = time.time()
        text = self.ocr.recognize_single_line(roi)
        time_ms = (time.time() - start) * 1000
        
        return RegionOCRResult(
            region=region,
            text=text,
            time_ms=time_ms,
            worker_id=f"worker_{self.worker_id}"
        )

def process_region_batch(args):
    """处理一批区域（用于进程池）"""
    worker_id, device, image, regions = args
    worker = OCRWorker(worker_id, device)
    results = []
    
    for region in regions:
        result = worker.process_region(image, region)
        results.append(result)
        
    return results

class ParallelRegionOCR:
    """并行区域OCR处理器"""
    
    def __init__(self, num_workers: int = None, use_gpu: bool = True, gpu_ids: List[int] = None):
        """
        初始化并行OCR处理器
        
        Args:
            num_workers: 工作器数量（None则自动检测）
            use_gpu: 是否使用GPU
            gpu_ids: 指定GPU ID列表（如 [0, 1, 2]）
        """
        self.use_gpu = use_gpu
        
        # 确定工作器数量和设备
        if use_gpu and gpu_ids:
            self.devices = [f"cuda:{gpu_id}" for gpu_id in gpu_ids]
            self.num_workers = len(gpu_ids)
        elif use_gpu:
            # 自动检测GPU数量
            try:
                import torch
                gpu_count = torch.cuda.device_count()
                self.devices = [f"cuda:{i}" for i in range(gpu_count)]
                self.num_workers = gpu_count
            except:
                print("⚠️ 无法检测GPU，使用CPU")
                self.devices = ["cpu"]
                self.num_workers = num_workers or mp.cpu_count()
        else:
            self.devices = ["cpu"] * (num_workers or mp.cpu_count())
            self.num_workers = len(self.devices)
        
        print(f"🚀 并行OCR处理器初始化:")
        print(f"   - 工作器数量: {self.num_workers}")
        print(f"   - 设备列表: {self.devices}")
    
    def recognize_regions_parallel_thread(self, 
                                         image: np.ndarray,
                                         regions: List[Tuple[int, int, int, int]],
                                         labels: List[str] = None) -> List[str]:
        """
        使用线程池并行识别（共享内存，适合I/O密集型）
        
        Args:
            image: 输入图像
            regions: 区域坐标列表
            labels: 可选的区域标签
            
        Returns:
            识别文本列表
        """
        print(f"🔍 使用线程池并行识别 {len(regions)} 个区域...")
        start_time = time.time()
        
        # 创建Region对象
        region_objs = []
        for i, (x1, y1, x2, y2) in enumerate(regions):
            label = labels[i] if labels and i < len(labels) else f"region_{i}"
            region_objs.append(Region(x1, y1, x2, y2, label, i))
        
        # 创建工作器
        workers = [OCRWorker(i, self.devices[i % len(self.devices)]) 
                  for i in range(self.num_workers)]
        
        # 使用线程池并行处理
        results = [None] * len(regions)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交任务
            future_to_region = {}
            for i, region in enumerate(region_objs):
                worker = workers[i % self.num_workers]
                future = executor.submit(worker.process_region, image, region)
                future_to_region[future] = i
            
            # 收集结果
            for future in as_completed(future_to_region):
                idx = future_to_region[future]
                result = future.result()
                results[idx] = result.text
                
                # 显示进度
                print(f"   [{result.worker_id}] {result.region.label}: {result.text[:30]}..." 
                      if len(result.text) > 30 else 
                      f"   [{result.worker_id}] {result.region.label}: {result.text}")
        
        total_time = (time.time() - start_time) * 1000
        print(f"\n✅ 并行识别完成:")
        print(f"   - 总耗时: {total_time:.1f}ms")
        print(f"   - 平均每区域: {total_time/len(regions):.1f}ms")
        print(f"   - 加速比: {len(regions)/(total_time/1000)/self.num_workers:.2f}x")
        
        return results
    
    def recognize_regions_parallel_process(self,
                                          image: np.ndarray,
                                          regions: List[Tuple[int, int, int, int]],
                                          labels: List[str] = None) -> List[str]:
        """
        使用进程池并行识别（独立内存，适合CPU密集型）
        
        Args:
            image: 输入图像
            regions: 区域坐标列表
            labels: 可选的区域标签
            
        Returns:
            识别文本列表
        """
        print(f"🔍 使用进程池并行识别 {len(regions)} 个区域...")
        start_time = time.time()
        
        # 创建Region对象
        region_objs = []
        for i, (x1, y1, x2, y2) in enumerate(regions):
            label = labels[i] if labels and i < len(labels) else f"region_{i}"
            region_objs.append(Region(x1, y1, x2, y2, label, i))
        
        # 将区域分配给不同的工作器
        region_batches = [[] for _ in range(self.num_workers)]
        for i, region in enumerate(region_objs):
            region_batches[i % self.num_workers].append(region)
        
        # 准备进程池参数
        process_args = []
        for i in range(self.num_workers):
            if region_batches[i]:  # 只处理非空批次
                device = self.devices[i % len(self.devices)]
                process_args.append((i, device, image, region_batches[i]))
        
        # 使用进程池处理
        results_dict = {}
        
        with ProcessPoolExecutor(max_workers=len(process_args)) as executor:
            futures = [executor.submit(process_region_batch, args) for args in process_args]
            
            for future in as_completed(futures):
                batch_results = future.result()
                for result in batch_results:
                    results_dict[result.region.index] = result.text
                    print(f"   [{result.worker_id}] {result.region.label}: {result.text[:30]}..."
                          if len(result.text) > 30 else
                          f"   [{result.worker_id}] {result.region.label}: {result.text}")
        
        # 按原始顺序排列结果
        results = [results_dict.get(i, "") for i in range(len(regions))]
        
        total_time = (time.time() - start_time) * 1000
        print(f"\n✅ 并行识别完成:")
        print(f"   - 总耗时: {total_time:.1f}ms")
        print(f"   - 平均每区域: {total_time/len(regions):.1f}ms")
        
        return results
    
    def recognize_regions_gpu_batch(self,
                                   image: np.ndarray,
                                   regions: List[Tuple[int, int, int, int]],
                                   labels: List[str] = None,
                                   batch_size: int = 8) -> List[str]:
        """
        GPU批处理识别（将多个区域打包到GPU）
        
        Args:
            image: 输入图像
            regions: 区域坐标列表
            labels: 可选的区域标签
            batch_size: 每批处理的区域数
            
        Returns:
            识别文本列表
        """
        if not self.use_gpu:
            print("⚠️ GPU批处理需要GPU支持，切换到线程池模式")
            return self.recognize_regions_parallel_thread(image, regions, labels)
        
        print(f"🔍 使用GPU批处理识别 {len(regions)} 个区域...")
        print(f"   - 批大小: {batch_size}")
        print(f"   - 批次数: {(len(regions) + batch_size - 1) // batch_size}")
        
        start_time = time.time()
        results = []
        
        # 创建GPU工作器
        gpu_workers = []
        for i, device in enumerate(self.devices):
            try:
                worker = OCRWorker(i, device)
                gpu_workers.append(worker)
            except Exception as e:
                print(f"⚠️ 无法初始化GPU {device}: {e}")
        
        if not gpu_workers:
            print("❌ 没有可用的GPU工作器")
            return []
        
        # 批处理
        for batch_start in range(0, len(regions), batch_size):
            batch_end = min(batch_start + batch_size, len(regions))
            batch_regions = regions[batch_start:batch_end]
            
            # 选择工作器（轮询）
            worker = gpu_workers[(batch_start // batch_size) % len(gpu_workers)]
            
            # 处理批次
            for i, (x1, y1, x2, y2) in enumerate(batch_regions):
                idx = batch_start + i
                label = labels[idx] if labels and idx < len(labels) else f"region_{idx}"
                region = Region(x1, y1, x2, y2, label, idx)
                
                result = worker.process_region(image, region)
                results.append(result.text)
                
                print(f"   [GPU:{worker.worker_id}] {label}: {result.text[:30]}..."
                      if len(result.text) > 30 else
                      f"   [GPU:{worker.worker_id}] {label}: {result.text}")
        
        total_time = (time.time() - start_time) * 1000
        print(f"\n✅ GPU批处理完成:")
        print(f"   - 总耗时: {total_time:.1f}ms")
        print(f"   - 平均每区域: {total_time/len(regions):.1f}ms")
        print(f"   - 吞吐量: {len(regions)/(total_time/1000):.1f} 区域/秒")
        
        return results


def demo_parallel_ocr():
    """演示并行OCR"""
    
    print("🎭 并行OCR演示")
    print("=" * 60)
    
    # 创建测试图像
    test_image = np.ones((800, 1200, 3), dtype=np.uint8) * 255
    
    # 创建多个测试区域
    test_regions = []
    test_texts = []
    
    for i in range(20):  # 创建20个区域
        x = (i % 4) * 300 + 50
        y = (i // 4) * 150 + 50
        test_regions.append((x, y, x + 250, y + 100))
        
        text = f"Region {i+1}"
        test_texts.append(text)
        
        # 在图像上绘制
        cv2.rectangle(test_image, (x, y), (x+250, y+100), (240, 240, 240), -1)
        cv2.putText(test_image, text, (x+10, y+50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    labels = [f"区域{i+1}" for i in range(len(test_regions))]
    
    # 测试不同的并行方式
    
    # 1. 线程池并行（使用2个工作器）
    print("\n1. 线程池并行:")
    processor1 = ParallelRegionOCR(num_workers=2, use_gpu=False)
    results1 = processor1.recognize_regions_parallel_thread(test_image, test_regions, labels)
    
    # 2. 进程池并行（使用4个工作器）
    print("\n2. 进程池并行:")
    processor2 = ParallelRegionOCR(num_workers=4, use_gpu=False)
    results2 = processor2.recognize_regions_parallel_process(test_image, test_regions, labels)
    
    # 3. GPU批处理（如果有GPU）
    print("\n3. GPU批处理:")
    try:
        processor3 = ParallelRegionOCR(use_gpu=True, gpu_ids=[0, 1])  # 使用GPU 0和1
        results3 = processor3.recognize_regions_gpu_batch(test_image, test_regions, labels, batch_size=5)
    except:
        print("   ⚠️ GPU不可用，跳过")
        results3 = []
    
    # 比较结果
    print("\n📊 结果对比:")
    print(f"   线程池结果数: {len(results1)}")
    print(f"   进程池结果数: {len(results2)}")
    if results3:
        print(f"   GPU批处理结果数: {len(results3)}")


def benchmark_parallel_performance(image_path: str, num_regions: int = 50):
    """
    性能基准测试
    
    Args:
        image_path: 图像路径
        num_regions: 测试区域数量
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    h, w = image.shape[:2]
    
    # 生成随机区域
    import random
    regions = []
    for _ in range(num_regions):
        x1 = random.randint(0, w - 100)
        y1 = random.randint(0, h - 50)
        x2 = min(x1 + random.randint(50, 200), w)
        y2 = min(y1 + random.randint(30, 100), h)
        regions.append((x1, y1, x2, y2))
    
    print(f"📊 性能基准测试")
    print(f"   - 图像: {image_path}")
    print(f"   - 区域数: {num_regions}")
    print("=" * 60)
    
    results = {}
    
    # 测试不同配置
    configs = [
        ("单线程", 1, False),
        ("2线程", 2, False),
        ("4线程", 4, False),
        ("8线程", 8, False),
    ]
    
    # 如果有GPU，添加GPU测试
    try:
        import torch
        if torch.cuda.is_available():
            configs.append(("GPU", 1, True))
            if torch.cuda.device_count() > 1:
                configs.append((f"{torch.cuda.device_count()}GPU", torch.cuda.device_count(), True))
    except:
        pass
    
    for name, workers, use_gpu in configs:
        print(f"\n测试 {name}:")
        processor = ParallelRegionOCR(num_workers=workers, use_gpu=use_gpu)
        
        start = time.time()
        if workers == 1:
            # 单线程使用线程池方法
            _ = processor.recognize_regions_parallel_thread(image, regions)
        else:
            # 多线程/GPU使用对应方法
            if use_gpu:
                _ = processor.recognize_regions_gpu_batch(image, regions)
            else:
                _ = processor.recognize_regions_parallel_thread(image, regions)
        
        elapsed = time.time() - start
        results[name] = elapsed
        
        print(f"   耗时: {elapsed:.2f}秒")
        print(f"   速度: {num_regions/elapsed:.1f} 区域/秒")
    
    # 显示加速比
    print("\n🎯 性能总结:")
    baseline = results.get("单线程", 1)
    for name, elapsed in results.items():
        speedup = baseline / elapsed
        print(f"   {name}: {elapsed:.2f}秒 (加速 {speedup:.2f}x)")


if __name__ == "__main__":
    # 运行演示
    demo_parallel_ocr()
    
    # 性能测试（需要提供真实图像）
    # benchmark_parallel_performance("your_image.jpg", num_regions=100)
    
    print("\n" + "="*60)
    print("使用示例:")
    print("="*60)
    print("""
from parallel_ocr_regions import ParallelRegionOCR
import cv2

# 初始化并行处理器
processor = ParallelRegionOCR(
    num_workers=4,        # 使用4个工作器
    use_gpu=True,         # 使用GPU
    gpu_ids=[0, 1]        # 使用GPU 0和1
)

# 读取图像
image = cv2.imread("image.jpg")

# 定义区域
regions = [(x1,y1,x2,y2), ...]

# 并行识别
texts = processor.recognize_regions_parallel_thread(image, regions)
""")
