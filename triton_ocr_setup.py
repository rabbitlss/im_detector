#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triton Server OCR部署配置
将UltraFastOCR模型部署到NVIDIA Triton Server
"""

import os
import json
import numpy as np
from pathlib import Path

def create_triton_config():
    """创建Triton模型配置文件"""
    
    config = """
name: "ocr_model"
platform: "python"
max_batch_size: 32
input [
  {
    name: "image"
    data_type: TYPE_UINT8
    dims: [-1, -1, 3]
  }
]
output [
  {
    name: "text"
    data_type: TYPE_STRING
    dims: [1]
  }
]

# 实例组配置 - 每个GPU一个实例
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  },
  {
    count: 1
    kind: KIND_GPU
    gpus: [1]
  }
]

# 动态批处理
dynamic_batching {
  max_queue_delay_microseconds: 100
}

# 模型预热
model_warmup [
  {
    name: "warmup"
    batch_size: 1
    inputs: {
      "image": {
        data_type: TYPE_UINT8
        dims: [32, 100, 3]
        random_data: true
      }
    }
  }
]

# 版本策略
version_policy: { latest { num_versions: 1 }}

# 优化配置
optimization {
  cuda {
    graphs: true
  }
}
"""
    return config

def create_model_py():
    """创建Triton Python后端模型文件"""
    
    model_code = '''# -*- coding: utf-8 -*-
"""
Triton Python Backend OCR Model
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils
from ultrafast_ocr.core import UltraFastOCR
import cv2

class TritonPythonModel:
    """Triton Python Backend Model for OCR"""
    
    def initialize(self, args):
        """初始化模型 - 只执行一次"""
        
        # 解析模型配置
        self.model_config = json.loads(args['model_config'])
        
        # 获取GPU ID
        self.device_id = int(args['model_instance_device_id'])
        
        print(f"[OCR Model] Initializing on GPU {self.device_id}")
        
        # 设置GPU
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_id)
        
        # 初始化OCR模型 - 只初始化一次！
        print(f"[OCR Model] Loading UltraFastOCR model...")
        self.ocr_model = UltraFastOCR()
        
        # 预热模型
        self._warmup()
        
        print(f"[OCR Model] ✅ Model loaded and warmed up on GPU {self.device_id}")
        print(f"[OCR Model] 💾 Model will stay in memory until server shutdown")
    
    def _warmup(self):
        """预热模型"""
        print(f"[OCR Model] Warming up model...")
        
        # 创建预热图像
        warmup_images = []
        for i in range(3):
            h, w = (32 + i*16, 100 + i*50)
            img = np.ones((h, w, 3), dtype=np.uint8) * 255
            warmup_images.append(img)
        
        # 预热推理
        for img in warmup_images:
            for _ in range(2):
                _ = self.ocr_model.recognize_single_line(img)
        
        print(f"[OCR Model] ✅ Warmup completed")
    
    def execute(self, requests):
        """执行推理 - 每次请求都调用"""
        
        responses = []
        
        for request in requests:
            # 获取输入
            image = pb_utils.get_input_tensor_by_name(request, "image")
            image_np = image.as_numpy()
            
            # OCR识别
            text = self.ocr_model.recognize_single_line(image_np)
            
            # 创建输出
            output_tensor = pb_utils.Tensor(
                "text",
                np.array([text], dtype=object)
            )
            
            # 创建响应
            response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(response)
        
        return responses
    
    def finalize(self):
        """清理资源 - 服务器关闭时调用"""
        print(f"[OCR Model] Releasing model from GPU {self.device_id}")
        del self.ocr_model
'''
    return model_code

def setup_triton_repository():
    """设置Triton模型仓库"""
    
    print("🚀 设置Triton Server OCR模型仓库")
    
    # 创建目录结构
    base_dir = Path("model_repository")
    model_dir = base_dir / "ocr_model"
    version_dir = model_dir / "1"
    
    # 创建目录
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 创建配置文件
    config_path = model_dir / "config.pbtxt"
    with open(config_path, 'w') as f:
        f.write(create_triton_config())
    print(f"✅ 创建配置文件: {config_path}")
    
    # 2. 创建模型文件
    model_path = version_dir / "model.py"
    with open(model_path, 'w') as f:
        f.write(create_model_py())
    print(f"✅ 创建模型文件: {model_path}")
    
    print("\n📁 模型仓库结构:")
    print("model_repository/")
    print("└── ocr_model/")
    print("    ├── config.pbtxt")
    print("    └── 1/")
    print("        └── model.py")
    
    print("\n🎯 下一步:")
    print("1. 启动Triton Server:")
    print("   docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \\")
    print("     -v ${PWD}/model_repository:/models \\")
    print("     nvcr.io/nvidia/tritonserver:23.10-py3 \\")
    print("     tritonserver --model-repository=/models")
    print("\n2. 使用客户端调用:")
    print("   python triton_ocr_client.py")

if __name__ == "__main__":
    setup_triton_repository()
