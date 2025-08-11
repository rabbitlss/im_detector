# -*- coding: utf-8 -*-
"""
安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="im-detector",
    version="1.0.0",
    author="IM Detector Team",
    author_email="contact@example.com",
    description="IM界面元素检测器：基于GPT-4V自动标注和YOLO训练",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/im-detector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "acceleration": [
            "onnx>=1.14.0",
            "onnxruntime-gpu>=1.15.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "im-detector=im_detector.end2end_pipeline:main",
            "im-detector-demo=im_detector.end2end_pipeline:demo_usage",
            "im-detector-label=im_detector.auto_labeler:main",
            "im-detector-train=im_detector.yolo_trainer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "im_detector": ["config/*.yaml", "templates/*.json"],
    },
    zip_safe=False,
)
