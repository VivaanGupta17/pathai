"""
PathAI Setup

Multiple Instance Learning for Whole Slide Image Analysis
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read long description from README
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
req_file = this_dir / "requirements.txt"
if req_file.exists():
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)

setup(
    name="pathai",
    version="1.0.0",
    author="PathAI Contributors",
    author_email="",
    description="Deep Learning for Whole Slide Image Analysis in Digital Pathology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/pathai",
    packages=find_packages(where=".", exclude=["tests*", "notebooks*", "scripts*"]),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "Pillow>=9.5.0",
        "opencv-python>=4.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "timm>=0.9.0",
    ],
    extras_require={
        "wsi": [
            "openslide-python>=1.2.0",
        ],
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "pre-commit>=3.3.0",
        ],
        "viz": [
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
            "tensorboard>=2.13.0",
        ],
        "all": [
            "openslide-python>=1.2.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
            "tensorboard>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pathai-extract=scripts.extract_features:main",
            "pathai-train=scripts.train:main",
            "pathai-evaluate=scripts.evaluate:main",
            "pathai-heatmap=scripts.generate_heatmap:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords=[
        "digital pathology",
        "computational pathology",
        "multiple instance learning",
        "whole slide image",
        "deep learning",
        "cancer detection",
        "CLAM",
        "TransMIL",
        "attention MIL",
        "Camelyon16",
        "TCGA",
        "histopathology",
        "H&E staining",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/pathai/issues",
        "Source": "https://github.com/your-username/pathai",
        "Documentation": "https://github.com/your-username/pathai/blob/main/docs/PATHOLOGY_AI.md",
    },
    include_package_data=True,
    zip_safe=False,
)
