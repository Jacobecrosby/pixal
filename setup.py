from setuptools import setup, find_packages

setup(
    name="pixal",
    version="0.1.0",
    description="PIXel-based Anomaly Detection Tool for ATLAS Components",
    author="Dr. Jacob E. Crosby",
    packages=find_packages(where="pixal"),
    package_dir={"": "pixal"},
    install_requires=[
        "rembg",
        "pillow",
        "paramiko",
        "tensorflow",
        "numpy",
        "argparse",
        "openhtf",
        "pandas",
        "pymongo",
        "pyyaml",
        "scikit-learn",
        "opencv-python",
        "numba",
        "matplotlib",
        "tqdm",
        "onnxruntime-gpu",
        "pyyaml",
    ],
    entry_points={
    "console_scripts": [
        "pixal=cli:main",
    ]
    },
    include_package_data=True,
)
