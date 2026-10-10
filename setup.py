from setuptools import find_packages, setup

setup(
    name="het_control",
    version="0.0.1",
    description="Adaptive Diversity Control for cooperative MARL",
    url="https://github.com/Svar7769/AD2C-Diversity",
    author="Svar",
    author_email="svarrajankumar_patel@student.uml.edu",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch==2.10.0",
        "torchvision==0.25.0",
        "tensordict>=0.11,<0.12",
        "torchrl>=0.11,<0.12",
        "benchmarl==1.5.2",
        "vmas==1.5.2",
        "hydra-core>=1.3,<1.4",
        "numpy",
        "pyyaml",
        "matplotlib",
        "plotly",
        "wandb",
        "moviepy",
    ],
)
