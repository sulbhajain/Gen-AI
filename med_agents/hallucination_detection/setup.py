from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hallucination-detection",
    version="0.1.0",
    author="Based on work by Gabriel Y. Arteaga, Thomas B. Schön, Nicolas Pielawski",
    description="Fast and memory-efficient hallucination detection for LLMs using BatchEnsemble",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hallucination-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "datasets>=2.0.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.39.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ],
    },
)
