from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="satmae-denoiser",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-Spectral Satellite Image Denoising with SatMAE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/satmae-denoiser",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "satmae-train=scripts.train:main",
            "satmae-evaluate=scripts.evaluate:main",
            "satmae-export=scripts.export:main",
        ],
    },
)
