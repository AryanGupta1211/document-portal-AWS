from setuptools import setup, find_packages
from pathlib import Path

def parse_requirements(filename):
    with open(filename, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("-e")
        ]

setup(
    name="document_portal",
    author="Aryan Gupta",
    version="0.1",
    description="An intelligent document analysis and comparison system powered by LLMs",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "examples*"]),
    include_package_data=True,
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "dev": ["pytest", "pylint", "ipykernel"]
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
)