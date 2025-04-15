from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="competitive-pricing-strategy",
    version="0.1.0",
    description="A machine learning-based system for competitive pricing strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Competitive Pricing Team",
    author_email="example@domain.com",
    url="https://github.com/yourusername/competitive-pricing-strategy",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Business/Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
) 