from setuptools import setup, find_packages

setup(
    name='fixout',
    version='0.1.16',
    description='Algorithmic inspection for trustworthy ML models',
    packages=find_packages(),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fixouttech/fixout",
    author="FixOut",
    license="BSD",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.2",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.12.9",
)