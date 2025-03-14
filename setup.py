from setuptools import setup, find_packages

setup(

    name='fixout',
    version='0.1.21',
    description='Algorithmic inspection for trustworthy ML models',
    packages=find_packages(),
    package_data={'': ['web/templates/*.html',
                       'web/templates/utils-web/*.html',
                       'web/static/*.css',
                       'web/static/*.png',
                       'web/static/*.svg']},
    include_package_data=True,
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
    install_requires=[
        'Flask>=3.0.3',
        'Jinja2>=3.1.4',
        'numpy>=1.26.4',
        'pandas>=2.2.3',
        'plotly>=5.24.0',
        'scikit-image>=0.24.0',
        'scikit-learn>=1.5.2',
        'scipy>=1.11.4',
        'thrift>=0.20.0',
        'tinycss2>=1.4.0',
        'webcolors>=24.11.1'
    ],
)