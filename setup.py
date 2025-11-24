from setuptools import setup, find_packages

setup(
    name="tfrecx",
    version="0.1.1",
    author="Khaled Alkhouli",
    author_email="khaled.alkhouli03@gmail.com",
    description="Elegant toolkit for creating TFRecords from pandas DataFrames, images, and more.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Khoulii/tfrecx",
    packages=find_packages(exclude=["tests", "docs"]),
    entry_points={
        'console_scripts': [
            f'tfrx_csv2tfrec = tfrecx.cli:csv2tfrec',
            f'tfrx_json2tfrec = tfrecx.cli:json2tfrec',
            f'tfrx_head = tfrecx.cli:head',
        ]
    },
    python_requires=">=3.10",
    install_requires=[
        "tensorflow>=2.20.0",
        "pandas>=2.3.3",
        "numpy>=1.26.0",
        "Pillow>=11.3.0"
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries"
    ],
    include_package_data=True,
)

