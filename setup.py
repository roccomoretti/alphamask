from setuptools import setup, find_packages

setup(
    name="alphamask",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[

    ],
    author="Felipe Engelberger",
    author_email="felipeengelberger@gmail.com",
    description="TBD",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/engelberger/alphamask",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
