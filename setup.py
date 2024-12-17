from setuptools import setup, find_packages

setup(
    name="alphamask",
    version="0.1.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'alphamask=alphamask.cli.main:main',
            'alphamask-predict=predict:main',
            'alphamask-check-jax=check_jax:main',
        ],
    },
    package_data={
        'alphamask': [
            'config/*.yaml',
            'config/*.json',
            'docs/*.md'
        ],
    },
    install_requires=[
        'pyyaml',
        'typing-extensions>=4.12.0',
        'pathlib',
        'jsonschema',
        'numpy>=1.23,<2.0',
        'cudf-cu12>=24.4.0',
        'numba>=0.57,<0.61',
        'jax==0.4.26',
        'ipython',
        'plotly==5.24.1',
        'rich>=10.0.0',
    ],
    dependency_links=[
        "git+https://github.com/sokrypton/ColabDesign.git@70a821c877a1a1a7f09682b3b6272b72e0719975#egg=colabdesign",
        "git+https://github.com/engelberger/frustrapy.git@dev#egg=frustrapy",
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
