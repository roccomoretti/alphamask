from setuptools import setup, find_packages

setup(
    name="alphamask",
    version="0.1",
    packages=find_packages(),
    scripts=[
        'alphamask/scripts/setup_experiments.py',
        'alphamask/scripts/run_experiments.py',
        'predict.py'
    ],
    package_data={
        'alphamask': ['config/*.yaml', 'config/*.json'],
    },
    install_requires=[
        'pyyaml',
        'typing',
        'pathlib',
        'jsonschema',
    ],
) 