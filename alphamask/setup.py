from setuptools import setup, find_packages

setup(
    name="alphamask",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'alphamask-setup=alphamask.scripts.setup_experiments:run',
            'alphamask-run=alphamask.scripts.run_experiments:main',
            'alphamask-predict=predict:main',
            'alphamask-check-jax=check_jax:main',
        ],
    },
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