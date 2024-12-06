from setuptools import setup, find_packages

setup(
    name="alphamask",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "biopython>=1.84",
        "igraph>=0.11.8",
        "leidenalg>=0.10.2",
        "kaleido==0.2.1",
        "numpy>=2.1.3",
        "texttable>=1.7.0",
        # Note: ColabDesign and frustrapy are installed from git repos
        # and should be installed separately or listed as dependency links
    ],
    dependency_links=[
        "git+https://github.com/sokrypton/ColabDesign.git@gamma#egg=colabdesign",
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
