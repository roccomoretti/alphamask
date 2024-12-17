Sure, here's the complete **Migration Guide** for transitioning your `alphamask` project to the new refactored structure. This guide outlines the key changes, provides examples for updating configurations, scripts, and packaging files, and ensures a smooth transition without breaking existing functionality.

---

# Migration Guide: AlphaMask Refactoring

## Overview

This guide helps you migrate from the old protein-specific configuration and script setup to the new flexible, configuration-driven system. The refactored structure enhances scalability, maintainability, and ease of use, allowing users to define any protein and its experiments through a unified YAML configuration.

## Key Changes

1. **Configuration Format**
2. **Command-Line Interface (CLI)**
3. **Setup and Runner Scripts**
4. **Packaging (`setup.py` and `pyproject.toml`)**
5. **Directory Structure**
6. **Dependency Management**
7. **Logging Enhancements**

---

## 1. Configuration Format

### Old Configuration (`config/test.yaml`)

Previously, configurations were directly tied to specific proteins with less abstraction.

```yaml
i89:
  sequence: "GSHMASMEDLQAEARAFLSEEMIAEFKAAFDMFDADGGGDISYKAVGTVFRMLGINPSKEVLDYLKEKIDVDGSGTIDFEEFLVLMVYIMKQDA"
  mutations:
    - "I81S"
  known_positions: [89]
  frustra_positions: [89] 
```

### New Configuration (`config/proteins.yaml`)

The new format introduces a hierarchical and strategy-based approach, allowing for more flexible and comprehensive configurations.

```yaml
proteins:
  i89:
    sequence: "GSHMASMEDLQAEARAFLSEEMIAEFKAAFDMFDADGGGDISYKAVGTVFRMLGINPSKEVLDYLKEKIDVDGSGTIDFEEFLVLMVYIMKQDA"
    iterative_masking:
      enabled: true
      mutations:
        - ["I89S"]
        - ["I89N"]
      mask_token: "X"
    apriori_masking:
      enabled: true
      experiments:
        - name: "Known_position_I89S"
          positions: [89]
          mutations: ["I89S"]
          conditions:
            - {mask: false, mutate: false}
            - {mask: true, mutate: false}
            - {mask: false, mutate: true}
            - {mask: true, mutate: true}
        - name: "Known_position_I89N"
          positions: [89]
          mutations: ["I89N"]
          conditions:
            - {mask: false, mutate: false}
            - {mask: true, mutate: false}
            - {mask: false, mutate: true}
            - {mask: true, mutate: true}
    frustra_masking:
      enabled: true
      top_positions: 10

# Global configuration
global_settings:
  mask_token: "X"
  output_dir: "./results"
  logging:
    enabled: true
    level: "INFO"
    file: "masking_analysis.log"
```

**Migration Steps:**

1. **Restructure YAML Files:**
   - Create a `proteins` key as the root.
   - Under each protein ID, define `sequence`, masking strategies (`iterative_masking`, `apriori_masking`, `frustra_masking`), and optional `uniprot_id`.

2. **Define Masking Strategies:**
   - **Iterative Masking:** Systematic position masking with defined mutations.
   - **A Priori Masking:** Controlled experiments based on known mutations and conditions.
   - **Frustra Masking:** Automated analysis based on protein frustration patterns.

---

## 2. Command-Line Interface (CLI)

### Old CLI Usage

Previously, you had separate console scripts for setup and running experiments.

```bash
python setup_experiments.py --path ./experiments
python run_experiments.py --container ./container.sif
```

### New CLI Usage

The refactored system consolidates commands under a unified `alphamask` CLI tool with subcommands.

```bash
# Setup experiments
alphamask setup --config config/proteins.yaml --path ./experiments

# Run experiments
alphamask run --config config/proteins.yaml --container ./container.sif --script ./predict.py --schema ./schema_validation.json
```

**Key Changes:**

- **Unified Entry Point:** All commands are accessible via the `alphamask` command.
- **Subcommands:** `setup` and `run` are now subcommands, enhancing clarity and organization.
- **Enhanced Arguments:** Additional arguments for flexibility and control.

---

## 3. Setup and Runner Scripts

### Old Scripts

- `alphamask/scripts/setup_experiments.py`
- `alphamask/scripts/run_experiments.py`

### New Structure

- **CLI Commands:** Handled by `alphamask/cli/main.py` and `alphamask/cli/commands.py`.
- **Configuration Management:** Handled by `alphamask/experiments/config.py`.
- **Experiment Classes:** Defined in `alphamask/experiments/base.py` and strategy-specific modules.

### Migration Steps:

1. **Remove Old Scripts:**
   - Delete or archive `alphamask/scripts/setup_experiments.py` and `alphamask/scripts/run_experiments.py`.

2. **Use New CLI:**
   - Use the `alphamask` CLI for all setup and run commands as outlined above.

---

## 4. Packaging (`setup.py` and `pyproject.toml`)

### Updated `setup.py`

```python
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
        'rich>=10.0.0',  # For better CLI output
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
```

**Key Changes:**

- **Unified CLI Entry Point:** Replaced multiple scripts with a single `alphamask` command.
- **Package Data Inclusion:** Added `docs/*.md` to include documentation within the package.
- **New Dependencies:** Added `rich` for enhanced CLI output.
- **Removed Redundant Scripts:** Removed `alphamask-setup` and `alphamask-run` in favor of the unified `alphamask` command.

### Updated `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=64.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "alphamask"
version = "0.1.1"
description = "TBD"
readme = "README.md"
authors = [
    {name = "Felipe Engelberger", email = "felipeengelberger@gmail.com"},
]
requires-python = ">=3.10"
license = {text = "MIT"}
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
dependencies = [
    "numpy>=1.23,<2.0",
    "cudf-cu12>=24.4.0",
    "numba>=0.57,<0.61",
    "jax==0.4.26",
    "ipython",
    "plotly==5.24.1",
    "pyyaml",
    "pathlib",
    "jsonschema",
    "typing-extensions>=4.12.0",
    "rich>=10.0.0",
]

[project.urls]
Homepage = "https://github.com/engelberger/alphamask"
Repository = "https://github.com/engelberger/alphamask"

[tool.setuptools]
packages = ["alphamask", "alphamask.cli"]

[tool.setuptools.package-data]
alphamask = [
    "py.typed",
    "config/*.yaml",
    "config/*.json",
    "docs/*.md"
]

[project.scripts]
alphamask = "alphamask.cli.main:main"
```

**Key Changes:**

- **Added CLI Package:** Included `"alphamask.cli"` in the `packages` list.
- **Included Documentation:** Added `"docs/*.md"` to `package-data` for packaging documentation.
- **Unified Script Entry Point:** Defined `alphamask` under `[project.scripts]` pointing to the new CLI.

---

## 5. Directory Structure

### Old Structure

```
alphamask/
├── experiments/
│   ├── base.py
│   ├── runner.py
│   └── mutations.py
├── scripts/
│   ├── setup_experiments.py
│   └── run_experiments.py
└── config/
    └── test.yaml
```

### New Structure

```
alphamask/
├── cli/
│   ├── __init__.py
│   ├── main.py
│   └── commands.py
├── experiments/
│   ├── base.py
│   ├── config.py
│   ├── runner.py
│   └── types.py
├── config/
│   ├── proteins.yaml
│   ├── defaults.yaml
│   ├── schema_validation.json
│   └── test.yaml
├── docs/
│   └── migration_guide.md
├── scripts/
│   ├── setup_experiments.py  # Optional: Can be removed if fully replaced by CLI
│   └── run_experiments.py    # Optional: Can be removed if fully replaced by CLI
└── ...
```

**Migration Steps:**

1. **Create CLI Directory:**
   - Move `setup_experiments.py` and `run_experiments.py` functionality to `alphamask/cli/`.
   - Implement `main.py` and `commands.py` for handling CLI commands.

2. **Update Experiments Directory:**
   - Add `types.py` and `config.py` for type definitions and configuration management.
   - Refactor existing experiment classes to use the new type system.

3. **Add Documentation:**
   - Create `docs/migration_guide.md` for comprehensive migration instructions.
   - Include additional documentation as needed.

4. **Organize Configurations:**
   - Structure `config/proteins.yaml` with the new hierarchical format.
   - Ensure all necessary configuration files are included in `package_data`.

---

## 6. Dependency Management

### New Dependencies

- **Rich:** Added for improved CLI aesthetics and functionality.

### Updated `install_requires` in `setup.py` and `dependencies` in `pyproject.toml`:

```python
# setup.py
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
    'rich>=10.0.0',  # Added for better CLI output
],
```

```toml
# pyproject.toml
dependencies = [
    "numpy>=1.23,<2.0",
    "cudf-cu12>=24.4.0",
    "numba>=0.57,<0.61",
    "jax==0.4.26",
    "ipython",
    "plotly==5.24.1",
    "pyyaml",
    "pathlib",
    "jsonschema",
    "typing-extensions>=4.12.0",
    "rich>=10.0.0",  # Added for better CLI output
]
```

**Migration Steps:**

1. **Install New Dependencies:**
   - Run `pip install rich` or ensure it's included in your environment.

2. **Update Packaging Files:**
   - Ensure `setup.py` and `pyproject.toml` reflect the new dependencies.

---

## 7. Logging Enhancements

### Old Logging

Basic logging setup without much flexibility.

```python
logger = logging.getLogger(__name__)
```

### New Logging

Enhanced logging with configurable levels and output destinations.

```python
def setup_logging(debug: bool, log_file: str = None):
    """Configure logging"""
    log_level = logging.DEBUG if debug else logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
```

**Migration Steps:**

1. **Implement Enhanced Logging:**
   - Update your scripts to configure logging based on CLI arguments.
   - Utilize the `logging` module more effectively across the project.

2. **Use `rich` for CLI Output:**
   - Integrate `rich` to provide more visually appealing logs and outputs in the terminal.

---

## Final Steps

1. **Backup Your Project:**
   - Before making extensive changes, ensure you have a backup or are using version control (e.g., Git).

2. **Update Configuration Files:**
   - Migrate all existing configurations to the new `proteins.yaml` format.
   - Ensure `global_settings` are correctly defined.

3. **Test the New Setup:**
   - Run the new setup command to create directory structures.
     ```bash
     alphamask setup --config config/proteins.yaml --path ./experiments
     ```
   - Execute experiments using the new run command.
     ```bash
     alphamask run --config config/proteins.yaml --container ./container.sif --script ./predict.py --schema ./schema_validation.json
     ```

4. **Validate and Iterate:**
   - Check logs for any errors or warnings.
   - Ensure all experiments run as expected without breaking existing functionality.

5. **Update Documentation:**
   - Reflect all changes in your project's README and other relevant documentation.
   - Ensure the `migration_guide.md` is comprehensive for users transitioning to the new system.

6. **Remove Deprecated Scripts (Optional):**
   - Once confirmed that the new CLI works flawlessly, consider removing old scripts to avoid confusion.

---

## Troubleshooting

- **Configuration Errors:**
  - Ensure all required fields in `proteins.yaml` are correctly defined.
  - Validate mutations and positions against the protein sequence.

- **Logging Issues:**
  - Verify log file paths and permissions.
  - Check if `rich` is properly installed and integrated.

- **Dependency Conflicts:**
  - Ensure all dependencies are compatible with each other.
  - Use virtual environments to manage dependencies effectively.

---

By following this migration guide, you should be able to transition your `alphamask` project to the new refactored structure seamlessly, leveraging enhanced flexibility, maintainability, and scalability.
