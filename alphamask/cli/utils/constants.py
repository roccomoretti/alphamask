
HELP_TEXTS = {
    "setup": """
        # AlphaMask Setup Command

        Sets up the experiment directory structure and resources.

        ## Usage
        ```bash
        alphamask setup [options]
        ```

        ## Options
        - `--config`: Path to protein configuration file (default: config/proteins.yaml)
        - `--path`: Base path for experiment setup (default: .)
        - `--setup-path`: Base path for setup files (default: ~/alphamask_setup)
        - `--force`: Force setup even if directories exist

        ## Example
        ```bash
        alphamask setup --config my_config.yaml --path /path/to/experiments
        ```
    """,
    "run": """
        # AlphaMask Run Command

        Runs protein experiments based on configuration.

        ## Usage
        ```bash
        alphamask run [options]
        ```

        ## Options
        - `--config`: Path to protein configuration file
        - `--proteins`: Specific proteins to run (optional)
        - `--container`: Path to Singularity container
        - `--script`: Path to prediction script
        - `--schema`: Path to JSON schema
        - `--partition`: SLURM partition (default: clara)
        - `--gpu-type`: GPU type to request (default: rtx2080ti)
        - `--force-local`: Force local execution

        ## Example
        ```bash
        alphamask run --config config.yaml --container container.sif --script predict.py --schema schema.json
        ```
    """,
    "config": """
        # AlphaMask Configuration Guide

        Configuration files use YAML format with the following structure:

        ```yaml
        proteins:
          protein_id:
            sequence: "PROTEIN_SEQUENCE"
            iterative_masking:
              enabled: true
              mutations: [["I89S"]]
            apriori_masking:
              enabled: true
              experiments:
                - name: "experiment_name"
                  positions: [89]
                  mutations: ["I89S"]
                  conditions:
                    - {mask: false, mutate: false}
                    - {mask: true, mutate: false}
                    - {mask: false, mutate: true}
                    - {mask: true, mutate: true}
            frustra_masking:
              enabled: true
              top_positions: 10

        global_settings:
          mask_token: "X"
          output_dir: "./results"
          logging:
            enabled: true
            level: "INFO"
            file: "masking_analysis.log"
        ```

        See migration guide for more details.
        """
}