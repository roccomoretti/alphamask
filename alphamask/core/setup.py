import os
import subprocess
import tempfile
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import shutil
class SetupAlphaFoldColabDesign:
    """
    Sets up AlphaFold and ColabDesign environment.
    
    Attributes:
        unified_memory (bool): Whether to use unified memory
        parent_path (str): Path to parent directory
        setup_path (str): Path to setup directory
        python_colab (str): Path to Python interpreter
        colabdesign_path (str): Path to ColabDesign installation
        ENV (Dict[str, str]): Environment variables
    """
    
    def __init__(
        self,
        unified_memory: bool,
        parent_path: str,
        setup_path: str,
        python_colab: str = "/usr/bin/python3.10",
        colabdesign_path: str = "/usr/local/lib/python3.10/dist-packages/colabdesign",
    ):
        self.unified_memory = unified_memory
        self.parent_path = Path(parent_path)
        self.setup_path = Path(setup_path)
        self.python_colab = python_colab
        self.colabdesign_path = colabdesign_path
        self.ENV = (
            {"TF_FORCE_UNIFIED_MEMORY": "1", "XLA_PYTHON_CLIENT_MEM_FRACTION": "4.0"}
            if unified_memory
            else {}
        )

    def setup(self) -> None:
        """Set up the environment and install required components."""
        self._set_environment()
        self._create_directories()
        self._check_and_install_components()

    def _set_environment(self) -> None:
        """Set environment variables."""
        for k, v in self.ENV.items():
            os.environ[k] = v

    def _create_directories(self) -> None:
        """Create necessary directories."""
        self.setup_path.mkdir(exist_ok=True)

    def _check_and_install_components(self) -> None:
        """Check and install required components."""
        if (self.setup_path / "params").is_dir():
            print("Setup path is present.")
        else:
            print("Setup path is not present. Installing ColabDesign...")
            self.install_colab_design()

        if (self.setup_path / "hhsuite").is_dir():
            print("HHsuite is present.")
        else:
            print("HHsuite is not present. Installing HHsuite...")
            self.install_hhsuite()

        print("mmseqs2 is imported.")

    def install_colab_design(self) -> None:
        """Install ColabDesign and its dependencies."""
        def run_command(command: str) -> bytes:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
            output, error = process.communicate()

            if error:
                raise Exception(f"Error occurred while executing command: {error}")

            return output

        params_path = self.setup_path / "params"
        params_path.mkdir(exist_ok=True)
        print(f"Created params directory at {params_path}")

        print("Installing ColabDesign...")

        commands = [
            f"apt-get install aria2 -qq",
            f"cd {self.setup_path} && aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
            f"tar -xf {self.setup_path}/alphafold_params_2022-12-06.tar -C {params_path}",
            f"touch {params_path/'done.txt'}",
        ]

        for command in commands:
            run_command(command)

        print("Installing Python dependencies...")
        run_command(
            f"{self.python_colab} -m pip -q install git+https://github.com/sokrypton/ColabDesign.git@gamma"
        )
        run_command(
            f"ln -s {self.colabdesign_path} {self.setup_path/'colabdesign'}"
        )
        run_command(
            f"wget https://raw.githubusercontent.com/sokrypton/ColabFold/main/colabfold/colabfold.py -O {self.setup_path/'colabfold_utils.py'}"
        )

    def install_hhsuite(self) -> None:
        """Install HHsuite."""
        hhsuite_path = self.setup_path / "hhsuite"
        hhsuite_path.mkdir(exist_ok=True)
        
        os.system(
            f"curl -fsSL https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-SSE2-Linux.tar.gz | tar xz -C {hhsuite_path}"
        )

        if "hhsuite" not in os.environ["PATH"]:
            os.environ["PATH"] += f":{hhsuite_path/'bin'}:{hhsuite_path/'scripts'}"


class ColabDesignUtils:
    """
    Utility functions for ColabDesign.
    """
    
    def __init__(self, setup_path: str):
        self.setup_path = Path(setup_path)

    def run_hhalign(
        self, 
        query_sequence: str, 
        target_sequence: str, 
        query_a3m: Optional[str] = None, 
        target_a3m: Optional[str] = None
    ) -> Tuple[List, List]:
        """
        Run HHalign on sequences.
        
        Args:
            query_sequence: Query sequence
            target_sequence: Target sequence
            query_a3m: Path to query a3m file
            target_a3m: Path to target a3m file
            
        Returns:
            Tuple containing alignment results and start indices
        """
        with tempfile.NamedTemporaryFile() as tmp_query, \
             tempfile.NamedTemporaryFile() as tmp_target, \
             tempfile.NamedTemporaryFile() as tmp_alignment:
            
            if query_a3m is None:
                tmp_query.write(f">Q\n{query_sequence}\n".encode())
                tmp_query.flush()
                query_a3m = tmp_query.name
                
            if target_a3m is None:
                tmp_target.write(f">T\n{target_sequence}\n".encode())
                tmp_target.flush()
                target_a3m = tmp_target.name
                
            os.system(
                f"hhalign -hide_cons -i {query_a3m} -t {target_a3m} -o {tmp_alignment.name}"
            )
            
            from colabdesign.af.contrib import predict
            X, start_indices = predict.parse_hhalign_output(tmp_alignment.name)
            
        return X, start_indices

    def run_hhfilter(self, input: str, output: str, id: int = 90, qid: int = 10) -> None:
        """
        Run HHfilter on input file.
        
        Args:
            input: Input file path
            output: Output file path
            id: Maximum pairwise sequence identity (%)
            qid: Minimum sequence identity with query (%)
        """
        print(f"Current PATH: {os.environ['PATH']}")
        print(f"hhfilter location: {shutil.which('hhfilter')}")

        if shutil.which('hhfilter') is None:
            print("hhfilter not found in PATH. Adding it now.")
            hhsuite_bin = self.setup_path / 'hhsuite/bin'
            hhsuite_scripts = self.setup_path / 'hhsuite/scripts'
            os.environ["PATH"] = f"{os.environ['PATH']}:{hhsuite_bin}:{hhsuite_scripts}"
            print(f"Updated PATH: {os.environ['PATH']}")
            print(f"hhfilter location after update: {shutil.which('hhfilter')}")

        os.system(f"hhfilter -id {id} -qid {qid} -i {input} -o {output}") 