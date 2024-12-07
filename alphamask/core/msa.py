import os
import re
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from IPython import get_ipython
import subprocess
import tempfile
import gc
import jax
import jax.numpy as jnp

from colabdesign.af.contrib import predict

def is_colab_environment():
    """Check if code is running in Google Colab."""
    try:
        from google.colab import files
        return True
    except ImportError:
        return False

class PrepInputs:
    """
    Prepares input data for AlphaFold prediction.
    
    Attributes:
        sequence (str): Input sequence
        jobname (str): Name of the job
        copies (int): Number of copies
        msa_method (str): Method for MSA generation
        custom_a3m_path (str): Path to custom a3m file
        pair_mode (str): Pairing mode for MSA
        cov (int): Coverage threshold
        id (int): Identity threshold
        qid (int): Query identity threshold
        do_not_filter (bool): Whether to skip filtering
        template_mode (str): Mode for template usage
        pdb (str): PDB identifier
        chain (str): Chain identifier
        rm_template_seq (bool): Whether to remove template sequence
        propagate_to_copies (bool): Whether to propagate to copies
        do_not_align (bool): Whether to skip alignment
        setupPath (Path): Path to setup directory
        parentPath (Path): Path to parent directory
        overwrite (bool): Whether to overwrite existing files
        show_figures (bool): Whether to show figures
    """
    
    def __init__(
        self,
        sequence: str,
        jobname: str,
        copies: int,
        msa_method: str,
        custom_a3m_path: str,
        pair_mode: str,
        cov: int,
        id: int,
        qid: int,
        do_not_filter: bool,
        template_mode: str,
        pdb: str,
        chain: str,
        rm_template_seq: bool,
        propagate_to_copies: bool,
        do_not_align: bool,
        setupPath: str,
        parentPath: str,
        overwrite: bool,
        show_figures: bool,
    ):
        self.sequence = sequence
        self.jobname = jobname
        self.copies = copies
        self.msa_method = msa_method
        self.custom_a3m_path = Path(custom_a3m_path) if custom_a3m_path else None
        self.pair_mode = pair_mode
        self.cov = cov
        self.id = id
        self.qid = qid
        self.do_not_filter = do_not_filter
        self.template_mode = template_mode
        self.pdb = pdb
        self.chain = chain
        self.rm_template_seq = rm_template_seq
        self.propagate_to_copies = propagate_to_copies
        self.do_not_align = do_not_align
        self.setupPath = Path(setupPath)
        self.parentPath = Path(parentPath)
        self.overwrite = overwrite
        self.show_figures = show_figures
        
        # Initialize attributes that will be set later
        self.u_sequences: List[str] = []
        self.u_cyclic: List[bool] = []
        self.u_sub_lengths: List[List[int]] = []
        self.u_lengths: List[int] = []
        self.sub_seq: str = ""
        self.input_opts: Dict = {}
        self.msa: np.ndarray = None
        self.deletion_matrix: np.ndarray = None
        self.Ls: List[int] = []
        self.has_templates: bool = False
        self.batches: List = []
        
        # Check for required dependencies
        try:
            import requests
        except ImportError:
            raise ImportError("Please install requests: pip install requests")
            
        # Verify setupPath contains required files
        colabfold_utils = self.setupPath / "colabfold_utils.py"
        if not colabfold_utils.exists():
            print("WARNING: colabfold_utils.py not found, downloading from ColabFold...")
            import urllib.request
            url = "https://raw.githubusercontent.com/sokrypton/ColabFold/main/colabfold/colabfold.py"
            urllib.request.urlretrieve(url, str(colabfold_utils))

        # Initialize colab-specific attributes
        self.is_colab = is_colab_environment()
        if self.is_colab:
            from google.colab import files
            self.colab_files = files
            
    def filter_options(self) -> None:
        """Filter and validate input options."""
        self.sequence = self.sequence.upper()
        self.sequence = re.sub("[^A-Z:/()]", "", self.sequence)
        self.sequence = re.sub("\(", ":(", self.sequence)
        self.sequence = re.sub("\)", "):", self.sequence)
        self.sequence = re.sub(":+", ":", self.sequence)
        self.sequence = re.sub("/+", "/", self.sequence)
        self.sequence = re.sub("^[:/]+", "", self.sequence)
        self.sequence = re.sub("[:/]+$", "", self.sequence)
        self.jobname = re.sub(r"\W+", "", self.jobname)

    def process_sequence(self) -> None:
        """Process input sequence and set up job parameters."""
        sequences = self.sequence.split(":")
        self.u_sequences = predict.get_unique_sequences(sequences)
        self.u_cyclic = [x.startswith("(") for x in self.u_sequences]
        self.u_sub_lengths = [[len(y) for y in x.split("/")] for x in self.u_sequences]
        self.u_sequences = [
            x.replace("(", "").replace(")", "").replace("/", "")
            for x in self.u_sequences
        ]
        
        if len(sequences) > len(self.u_sequences):
            print("WARNING: use copies to define homooligomers")
            
        self.u_lengths = [len(x) for x in self.u_sequences]
        self.sub_seq = "".join(self.u_sequences)
        seq = self.sub_seq * self.copies

        self.jobname = f"{self.jobname}_{predict.get_hash(seq)[:5]}"
        self._handle_existing_job()
        
        print("jobname", self.jobname)
        print(f"length={self.u_lengths} copies={self.copies}")

        self.input_opts = {
            "sequence": self.u_sequences,
            "copies": self.copies,
            "msa_method": self.msa_method,
            "pair_mode": self.pair_mode,
            "do_not_filter": self.do_not_filter,
            "cov": self.cov,
            "id": self.id,
            "template_mode": self.template_mode,
            "propagate_to_copies": self.propagate_to_copies,
        }
        print(self.input_opts)

    def _handle_existing_job(self) -> None:
        """Handle cases where job directory already exists."""
        def check(folder: str) -> bool:
            return (self.parentPath / folder).exists()

        if check(self.jobname):
            n = 0
            while check(f"{self.jobname}_{n}"):
                n += 1
                
            if self.overwrite:
                print(
                    f"WARNING: {self.jobname} already exists. Using the same jobname. "
                    "If you want to run the job with a different jobname, set overwrite to False. "
                    "If you did not change other parameters your files will be overwritten."
                )
            else:
                print(f"WARNING: {self.jobname} already exists. Changing jobname to {self.jobname}_{n}")
                self.jobname = f"{self.jobname}_{n}"

    def get_msa(self) -> None:
        """Get Multiple Sequence Alignment."""
        def run_mmseqs2_wrapper(*args, **kwargs):
            kwargs["user_agent"] = "colabdesign/gamma"
            return run_mmseqs2(*args, **kwargs)

        os.makedirs(self.parentPath / self.jobname, exist_ok=True)
        input_path = self.parentPath / self.jobname / "in"
        input_path.mkdir(exist_ok=True)
        
        self.Ls = [len(x) for x in self.u_sequences]
        
        if self.msa_method == "mmseqs2":
            self._handle_mmseqs2_msa(input_path, run_mmseqs2_wrapper)
        elif self.msa_method == "single_sequence":
            self._handle_single_sequence_msa()
        else:
            self._handle_custom_msa()

        if len(self.msa) > 1:
            predict.plot_msa(self.msa, self.Ls)
            plt.savefig(
                input_path / "msa_feats.png",
                dpi=200,
                bbox_inches="tight",
            )
            if "ipykernel" in sys.modules and self.show_figures:
                plt.show()
            else:
                plt.close()

    def _handle_mmseqs2_msa(self, input_path: Path, run_mmseqs2_wrapper: Any) -> None:
        """Handle MMseqs2 MSA generation."""
        msa_file = input_path / "msa.a3m"
        if msa_file.exists():
            print("msa.a3m file is already present, loading from file.")
            print("please check that this a3m file contains the sequences you expect")
            print("this behaviour was implemented for HPC Slurm cluster usage")
            self.msa, self.deletion_matrix = predict.parse_a3m(msa_file)
        else:
            from alphamask.utils.colabfold_utils import run_mmseqs2
            from alphamask.core.setup import ColabDesignUtils
            
            utils = ColabDesignUtils(self.setupPath)
            
            self.msa, self.deletion_matrix = predict.get_msa(
                self.u_sequences,
                input_path,
                mode=self.pair_mode,
                cov=self.cov,
                id=self.id,
                qid=self.qid,
                max_msa=4096,
                do_not_filter=self.do_not_filter,
                mmseqs2_fn=run_mmseqs2,
                hhfilter_fn=utils.run_hhfilter,
            )

    def _handle_single_sequence_msa(self) -> None:
        """Handle single sequence MSA generation."""
        msa_file = self.parentPath / self.jobname / "in/msa.a3m"
        with open(msa_file, "w") as a3m:
            a3m.write(f">{self.jobname}\n{self.sub_seq}\n")
        self.msa, self.deletion_matrix = predict.parse_a3m(msa_file)

    def _handle_custom_msa(self) -> None:
        """Handle custom MSA file processing."""
        msa_format = self.msa_method.split("_")[1]
        print(f"MSA mode: {self.msa_method}")
        
        print(f"google_colab: {self.is_colab}")
        print(f"local_run: {not self.is_colab}")

        if self.is_colab and not self.custom_a3m_path:
            print("WARNING: uploading MSA file via google colab api")
            msa_dict = self.colab_files.upload()
            lines = []
            for k, v in msa_dict.items():
                lines += v.decode().splitlines()
            self.custom_a3m_path = k
        
        self._process_custom_msa(msa_format)

    def _process_custom_msa(self, msa_format: str) -> None:
        """Process custom MSA file."""
        if not self.custom_a3m_path.exists():
            raise ValueError(f"Invalid path: {self.custom_a3m_path}. File does not exist.")
            
        with open(self.custom_a3m_path, "r") as file:
            lines = [line.replace("\x00", "") for line in file.readlines()]
            input_lines = [line for line in lines if line.strip() and not line.startswith("#")]

        input_path = self.parentPath / self.jobname / "in"
        msa_file = input_path / f"msa.{msa_format}"
        
        if not msa_file.exists():
            with open(msa_file, "w") as msa:
                msa.write("\n".join(input_lines))

        if msa_format != "a3m":
            os.system(
                f"perl hhsuite/scripts/reformat.pl {msa_format} a3m {msa_file} {input_path/'msa.a3m'}"
            )

        self._filter_msa(input_path)

    def _filter_msa(self, input_path: Path) -> None:
        """Filter MSA based on parameters."""
        if "hhsuite" not in os.environ["PATH"]:
            os.environ["PATH"] += f":{self.setupPath/'hhsuite/bin'}:{self.setupPath/'hhsuite/scripts'}"

        filt_file = input_path / "msa.filt.a3m"
        if not filt_file.exists():
            if self.do_not_filter:
                print("WARNING: not filtering MSA. Using 0 cov, 0 qid and 100 id")
                os.system(
                    f"hhfilter -qid 0 -id 100 -cov 0 -i {input_path/'msa.a3m'} -o {filt_file}"
                )
            else:
                print(f"Filtering MSA with HHFilter. Using {self.cov} cov, {self.qid} qid and {self.id} id")
                try:
                    command = [
                        "hhfilter",
                        "-qid", str(self.qid),
                        "-id", str(self.id),
                        "-cov", str(self.cov),
                        "-i", str(input_path/"msa.a3m"),
                        "-o", str(filt_file)
                    ]
                    result = subprocess.run(
                        command,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    print(f"Running command: {' '.join(command)}")
                    print("hhfilter ran successfully:")
                    print(result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"Error running hhfilter: {e}")
                    print(f"hhfilter stderr output:\n{e.stderr}")
                except FileNotFoundError:
                    print("hhfilter command not found. Make sure it's installed and in your PATH.")

        self.msa, self.deletion_matrix = predict.parse_a3m(filt_file) 

    def use_templates(self):
        """Process and set up templates if needed."""
        self.has_templates = self.template_mode in ["mmseqs2", "custom"]
        
        if self.has_templates:
            print("aligning template")
            template_msa = f"{self.parentPath}/{self.jobname}/in/msa.a3m"
            if self.template_mode == "mmseqs2":
                # Import here to avoid circular imports
                from alphamask.utils.colabfold_utils import run_mmseqs2
                
                predict.get_msa(
                    self.u_sequences,
                    self.jobname,
                    mode="unpaired",
                    mmseqs2_fn=lambda *x: run_mmseqs2(*x, user_agent="colabdesign/gamma"),
                    do_not_filter=True,
                    do_not_return=True,
                    output_a3m=f"{self.parentPath}/{self.jobname}/in/msa_tmp.a3m",
                )
                template_msa = f"{self.parentPath}/{self.jobname}/in/msa_tmp.a3m"
                if not self.propagate_to_copies and self.copies > 1:
                    new_msa = []
                    with open(template_msa, "r") as handle:
                        for line in handle:
                            if not line.startswith(">"):
                                new_msa.append(line.rstrip())
                    with open(template_msa, "w") as handle:
                        for n, seq in enumerate(new_msa):
                            handle.write(f">{n}\n{seq*self.copies}\n")

                templates = {}
                print("ID\tpdb\tcid\tevalue")
                for line in open(f"{self.parentPath}/{self.jobname}/in/msa/_env/pdb70.m8", "r"):
                    p = line.rstrip().split()
                    M, target_id, qid, e_value = p[0], p[1], p[2], p[10]
                    M = int(M)
                    if M not in templates:
                        templates[M] = []
                    if len(templates[M]) < 4:
                        print(f"{int(M)}\t{target_id}\t{qid}\t{e_value}")
                        templates[M].append(target_id)
                if len(templates) == 0:
                    self.has_templates = False
                    print("ERROR: no templates found...")
                else:
                    Ms = sorted(list(templates.keys()))
                    pdbs, chains = [], []
                    for M in Ms:
                        for n, target_id in enumerate(templates[M]):
                            pdb_id, chain_id = target_id.split("_")
                            if len(pdbs) < n + 1:
                                pdbs.append([])
                                chains.append([])
                            pdbs[n].append(pdb_id)
                            chains[n].append(chain_id)
                    print(pdbs)
            else:
                pdbs, chains = [self.pdb], [self.chain]

            if self.has_templates:
                def get_pdb_local(pdb_code):
                    if pdb_code is None or pdb_code == "":
                        if self.is_colab:
                            upload_dict = self.colab_files.upload()
                            pdb_string = upload_dict[list(upload_dict.keys())[0]]
                            with open("tmp.pdb", "wb") as out:
                                out.write(pdb_string)
                            return "tmp.pdb"
                        else:
                            raise ValueError("PDB code cannot be empty in local environment")
                    elif os.path.isfile(pdb_code):
                        return pdb_code
                    elif len(pdb_code) == 4:
                        os.makedirs("tmp", exist_ok=True)
                        os.system(
                            f"wget -qnc https://files.rcsb.org/download/{pdb_code}.cif -P tmp/"
                        )
                        return f"tmp/{pdb_code}.cif"
                    else:
                        os.makedirs("tmp", exist_ok=True)
                        os.system(
                            f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v4.pdb -P tmp/"
                        )
                        return f"tmp/AF-{pdb_code}-F1-model_v4.pdb"

                def run_hhalign(query_sequence, target_sequence, query_a3m=None, target_a3m=None):
                    with tempfile.NamedTemporaryFile() as tmp_query, tempfile.NamedTemporaryFile() as tmp_target, tempfile.NamedTemporaryFile() as tmp_alignment:
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
                        X, start_indices = predict.parse_hhalign_output(tmp_alignment.name)
                    return X, start_indices

                def run_do_not_align(query_sequence, target_sequence, **kwargs):
                    return [query_sequence, target_sequence], [0, 0]

                self.input_opts.update({"pdbs": pdbs, "chains": chains})
                self.batches = []
                for pdb, chain in zip(pdbs, chains):
                    query_seq = "".join(self.u_sequences)
                    if "hhsuite" not in os.environ["PATH"]:
                        os.environ["PATH"] += f":{self.setupPath/'hhsuite/bin'}:{self.setupPath/'hhsuite/scripts'}"
                    
                    import shutil
                    assert shutil.which("hhalign") is not None, "hhalign not found. Something may have failed during the setup step."
                    
                    batch = predict.get_template_feats(
                        pdb,
                        chain,
                        query_seq=query_seq,
                        query_a3m=template_msa,
                        copies=self.copies,
                        propagate_to_copies=self.propagate_to_copies,
                        use_seq=not self.rm_sequence,
                        get_pdb_fn=get_pdb_local,
                        align_fn=run_do_not_align if self.do_not_align else run_hhalign,
                    )
                    self.batches.append(batch)

                plt.figure(figsize=(3 * len(self.batches), 3))
                for n, batch in enumerate(self.batches):
                    plt.subplot(1, len(self.batches), n + 1)
                    plt.title(f"template features {n+1}")
                    dgram = batch["dgram"].argmax(-1).astype(float)
                    dgram[batch["dgram"].sum(-1) == 0] = np.nan
                    Ln = dgram.shape[0]
                    plt.imshow(dgram, extent=(0, Ln, Ln, 0))
                    predict.plot_ticks(self.Ls * self.copies)
                plt.savefig(
                    f"{self.parentPath}/{self.jobname}/in/template_feats.png",
                    dpi=200,
                    bbox_inches="tight",
                )
                if "ipykernel" in sys.modules and self.show_figures:
                    plt.show()
                else:
                    plt.close()
        else:
            self.batches = [None]

        print("GC", gc.collect())

class MSAUtils:
    """
    Utility class for MSA manipulation and visualization.
    
    Attributes:
        restypes (List[str]): List of standard amino acid types
        restypes_with_x_and_gap (List[str]): List of amino acids plus X and gap
        offset_zero_indexing (int): Offset for zero-based indexing
    """
    
    def __init__(self):
        self.restypes = [
            "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
            "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
        ]
        self.restypes_with_x_and_gap = self.restypes + ["X", "-"]
        self.offset_zero_indexing = 1

    def mask_columns_list_msa(
        self, 
        arr: np.ndarray, 
        cols: List[int], 
        mask_identity: str
    ) -> np.ndarray:
        """
        Mask specific columns in MSA.
        
        Args:
            arr: MSA array
            cols: List of columns to mask
            mask_identity: Character to use for masking
            
        Returns:
            Masked MSA array
        """
        arr = arr.copy()
        
        if not cols:
            raise ValueError("The list of columns cannot be empty.")
        if mask_identity not in self.restypes_with_x_and_gap:
            raise ValueError(
                f"Invalid mask_identity: {mask_identity}. Must be one of {self.restypes_with_x_and_gap}"
            )
            
        mask = self.restypes_with_x_and_gap.index(mask_identity)
        cols = [col - 1 for col in cols]  # Convert to zero indexing
        arr[1:, cols] = mask
        return arr

    def mask_columns_list_deletion_matrix(
        self, 
        arr: np.ndarray, 
        cols: List[int]
    ) -> np.ndarray:
        """
        Mask specific columns in deletion matrix.
        
        Args:
            arr: Deletion matrix array
            cols: List of columns to mask
            
        Returns:
            Masked deletion matrix array
        """
        arr = arr.copy()
        
        if not cols:
            raise ValueError("The list of columns cannot be empty.")
            
        mask = 0
        cols = [col - 1 for col in cols]
        arr[1:, cols] = mask
        return arr

    def mutate_first_sequence(
        self, 
        arr: np.ndarray, 
        mutations: List[str]
    ) -> np.ndarray:
        """
        Apply mutations to the first sequence in MSA.
        
        Args:
            arr: MSA array
            mutations: List of mutations in format 'A123B'
            
        Returns:
            MSA array with mutations applied
        """
        arr = arr.copy()
        
        for mutation in mutations:
            if not re.match(r"^[A-Z]\d+[A-Z]$", mutation):
                raise ValueError(f"Invalid mutation format: {mutation}. Expected format: S146D")

            original_residue = mutation[0]
            position = int(mutation[1:-1]) - self.offset_zero_indexing
            new_residue = mutation[-1]

            if position < 0 or position >= arr.shape[1]:
                raise ValueError(f"Invalid residue position: {position + self.offset_zero_indexing}")

            if self.restypes_with_x_and_gap[int(arr[0, position])] != original_residue:
                raise ValueError(
                    f"Original residue mismatch at position {position + self.offset_zero_indexing}"
                )

            if new_residue not in self.restypes:
                raise ValueError(f"Invalid new residue: {new_residue}. Must be one of {self.restypes}")

            arr[0, position] = self.restypes.index(new_residue)
            print(f"Mutated residue {original_residue}{position + self.offset_zero_indexing}{new_residue}")
            
        return arr

    def mask_mutated_positions(
        self, 
        arr: np.ndarray, 
        mutations: List[str], 
        mask_identity: str
    ) -> np.ndarray:
        """
        Mask positions that have been mutated.
        
        Args:
            arr: MSA array
            mutations: List of mutations
            mask_identity: Character to use for masking
            
        Returns:
            MSA array with mutated positions masked
        """
        arr = arr.copy()
        mutated_positions = []
        
        for mutation in mutations:
            if not re.match(r"^[A-Z]\d+[A-Z]$", mutation):
                raise ValueError(f"Invalid mutation format: {mutation}. Expected format: S146D")
                
            print(f"Mutation before zero indexing: {mutation}")
            position = int(mutation[1:-1]) - self.offset_zero_indexing
            print(f"Mutation after zero indexing: {mutation[0]}{position + 1}{mutation[-1]}")
            mutated_positions.append(position)

        if mask_identity not in self.restypes_with_x_and_gap:
            raise ValueError(
                f"Invalid mask_identity: {mask_identity}. Must be one of {self.restypes_with_x_and_gap}"
            )

        mask_index = self.restypes_with_x_and_gap.index(mask_identity)
        arr[1:, mutated_positions] = mask_index
        
        print(f"Masked positions (mutation): {', '.join(map(lambda x: str(x + 1), mutated_positions))}")
        return arr

    def mask_mutated_positions_deletion_matrix(
        self, 
        arr: np.ndarray, 
        mutations: List[str]
    ) -> np.ndarray:
        """
        Mask mutated positions in deletion matrix.
        
        Args:
            arr: Deletion matrix array
            mutations: List of mutations
            
        Returns:
            Deletion matrix with mutated positions masked
        """
        arr = arr.copy()
        mutated_positions = []
        
        for mutation in mutations:
            if not re.match(r"^[A-Z]\d+[A-Z]$", mutation):
                raise ValueError(f"Invalid mutation format: {mutation}. Expected format: S146D")
                
            # Convert to zero-based indexing
            position = int(mutation[1:-1]) - self.offset_zero_indexing
            mutated_positions.append(position)
            
        arr[1:, mutated_positions] = 0
        # Print positions in 1-based indexing for consistency with user input
        print(f"Masked positions (deletion matrix): {', '.join(map(lambda x: str(x + 1), mutated_positions))}")
        return arr

    def convert_to_letters(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert numeric MSA array to letter representation.
        
        Args:
            arr: Numeric MSA array
            
        Returns:
            MSA array with letter representation
        """
        return np.array([
            list(map(lambda x: self.restypes_with_x_and_gap[int(x)], row))
            for row in arr
        ])

    def plot_2d_array(
        self,
        array: np.ndarray,
        title: str,
        xaxis_title: str,
        yaxis_title: str,
        save_to_pdf: Optional[str] = None,
    ) -> None:
        """Plot 2D array visualization."""
        # Define custom colorscale
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#1a55FF', '#55a2FF', '#55FFB1', '#a2FF55', '#FFEA1a',
            '#FF551a', '#FF1a55', '#FF1aa3', '#B51aFF', '#1a8CFF',
            '#1aFF55', '#7F1aFF'
        ]
        
        # Create proper colorscale format
        n_colors = len(colors)
        colorscale = [
            [i/(n_colors-1), color] for i, color in enumerate(colors)
        ]
        
        import plotly.graph_objects as go
        
        fig = go.Figure(data=go.Heatmap(
            z=array,
            colorscale=colorscale,
            showscale=True
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            autosize=False,
            width=500,
            height=500,
            margin=dict(l=65, r=50, b=65, t=90),
        )

        if save_to_pdf:
            # Create directory if it doesn't exist
            save_path = Path(save_to_pdf)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(save_path))

        fig.show()

    def get_coevolution(self, msa_array: np.ndarray) -> np.ndarray:
        """
        Calculate coevolution matrix from Multiple Sequence Alignment (MSA).
        
        This method implements Direct Coupling Analysis (DCA) to identify 
        coevolving residue pairs in protein sequences. The steps are:
        1. Convert MSA to one-hot encoding
        2. Calculate covariance matrix
        3. Calculate inverse covariance (precision matrix)
        4. Convert to partial correlation coefficients
        5. Apply Average Product Correction (APC) to reduce bias
        
        This was rewritten based on Sergey's implementation in ColabDesign.
        
        Args:
            msa_array: MSA array of shape (num_sequences, sequence_length)
                where each element is an integer representing an amino acid
            
        Returns:
            contact_scores: Matrix of shape (sequence_length, sequence_length)
                containing coevolution scores between residue pairs
        """
        @jax.jit
        def _calculate_coevolution(msa_array):
            # Convert MSA to one-hot encoding
            # 22 represents 20 amino acids + gap + unknown
            one_hot_msa = jax.nn.one_hot(msa_array, num_classes=22)
            num_sequences, sequence_length, num_amino_acids = one_hot_msa.shape
            
            # Reshape to 2D matrix for covariance calculation
            # Each row represents a sequence, each column a position-amino_acid pair
            flattened_msa = one_hot_msa.reshape(num_sequences, -1)
            
            # Calculate covariance matrix
            covariance_matrix = jnp.cov(flattened_msa.T)
            
            # Add shrinkage to ensure matrix is invertible
            # Shrinkage parameter scales with 1/sqrt(N) where N is number of sequences
            shrinkage_factor = 4.5 / jnp.sqrt(num_sequences)
            shrinkage_matrix = shrinkage_factor * jnp.eye(covariance_matrix.shape[0])
            regularized_covariance = covariance_matrix + shrinkage_matrix
            
            # Calculate inverse covariance (precision matrix)
            precision_matrix = jnp.linalg.inv(regularized_covariance)
            
            # Convert to partial correlation coefficients
            # This normalizes the precision matrix by its diagonal elements
            precision_diag = jnp.diag(precision_matrix)
            partial_correlations = precision_matrix / jnp.sqrt(
                precision_diag[:, None] * precision_diag[None, :]
            )
            
            # Reshape and compute coupling scores
            # Only consider the first 20 amino acids (exclude gap and unknown)
            reshaped_correlations = partial_correlations.reshape(
                sequence_length, num_amino_acids,
                sequence_length, num_amino_acids
            )
            coupling_scores = jnp.sqrt(
                jnp.square(reshaped_correlations[:, :20, :, :20]).sum((1, 3))
            )
            
            # Zero out diagonal (self-contacts)
            positions = jnp.arange(sequence_length)
            coupling_scores = coupling_scores.at[positions, positions].set(0)
            
            # Apply Average Product Correction (APC)
            # This helps remove background noise and phylogenetic bias
            row_means = coupling_scores.sum(0, keepdims=True)  # Mean per column
            col_means = coupling_scores.sum(1, keepdims=True)  # Mean per row
            matrix_mean = coupling_scores.sum()  # Overall mean
            
            # APC correction: subtract product of row/col means divided by matrix mean
            apc_correction = (row_means * col_means) / matrix_mean
            corrected_scores = coupling_scores - apc_correction
            
            # Zero out diagonal again after APC correction
            return corrected_scores.at[positions, positions].set(0)
        
        # Convert JAX array to numpy array for compatibility
        return np.array(_calculate_coevolution(msa_array)) 