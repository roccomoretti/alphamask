import os
import re
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
from IPython import get_ipython
import subprocess
import tempfile
import gc
import jax
import jax.numpy as jnp
import logging

# Configure logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
plt.set_loglevel('warning')  # This will suppress matplotlib debug messages

from colabdesign.af.contrib import predict

# Get logger for this module
logger = logging.getLogger(__name__)

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
        setup_path (Path): Path to setup directory
        parent_path (Path): Path to parent directory
        overwrite (bool): Whether to overwrite existing files
        show_figures (bool): Whether to show figures
        use_parent_dir (bool): Whether to use parent directory for job directories
        use_wt_msa (bool): Whether to use WT MSA
        wt_msa_path (Optional[Path]): Path to WT MSA
        mutations (List[str]): List of mutations
    """
    
    def __init__(
        self,
        sequence: str,
        jobname: str,
        copies: int = 1,
        msa_method: str = "mmseqs2",
        custom_a3m_path: Union[str, Path] = "",
        use_wt_msa: bool = False,
        wt_msa_path: Optional[Path] = None,
        mutations: Optional[List[str]] = None,
        pair_mode: str = "unpaired",
        cov: int = 0,
        id: int = 90,
        qid: int = 0,
        do_not_filter: bool = False,
        template_mode: str = "none",
        pdb: str = "",
        chain: str = "",
        rm_template_seq: bool = False,
        propagate_to_copies: bool = False,
        do_not_align: bool = False,
        setup_path: Optional[Union[str, Path]] = None,
        parent_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
        show_figures: bool = True,
        use_parent_dir: bool = False
    ):
        """Initialize PrepInputs."""
        self.logger = logging.getLogger(__name__)
        
        # Validate required parameters
        if not sequence:
            raise ValueError("sequence must not be empty")
            
        # Handle jobname - allow empty jobname, will be set in process_sequence
        self.jobname = str(jobname) if jobname else ""
        self.logger.info(f"Initializing PrepInputs with jobname: {self.jobname}")
        
        # Validate and convert paths
        if not setup_path:
            raise ValueError("setup_path must be provided")
        if not parent_path:
            raise ValueError("parent_path must be provided")
            
        # Convert paths to Path objects and ensure they exist
        self.setup_path = Path(setup_path)
        self.parent_path = Path(parent_path)
        
        if not self.setup_path.exists():
            raise FileNotFoundError(f"setup_path does not exist: {self.setup_path}")
        if not self.parent_path.exists():
            self.logger.info(f"Creating parent directory: {self.parent_path}")
            self.parent_path.mkdir(parents=True, exist_ok=True)
            
        # Handle custom_a3m_path
        self.custom_a3m_path = Path(custom_a3m_path) if custom_a3m_path else None
        
        self.logger.debug(f"Using setup_path: {self.setup_path} (type: {type(self.setup_path)})")
        self.logger.debug(f"Using parent_path: {self.parent_path} (type: {type(self.parent_path)})")
        self.logger.debug(f"Using custom_a3m_path: {self.custom_a3m_path} (type: {type(self.custom_a3m_path)})")
        
        # Store other parameters
        self.sequence = sequence
        self.copies = copies
        self.msa_method = msa_method
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
        self.overwrite = overwrite
        self.show_figures = show_figures
        self.use_parent_dir = use_parent_dir
        self.use_wt_msa = use_wt_msa
        self.wt_msa_path = Path(wt_msa_path) if wt_msa_path else None
        self.mutations = mutations or []
        
        # Initialize other attributes
        self.input_opts = {}
        self.msa = None
        self.deletion_matrix = None
        self.has_templates = False
        self.u_lengths = None
        self.u_sub_lengths = None
        self.u_cyclic = None
        self.Ls = None
        self.batches = None
        
        self.logger.debug(f"PrepInputs initialized with jobname: {self.jobname}")

    def process_sequence(self) -> None:
        """Process input sequence and set up job parameters."""
        self.logger.info("Processing sequence and setting up job parameters")
        
        # Process sequences
        sequences = self.sequence.split(":")
        self.u_sequences = predict.get_unique_sequences(sequences)
        self.u_cyclic = [x.startswith("(") for x in self.u_sequences]
        self.u_sub_lengths = [[len(y) for y in x.split("/")] for x in self.u_sequences]
        self.u_sequences = [
            x.replace("(", "").replace(")", "").replace("/", "")
            for x in self.u_sequences
        ]
        
        if len(sequences) > len(self.u_sequences):
            self.logger.warning("Use copies to define homooligomers")
            
        self.u_lengths = [len(x) for x in self.u_sequences]
        self.sub_seq = "".join(self.u_sequences)
        seq = self.sub_seq * self.copies

        # Generate jobname if not provided or append hash
        seq_hash = predict.get_hash(seq)[:5]
        if not self.jobname:
            self.jobname = f"job_{seq_hash}"
            self.logger.info(f"Generated jobname: {self.jobname}")
        else:
            # Format jobname to include hash if not already present
            if not self.jobname.endswith(seq_hash):
                # If jobname already has a hash, replace it
                if '_' in self.jobname and len(self.jobname.split('_')[-1]) == 5:
                    base_jobname = '_'.join(self.jobname.split('_')[:-1])
                    self.jobname = f"{base_jobname}_{seq_hash}"
                else:
                    self.jobname = f"{self.jobname}_{seq_hash}"
            self.logger.info(f"Using jobname with hash: {self.jobname}")
            
        # Create job directories after jobname is finalized
        self._create_directories()
        
        # Handle existing job
        self._handle_existing_job()
        
        self.logger.info(f"Sequence processing complete. Length={self.u_lengths} copies={self.copies}")

        # Set input options
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
        self.logger.debug(f"Input options set: {self.input_opts}")

    def filter_options(self) -> None:
        """Filter and validate input options."""
        self.logger.debug(f"Filtering options for jobname: {self.jobname}")
        self.sequence = self.sequence.upper()
        self.sequence = re.sub("[^A-Z:/()]", "", self.sequence)
        self.sequence = re.sub("\(", ":(", self.sequence)
        self.sequence = re.sub("\)", "):", self.sequence)
        self.sequence = re.sub(":+", ":", self.sequence)
        self.sequence = re.sub("/+", "/", self.sequence)
        self.sequence = re.sub("^[:/]+", "", self.sequence)
        self.sequence = re.sub("[:/]+$", "", self.sequence)
        self.jobname = re.sub(r"\W+", "", self.jobname)
        self.logger.debug(f"Filtered options: sequence={self.sequence}, jobname={self.jobname}")
    def _handle_existing_job(self) -> None:
        """Handle cases where job directory already exists."""
        self.logger.info("Checking for existing job directory")
        
        def check(folder: str) -> bool:
            """Check if a folder exists in the parent path."""
            if not self.parent_path:
                raise ValueError("parent_path not initialized")
            job_path = self.parent_path / folder
            return job_path.exists()

        if check(self.jobname):
            n = 0
            while check(f"{self.jobname}_{n}"):
                n += 1
                
            if self.overwrite:
                self.logger.warning(
                    f"{self.jobname} already exists. Using the same jobname. "
                    "If you want to run the job with a different jobname, set overwrite to False. "
                    "If you did not change other parameters your files will be overwritten."
                )
            else:
                new_jobname = f"{self.jobname}_{n}"
                self.logger.warning(f"{self.jobname} already exists. Changing jobname to {new_jobname}")
                self.jobname = new_jobname
                # Create directories for the new jobname
                self._create_directories()

    def get_msa(self) -> None:
        """Get Multiple Sequence Alignment."""
        # Create input directory based on use_parent_dir flag
        if self.use_parent_dir:
            input_path = self.parent_path / "in"
        else:
            input_path = self.parent_path / self.jobname / "in"
        
        input_path.mkdir(parents=True, exist_ok=True)
        
        self.Ls = [len(x) for x in self.u_sequences]
        
        # Check if we should use WT MSA
        if self.use_wt_msa and self.wt_msa_path and self.wt_msa_path.exists():
            self._handle_wt_msa(input_path)
            self.logger.info(f"WT MSA used: {self.wt_msa_path}")
        elif self.msa_method == "mmseqs2":
            self._handle_mmseqs2_msa(input_path)
            self.logger.info(f"MMseqs2 MSA used: {input_path / 'msa.a3m'}")
        elif self.msa_method == "single_sequence":
            self._handle_single_sequence_msa()
            self.logger.info(f"Single sequence MSA used: {input_path / 'msa.a3m'}")
        elif self.msa_method.startswith("custom_"):
            self._handle_custom_msa(input_path)
            self.logger.info(f"Custom MSA used: {input_path / 'msa.a3m'}")
        else:
            raise ValueError(f"Unknown MSA method: {self.msa_method}")

        if len(self.msa) > 1 and self.show_figures:
            predict.plot_msa(self.msa, self.Ls)
            plt.savefig(input_path / "msa_feats.png", dpi=200, bbox_inches="tight")
            if "ipykernel" in sys.modules:
                plt.show()
            else:
                plt.close()

    def _handle_mmseqs2_msa(self, input_path: Path) -> None:
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
            
            utils = ColabDesignUtils(self.setup_path)
            
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
        msa_file = self.parent_path / self.jobname / "in/msa.a3m"
        with open(msa_file, "w") as a3m:
            a3m.write(f">{self.jobname}\n{self.sub_seq}\n")
        self.msa, self.deletion_matrix = predict.parse_a3m(msa_file)

    def _handle_custom_msa(self, input_path: Path) -> None:
        """Handle custom MSA file processing.
        
        Args:
            input_path: Path to the input directory where MSA files should be saved
        """
        msa_format = self.msa_method.split("_")[1]
        self.logger.info(f"MSA mode: {self.msa_method}")
        
        if not self.custom_a3m_path:
            raise ValueError("custom_a3m_path must be provided for custom MSA mode")
            
        if not isinstance(self.custom_a3m_path, Path):
            self.custom_a3m_path = Path(self.custom_a3m_path)
            
        if not self.custom_a3m_path.exists():
            raise FileNotFoundError(f"MSA file not found: {self.custom_a3m_path}")
            
        # Create MSA file in input directory
        msa_file = input_path / f"msa.{msa_format}"
        self.logger.info(f"Creating MSA file at: {msa_file}")
        
        # For logging purposes log the value of use_wt_msa and mutations
        self.logger.info(f"use_wt_msa: {self.use_wt_msa}")
        self.logger.info(f"mutations: {self.mutations}")
        
        # Handle mutations if use_wt_msa is True
        if self.use_wt_msa and self.mutations:
            self.logger.info(f"Using WT MSA and applying mutations: {self.mutations}")
            modify_msa_first_sequence(
                msa_path=self.custom_a3m_path,
                mutations=self.mutations,
                output_path=msa_file
            )
            self.logger.info(f"Created modified MSA file at: {msa_file}")
        else:
            # Just copy the MSA file if no mutations or not using WT MSA
            shutil.copy2(self.custom_a3m_path, msa_file)
            self.logger.info(f"Copied MSA file to: {msa_file}")
        
        # TODO : Update this is very unsafe and not very easy to debug for the user if it fails
        if msa_format != "a3m":
            self.logger.info(f"Reformatting MSA from {msa_format} to a3m")

            if "hhsuite" not in os.environ["PATH"]:
                os.environ["PATH"] += f":{self.setup_path/'hhsuite/bin'}:{self.setup_path/'hhsuite/scripts'}"
            os.system(
                f"perl hhsuite/scripts/reformat.pl {msa_format} a3m {msa_file} {input_path/'msa.a3m'}"
            )
            
        self._filter_msa(input_path)

    def _filter_msa(self, input_path: Path) -> None:
        """Filter MSA based on parameters.
        
        Args:
            input_path: Path to the directory containing the MSA file
        """
        self.logger.info("Filtering MSA")
        
        # Ensure hhsuite is in PATH
        if "hhsuite" not in os.environ["PATH"]:
            os.environ["PATH"] += f":{self.setup_path/'hhsuite/bin'}:{self.setup_path/'hhsuite/scripts'}"
            
        # Check if hhfilter is available
        try:
            subprocess.run(["hhfilter", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise RuntimeError("hhfilter command not found. Make sure hhsuite is properly installed.")

        filt_file = input_path / "msa.filt.a3m"
        if not filt_file.exists():
            if self.do_not_filter:
                self.logger.warning("Not filtering MSA. Using 0 cov, 0 qid and 100 id")
                command = [
                    "hhfilter",
                    "-qid", "0",
                    "-id", "100",
                    "-cov", "0",
                    "-i", str(input_path/"msa.a3m"),
                    "-o", str(filt_file)
                ]
            else:
                self.logger.info(f"Filtering MSA with HHFilter. Using {self.cov} cov, {self.qid} qid and {self.id} id")
                command = [
                    "hhfilter",
                    "-qid", str(self.qid),
                    "-id", str(self.id),
                    "-cov", str(self.cov),
                    "-i", str(input_path/"msa.a3m"),
                    "-o", str(filt_file)
                ]
                
            self.logger.debug(f"Running command: {' '.join(command)}")
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                self.logger.debug("hhfilter output:")
                self.logger.debug(result.stdout)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error running hhfilter: {e}")
                self.logger.error(f"hhfilter stderr output:\n{e.stderr}")
                raise RuntimeError(f"hhfilter failed: {e.stderr}")

        # Parse filtered MSA
        self.msa, self.deletion_matrix = predict.parse_a3m(filt_file)
        self.logger.info(f"Filtered MSA has {len(self.msa)} sequences")

    def use_templates(self):
        """Process and set up templates if needed."""
        self.has_templates = self.template_mode in ["mmseqs2", "custom"]
        
        if self.has_templates:
            print("aligning template")
            template_msa = f"{self.parent_path}/{self.jobname}/in/msa.a3m"
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
                    output_a3m=f"{self.parent_path}/{self.jobname}/in/msa_tmp.a3m",
                )
                template_msa = f"{self.parent_path}/{self.jobname}/in/msa_tmp.a3m"
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
                for line in open(f"{self.parent_path}/{self.jobname}/in/msa/_env/pdb70.m8", "r"):
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
                        os.environ["PATH"] += f":{self.setup_path/'hhsuite/bin'}:{self.setup_path/'hhsuite/scripts'}"
                    
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
                    f"{self.parent_path}/{self.jobname}/in/template_feats.png",
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

    def _create_directories(self):
        """Create necessary directories for the job."""
        if self.use_parent_dir:
            # When running through CLI, use parent_path directly
            job_dir = self.parent_path
            self.logger.info(f"Using parent directory: {job_dir}")
        else:
            # Normal behavior: create job-specific subdirectory
            job_dir = self.parent_path / self.jobname
            self.logger.info(f"Creating job-specific subdirectory: {job_dir}")
            
        input_dir = job_dir / "in"
        output_dir = job_dir / "out"
        pdb_dir = output_dir / "pdbs"
        pkl_dir = output_dir / "pkl"
        
        # Create all required directories
        for directory in [job_dir, input_dir, output_dir, pdb_dir, pkl_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")

    def _handle_wt_msa(self, input_path: Path) -> None:
        """Handle WT MSA reuse and modification."""
        self.logger.info("Handling WT MSA")
        msa_file = input_path / "msa.a3m"
        
        # Copy WT MSA to input directory
        if not msa_file.exists():
            shutil.copy2(self.wt_msa_path, msa_file)
            logger.info(f"Copied WT MSA from {self.wt_msa_path} to {msa_file}")
        
        # Log that we have mutations if specified
        if self.mutations:
            self.logger.info(f"Mutations specified: {self.mutations}")
        else:
            self.logger.info("No mutations specified")
        
        # If mutations are specified, modify the MSA
        if self.mutations:
            try:
                modify_msa_first_sequence(
                    msa_path=msa_file,
                    mutations=self.mutations
                )
                logger.info(f"Modified MSA with mutations: {self.mutations}")
            except Exception as e:
                logger.error(f"Failed to modify MSA with mutations: {e}")
                raise
        
        # Parse the MSA
        self.msa, self.deletion_matrix = predict.parse_a3m(msa_file)
        logger.info(f"Using WT MSA with {len(self.msa)} sequences")

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

def modify_msa_first_sequence(
    msa_path: Path,
    mutations: List[str],
    output_path: Optional[Path] = None
) -> Path:
    """
    Modifies the first sequence of an MSA file with given mutations.
    Only used when use_wt_msa=True in config.
    
    Args:
        msa_path: Path to the original MSA file
        mutations: List of mutations in format ["A123B", ...]
        output_path: Optional path for modified MSA. If None, modifies in place.
    
    Returns:
        Path to the modified MSA file
    
    Raises:
        ValueError: If mutations are invalid or MSA file is malformed
        FileNotFoundError: If MSA file doesn't exist
    """
    try:
        # Validate input
        if not msa_path.exists():
            raise FileNotFoundError(f"MSA file not found: {msa_path}")
        
        # If output path is provided, copy MSA first
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(msa_path, output_path)
            working_path = output_path
        else:
            working_path = msa_path
        
        # Read MSA file
        with open(working_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            raise ValueError(f"Empty MSA file: {working_path}")
        
        # Get first sequence (after description line)
        if not lines[0].startswith('>'):
            raise ValueError(f"Invalid MSA format, missing description line: {working_path}")
        
        description_line = lines[0]
        sequence_line = lines[1]
        
        # Convert sequence to list for easier modification
        sequence = list(sequence_line.strip())
        
        # Apply mutations
        for mutation in mutations:
            if len(mutation) < 3:
                raise ValueError(f"Invalid mutation format: {mutation}")
            
            try:
                # Parse mutation (format: A123B)
                orig_aa = mutation[0]
                new_aa = mutation[-1]
                pos = int(mutation[1:-1]) - 1  # Convert to 0-based indexing
                
                # Validate position
                if pos < 0 or pos >= len(sequence):
                    raise ValueError(f"Position {pos+1} out of range for sequence")
                
                # Validate original amino acid
                if sequence[pos] != orig_aa:
                    raise ValueError(
                        f"Mismatch at position {pos+1}: expected {orig_aa}, found {sequence[pos]}"
                    )
                
                # Apply mutation
                sequence[pos] = new_aa
                logger.info(f"Applied mutation {mutation} at position {pos+1}")
                
            except ValueError as e:
                raise ValueError(f"Error processing mutation {mutation}: {str(e)}")
        
        # Write modified MSA
        with open(working_path, 'w') as f:
            f.write(description_line)  # Write original description
            f.write(''.join(sequence) + '\n')  # Write modified sequence
            # Write remaining lines unchanged
            f.writelines(lines[2:])
        
        logger.info(f"Successfully modified MSA with mutations: {mutations}")
        return working_path
        
    except Exception as e:
        logger.error(f"Error modifying MSA: {str(e)}")
        raise

def copy_msa_with_mutations(
    source_msa: Path,
    target_dir: Path,
    mutations: List[str]
) -> Path:
    """
    Copies an MSA file to a new location and applies mutations to the first sequence.
    
    Args:
        source_msa: Path to source MSA file
        target_dir: Directory to copy modified MSA to
        mutations: List of mutations to apply
    
    Returns:
        Path to the new modified MSA file
    """
    try:
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Create target path
        target_path = target_dir / "msa.a3m"
        
        # Copy and modify MSA
        return modify_msa_first_sequence(
            msa_path=source_msa,
            mutations=mutations,
            output_path=target_path
        )
        
    except Exception as e:
        logger.error(f"Error copying and modifying MSA: {str(e)}")
        raise 