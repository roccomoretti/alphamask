# AlphaMask

AlphaMask is a tool for analyzing protein sequences through various masking strategies.

## Masking Strategies

### 1. Iterative Masking
Systematically explores sequence positions by masking each position independently, with optional mutation analysis.

```mermaid
graph TD
    A[WT Sequence] --> B[Generate MSA]
    B --> C[Single Position Masking]
    C --> D[Mask Position 1]
    C --> E[Mask Position 2]
    C --> F[Mask Position ...]
    C --> G[Mask Position N]
    
    A --> H[Apply Mutations]
    H --> I[Mutation Set 1<br/>T150A]
    H --> J[Mutation Set 2<br/>L157R]
    H --> K[Mutation Set 3<br/>T150A+L157R]
    
    I --> L[Use WT MSA + Mutate Target]
    J --> L
    K --> L
    
    L --> M[Single Position Masking<br/>with Mutations]
    M --> N[Mask Position 1]
    M --> O[Mask Position 2]
    M --> P[Mask Position ...]
    M --> Q[Mask Position N]
```

### 2. A Priori Masking
Focused experiments on known positions of interest with controlled conditions.

```mermaid
graph TD
    subgraph "A Priori Experiment: Known_position_F21A"
        A[WT Sequence] --> B[Generate MSA]
        
        subgraph "Control Conditions"
            B --> C1[No Mask, No Mutation<br/>Control]
            B --> C2[Mask Position 21<br/>No Mutation]
            B --> C3[No Mask<br/>Mutate F21A]
            B --> C4[Mask Position 21<br/>Mutate F21A]
        end
    end

    subgraph "A Priori Experiment: Double_mutation_study"
        A2[WT Sequence] --> B2[Generate MSA]
        
        subgraph "Control Conditions"
            B2 --> D1[No Mask, No Mutation<br/>Control]
            B2 --> D2[Mask Positions 21,24<br/>No Mutation]
            B2 --> D3[No Mask<br/>Mutate F21A+Y24A]
            B2 --> D4[Mask Positions 21,24<br/>Mutate F21A+Y24A]
        end
    end
```

### 3. Frustra Masking
Analysis-driven approach using protein frustration patterns to identify positions of interest.

```mermaid
graph TD
    A[WT Sequence] --> B[Generate MSA]
    B --> C[Run Frustra Analysis]
    C --> D[Calculate Frustration Scores]
    D --> E[Sort Positions by Score]
    E --> F[Select Top N Positions]
    
    subgraph "Masking Experiments"
        F --> G1[No Mask, No Mutation<br/>Control]
        F --> G2[Mask Top Positions<br/>No Mutation]
        
        G2 --> H1[Position 1 from Top N]
        G2 --> H2[Position 2 from Top N]
        G2 --> H3[Position ... from Top N]
        G2 --> H4[Position N from Top N]
    end
    
    subgraph "Analysis"
        H1 --> I[Compare with Control]
        H2 --> I
        H3 --> I
        H4 --> I
        I --> J[Identify Critical Positions]
    end
```

## Installation

### 1. Environment Setup

First, ensure you're on a compute node with GPU access:
```bash
# Request an interactive GPU session (adjust parameters according to your cluster)
srun --job-name "alphamask_setup" \
     --gres=gpu:1 \  # Specify GPU requirements for your cluster
     --time 24:00:00 \
     --partition=YOUR_GPU_PARTITION \  # e.g., gpus, gpu, accelerated, etc.
     --pty bash
```

### 2. Load Required Modules
```bash
# Load CUDA module (version may vary by cluster)
module load cuda  # e.g., cuda/12.6, cuda/11.8, etc.

# Load any additional required modules
module load gcc   # If needed
module load python  # If needed
```

### 3. Create Conda Environment
```bash
# Using micromamba (recommended)
micromamba create -f environment.yml

# Or using conda
conda env create -f environment.yml

# Activate the environment
micromamba activate alphamask  # or conda activate alphamask
```

### 4. Verify Installation
```bash
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Check GPU visibility
nvidia-smi
```

### 5. Setup Experiment Directory

```bash
# Remove existing experiment folder if needed
rm -rf /path/to/workspace/my_experiments/ 

# Setup experiment folder
python -m alphamask setup --path /path/to/workspace/my_experiments 
```

### 6. Running Experiments

```bash
# Basic experiment run
python -m alphamask run \
    --container /path/to/container/vsc-frustra_masking.sif \
    --script /path/to/alphamask/predict.py \
    --schema /path/to/workspace/my_experiments/schema/schema_validation.json \
    --config /path/to/workspace/my_experiments/config/test.yaml \
    --partition YOUR_GPU_PARTITION \
    --gpu-type YOUR_GPU_TYPE

# Check available partitions and GPU types on your cluster
sinfo -o "%10P %10G %10O %10l %10c"  # For SLURM-based clusters
```

### 7. Extracting Results

```bash
# Extract all PDBs
alphamask extract-pdbs --config config.yaml

# Extract only best predictions
alphamask extract-pdbs --config config.yaml --best-only

# Extract specific models/seeds/recycles
alphamask extract-pdbs --config config.yaml \
    --models model_1 model_2 \
    --seeds 1 2 \
    --recycles 0 1

# Extract for specific proteins
alphamask extract-pdbs --config config.yaml \
    --proteins protein1 protein2 \
    --best-only
```

### Common Cluster-Specific Adjustments

1. **GPU Selection**: Different clusters use different GPU naming conventions:
   - Some use specific models (e.g., `a100`, `v100`, `quadro_rtx_8000`)
   - Others use generic names (e.g., `gpu:1`, `gpu:k80:1`)

2. **Partition Names**: Common variations include:
   - `gpu`, `gpus`, `accelerated`
   - `cuda`, `tesla`, `nvidia`
   - Check your cluster documentation for specific names

3. **Module Names**: Module naming conventions vary:
   - CUDA: `cuda/12.6`, `cuda/11.8`, `nvidia/cuda-12.6`
   - Python: `python/3.10`, `python3`, `anaconda3`

Always consult your cluster's documentation or system administrators for specific configuration details.
