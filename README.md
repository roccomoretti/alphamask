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


# Remove existing experiment folder
rm -rf /work/YOUR_USERNAME_WORKSPACE_FOLDER/my_experiments/ 

# Setup experiment folder
python -m alphamask setup --path /work/YOUR_USERNAME_WORKSPACE_FOLDER/my_experiments 

# Run experiment
python -m alphamask run --container ~/containers/vsc-frustra_masking.sif \
--script /home/sc.uni-leipzig.de/YOUR_USERNAME_WORKSPACE_FOLDER/github/alphamask/predict.py \
--schema /work/YOUR_USERNAME_WORKSPACE_FOLDER/my_experiments/schema/schema_validation.json \
--config /work/YOUR_USERNAME_WORKSPACE_FOLDER/my_experiments/config/test.yaml \
--partition paula --gpu-type a30

# Make sure you select the correct partition and gpu type
# You can find the available partitions and gpu types with the following command:
sinfo -o "%10P %10G %10O %10l %10c"

