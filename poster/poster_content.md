# Challenges in Interpreting Ultrafast Spectroscopy Data: A Case for Machine Learning in Materials Research

**Muhammed Arshad Thottappali**
Institute of Macromolecular Chemistry, Czech Academy of Sciences

ML4MatSci PhD Summer School — Sarajevo, June 10–12, 2026

---

## 1. THE PROBLEM: Why is TA Data Analysis Hard?

Transient Absorption (TA) spectroscopy probes photo-induced processes on femtosecond–nanosecond timescales. The raw data is a 2D matrix: ΔA(wavelength, time).

### Challenges:
- **Model ambiguity**: Multiple kinetic models can fit the same data equally well
- **Overlapping spectra**: Excited-state species have broad, overlapping spectral features
- **High dimensionality**: Typical dataset = 300 wavelengths × 250 time points = 75,000 data points
- **Expert bottleneck**: Traditional Global/Target Analysis requires a human expert to:
  1. Guess the number of species
  2. Guess the kinetic topology (sequential? branching? parallel?)
  3. Fit the ODE system — iterating until convergence
  4. This takes **hours per sample** and is subjective

### Our system:
- Two isotopic variants of a molecular material (H and D substituted)
- Sequential relaxation: A → B → C → Ground State
- Question: Can ML automate the model selection step?

---

## 2. THE APPROACH: DeepSKAN (Deep Spectroscopy Kinetic Analysis Network)

### Core Idea:
Treat the 2D TA map as a **grayscale image** and train a CNN to classify the underlying kinetic topology.

### Architecture:
- 7-block ResNet with asymmetric kernels (time vs wavelength)
- BatchNorm + Dropout for regularization
- Input: 256×64 normalized TA map
- Output: probability over 21 kinetic model classes (4 species)
- Trained entirely on **synthetic physics-simulated data** (no labeled experimental data needed)

### Training Pipeline:
```
Random kinetic model → ODE solver → synthetic TA map → CNN learns topology
                ↑                                            ↓
    103 or 21 viable topologies              Classification (softmax)
```

### Training Results:
- 50,000 synthetic samples, 40 epochs, T4 GPU
- **82% training accuracy, 72% validation accuracy** on 21 classes
- Loss converged from log(21)=3.04 (random) to 0.47 (trained)

---

## 3. APPLYING DeepSKAN TO REAL DATA

### CNN Predictions:

| Sample | Top-1 Class | Confidence | Predicted Topology |
|--------|-------------|------------|-------------------|
| CBK-D  | Class 19    | 40%        | A→B, A→C, B→C, B→D (branching) |
| CBK-H  | Class 6     | 59%        | A→B, B→C (sequential, 2 steps) |

### What happened:
- CBK-H: CNN predicted a sequential model (close to correct!)
- CBK-D: CNN predicted a branching model (wrong — we know it's sequential)
- The CNN confidence was **low** (40-59%) — honestly reporting uncertainty

### Automated GA/TA with CNN-predicted model:
- Decay times: τ₁ = 0.1 ps, τ₂ = 0.0 ps ← **Nonsensical**
- The CNN-preprocessed data (normalized to 256×64) lost physical scale information
- The optimizer collapsed to boundary values

---

## 4. DIRECT GLOBAL ANALYSIS (Without DeepSKAN)

When we **impose the known sequential model** and fit directly to raw data:

### Method:
- SVD extracts dominant kinetic components
- 3-exponential fit with Gaussian IRF convolution
- Global optimization (differential evolution) + LM polishing
- Full 2D decomposition → EADS + concentration profiles

### Results:

| Component | CBK-D (ps) | CBK-H (ps) | D/H Ratio | Assignment |
|-----------|-----------|-----------|-----------|------------|
| τ₁ (A→B)  | 9.4       | 13.8      | 0.69      | Vibrational cooling |
| τ₂ (B→C)  | 91.8      | 173.9     | 0.53      | Structural relaxation |
| τ₃ (C→GS) | >6 ns     | >6 ns     | ~2.5      | Radiative/non-rad decay |

- IRF: ~70 fs (consistent with our laser system)
- Fit residuals: ~10⁻⁴ (excellent)
- Clear isotope effect on τ₃ (normal KIE)
- Inverse isotope effect on τ₁, τ₂ (suggests tunneling or mode coupling)

---

## 5. WHY DID DeepSKAN FAIL? (Lessons Learned)

### The Sim-to-Real Gap:

| Factor | Synthetic Training Data | Real Experimental Data |
|--------|------------------------|----------------------|
| Noise | Random Gaussian | Structured (scatter, drift, coherent artifact) |
| Spectra | Random Gaussian sums | Physically structured bands |
| Time range | Fixed 0.1–10,000 ps | Includes negative time, variable range |
| Normalization | Simple max-norm | Needs careful handling of negative signals |
| IRF | Symmetric Gaussian | Can be asymmetric |

### Key Failure Points:
1. **Preprocessing destroyed information**: Interpolating to 256×64 and normalizing removed the amplitude and time-scale information the fitter needs
2. **Training data too simple**: Random Gaussian SADS don't represent real molecular spectra
3. **Low confidence = honest uncertainty**: The 40% confidence correctly flagged that the model was unsure — this is a feature, not a bug
4. **21 classes is ambiguous**: Several topologies produce similar TA maps — the classification problem is inherently hard

### What Would Fix It:
- Train on larger datasets with realistic noise
- Use real spectra (not random Gaussians) as SADS templates
- Separate CNN classification from quantitative fitting
- Hybrid approach: CNN for model suggestion → expert validates → classical fit

---

## 6. CONCLUSION: Where ML Helps and Where It Doesn't

### ML/DL IS useful for:
✓ Automated screening of many samples (e.g., combinatorial libraries)
✓ First-pass model suggestion when no prior knowledge exists
✓ Identifying unusual kinetics that don't match expected patterns
✓ Reducing human bias in model selection

### ML/DL is NOT (yet) a replacement for:
✗ Quantitative parameter extraction (decay times, amplitudes)
✗ Systems where the researcher already knows the model
✗ Low-confidence predictions without expert validation
✗ Data where the sim-to-real gap is large

### The Ideal Workflow:
```
Real TA data → CNN screens topology (fast, automated)
                    ↓
           Expert validates prediction
                    ↓
    Classical Global Analysis with constrained model
                    ↓
        EADS + concentrations + decay times (publication-quality)
```

---

## 7. TECHNICAL DETAILS

### Code & Reproducibility:
- GitHub: [link to repo]
- Framework: PyTorch 2.x
- Training: T4 GPU, ~3 hours
- Dataset: 50k synthetic samples, physics-based simulation

### Key References:
- Kollenz et al., J. Phys. Chem. B (2020) — original DeepSKAN
- van Stokkum et al., BBA (2004) — Global/Target Analysis theory
- Selvaraju et al., ICCV (2017) — Grad-CAM

---

---

## 7. FUTURE DIRECTIONS: From Data-Driven to Knowledge-Driven AI

### The Fundamental Limitation of DeepSKAN

DeepSKAN treats kinetic model identification as a **pure pattern recognition** problem — feed a 2D image to a CNN, get a class label. This ignores the most powerful information available to a real researcher: **domain knowledge about the molecule**.

A spectroscopist looking at a porphyrin NAnEn complex doesn't search through 21 random topologies. They think: *"This is a porphyrin — I expect vibrational cooling (ps), intersystem crossing (tens of ps), and a long-lived triplet."* The model choice is informed by **chemistry**, not by pixel patterns.

### Agentic AI: The Right Paradigm

Instead of training a CNN on synthetic data, a more effective approach is an **LLM-based agentic system** that:

1. **Reads** the molecular identity (structure, functional groups, known photophysics)
2. **Searches** published literature for similar molecules and their kinetic schemes
3. **Proposes** a kinetic model with chemical assignments (not abstract class numbers)
4. **Fits** the data with the proposed model using classical Global Analysis
5. **Validates** against literature values — flags if results are inconsistent

### Comparison: CNN vs Agent

| Aspect | CNN (DeepSKAN) | Agentic AI |
|--------|---------------|------------|
| Model selection | Learned from synthetic patterns | Reasoned from chemical knowledge |
| Training data | 50,000 simulated samples | Zero — uses published literature |
| Sim-to-real gap | Major failure point | Eliminated — works directly on real data |
| Output | Class number (e.g., "Class 6") | Chemical assignment (e.g., "S₁ → T₁ via ISC, τ~100ps") |
| Interpretability | Black box | Explainable (cites sources) |
| Generalization | Only works for the specific species count trained on | Works for any molecule |
| Time to deploy | Hours of GPU training | Minutes (prompt engineering) |

### What This Would Look Like in Practice

```
INPUT:  TA data (CSV) + molecular structure (SMILES or name)

AGENT WORKFLOW:
  ┌─────────────────────────────────────────────────────┐
  │ Step 1: Identify molecule type                       │
  │   "NAnEn porphyrin complex, D-substituted"           │
  ├─────────────────────────────────────────────────────┤
  │ Step 2: Literature search (automated)                │
  │   → "Porphyrin excited-state dynamics typically      │
  │      show: S₁ vib. cooling (1-15 ps), ISC to T₁     │
  │      (50-200 ps), T₁ decay (ns-μs)"                 │
  │   → Cites: Smith et al. 2022, Kumar et al. 2020     │
  ├─────────────────────────────────────────────────────┤
  │ Step 3: Propose model                                │
  │   "Sequential A→B→C→GS (3 lifetimes)"               │
  │   A = S₁(hot), B = S₁(relaxed), C = T₁             │
  ├─────────────────────────────────────────────────────┤
  │ Step 4: Fit data                                     │
  │   Global Analysis → τ₁=9.4ps, τ₂=92ps, τ₃>6ns      │
  ├─────────────────────────────────────────────────────┤
  │ Step 5: Validate                                     │
  │   "τ₂=92ps consistent with ISC in metalloporphyrins │
  │    (literature range: 50-200 ps). ✓"                 │
  └─────────────────────────────────────────────────────┘

OUTPUT: Decay times + EADS + chemical assignments + confidence
        + literature references for validation
```

### Why This Matters for Materials Science

- **Scales** — can analyze hundreds of samples with molecular context
- **Democratizes** — non-experts get expert-level model selection
- **Reproducible** — the agent's reasoning chain is fully traceable
- **Current** — always uses the latest published knowledge
- **No training required** — works out-of-the-box on any new molecule

### Relevance to ML4MatSci Focus Areas

This directly connects to multiple school topics:
- **Agentic AI & Multi-Agent Systems** — the core approach
- **Large Language Models (LLMs)** — the reasoning engine
- **Explainable AI (XAI)** — the agent explains its model choice with citations
- **Foundation models** — LLMs as foundation models for scientific reasoning

### Implementation Path

A prototype would use:
- **LLM** (Claude, GPT-4) as the reasoning agent
- **Tools** (MCP or function calling): data loader, GA fitter, paper search API
- **Structured prompts**: molecular identity → relevant kinetic schemes → fitting constraints
- **Validation loop**: compare extracted τ values to literature ranges

This could be built in 1-2 weeks as a proof-of-concept and demonstrated interactively.

---

## 8. ALTERNATIVE ML APPROACHES (Beyond CNN Classification)

### Physics-Informed Neural Networks (PINNs)

Instead of classifying the model topology, embed the kinetic ODEs into the neural network's loss function:

$$ L = L_{\text{data}} + \alpha \cdot L_{\text{physics}} $$

Where:
- $L_{\text{data}}$ = mismatch between reconstructed and measured TA map
- $L_{\text{physics}}$ = violation of the kinetic rate equations (dC/dt = K·C)

**Advantages over CNN:**
- Fits a single dataset — no training data needed
- Rate constants are trainable parameters — come out directly in ps
- Physics constraint prevents nonsensical results (τ=0)
- Handles degenerate cases (equal lifetimes) gracefully via autograd

### Bayesian Global Analysis

Add uncertainty quantification to the fitting:
- MCMC sampling over rate constants → full posterior distributions
- Report: "τ₂ = 92 ± 15 ps" instead of just "92 ps"
- Can answer: "Is the D/H difference in τ₂ statistically significant?"

### Non-negative Matrix Factorization (NMF) / MCR-ALS

Unsupervised decomposition of the 2D TA matrix:
- No model assumption needed
- Extracts components with physical constraints (non-negative concentrations)
- Good for exploratory analysis when the kinetic scheme is unknown

---

## FIGURES (for poster layout)

1. **Raw TA heatmaps** (both samples) — `results/sequential_full/` or `results/CBK-D/_raw.png`
2. **Training curve** — `results/training_curve.png`
3. **CNN confidence bar chart** — `results/CBK-D/_confidence.png` and `results/CBK-H/_confidence.png`
4. **EADS** — `results/sequential_full/eads.png`
5. **Concentration profiles** — `results/sequential_full/concentrations.png`
6. **EADS D vs H comparison** — `results/sequential_full/eads_comparison.png`
7. **Isotope effect bar chart** — `results/sequential/isotope_effect.png`
8. **Kinetic scheme diagram** (A→B→C→GS with τ values)
9. **Pipeline overview with failure point** — `poster/pipeline_overview.png`
10. **Comparison table** (CNN vs Direct) — `poster/comparison_table.png`
11. **Agentic workflow diagram** (new — for Future Directions)
