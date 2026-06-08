# DeepSKAN v1 — Complete Deep Explanation
## From Physics to Pixels to Predictions

**Purpose of this document:** Explain EVERYTHING about DeepSKAN v1 so you can answer any question at the poster session — from "what is a convolution?" to "why asymmetric kernels?" to "how does backpropagation work?"

---

## Table of Contents

1. [The Big Picture — What Are We Doing?](#1-the-big-picture)
2. [The Physics — Kinetic Models and TA Data](#2-the-physics)
3. [The Key Insight — TA as an Image](#3-the-key-insight)
4. [Neural Networks 101 — Building Blocks](#4-neural-networks-101)
5. [Convolutional Neural Networks — Why They Work for Images](#5-cnns)
6. [The DeepSKAN Architecture — Block by Block](#6-the-architecture)
7. [Residual Connections — Why They Matter](#7-residual-connections)
8. [BatchNorm and Dropout — Training Stability](#8-batchnorm-and-dropout)
9. [The Training Process — How the CNN Learns](#9-training)
10. [Synthetic Data Generation — The Physics Engine](#10-synthetic-data)
11. [Loss Function and Optimization — Finding the Best Weights](#11-loss-and-optimization)
12. [Inference — From Data to Prediction](#12-inference)
13. [What Can Go Wrong — Failure Modes](#13-failure-modes)
14. [Our Specific Results — CBK-H and CBK-D](#14-our-results)
15. [Quick Reference — Numbers to Remember](#15-quick-reference)

---

## 1. The Big Picture — What Are We Doing? {#1-the-big-picture}

### The Problem
You measure a transient absorption (TA) spectrum. It's a 2D matrix:
- **Rows** = wavelengths (430–800 nm, 313 points)
- **Columns** = time delays (−2 to 6000 ps, 258 points)
- **Values** = ΔA (change in absorbance, typically ±0.001)

The question: **What is the kinetic model?** Is it:
- A → B → C → GS (sequential, 3 steps)?
- A → B, A → C (parallel/branching)?
- A → B → C, with B also going directly to GS (branching + sequential)?

Traditionally, a human expert looks at the data, guesses a model, fits it, checks if the fit is good, tries another model if not — repeating for hours.

### The DeepSKAN Solution
Instead of a human, we train a **computer** to recognize which kinetic model produced a given TA map — just by looking at the 2D pattern of colors (positive/negative signals evolving over time).

This is essentially **image classification** — the same technology that recognizes cats vs dogs in photos, but applied to spectroscopy data.

### The Pipeline
```
Real TA data (313×258) 
    → preprocess (interpolate to 256×64, normalize)
    → CNN reads it like an image
    → outputs probabilities for 21 possible kinetic models
    → highest probability = predicted model
    → then use classical fitting (GA/TA) with that model
```

---

## 2. The Physics — Kinetic Models and TA Data {#2-the-physics}

### What Generates TA Data?

After a laser pulse excites the sample, the molecule goes through a sequence of excited states. At each moment in time, the observed signal is:

$$\Delta A(\lambda, t) = \sum_{i} C_i(t) \cdot \varepsilon_i(\lambda)$$

Where:
- $C_i(t)$ = concentration (population) of species $i$ at time $t$
- $\varepsilon_i(\lambda)$ = the difference spectrum of species $i$ (its spectral fingerprint)

The concentrations evolve according to a **rate equation** (coupled ODEs):

$$\frac{d\mathbf{C}}{dt} = \mathbf{K} \cdot \mathbf{C}$$

Where **K** is the rate matrix. The structure of K (which entries are zero vs non-zero) defines the **kinetic topology** — and that's what DeepSKAN classifies.

### The 21 Classes (for 4 species)

With 4 species (A, B, C, and Ground State), there are theoretically many possible connection patterns. After removing:
- Physically impossible models (orphan states that never get populated)
- Models where more than 2 species feed into one state (unrealistic)
- Isomorphic duplicates (same topology, just relabelled)

We're left with **21 unique, physically meaningful kinetic topologies**.

Examples:
- **Class 6:** A→B→C (simple sequential, 2 transfers)
- **Class 7:** A→B→C→GS (full sequential, 3 transfers)
- **Class 19:** A→B, A→C, B→C, B→D (complex branching)

Each class produces a **different-looking 2D TA map** because the concentrations evolve differently.

### Why This Is Hard for a Human

Some topologies produce very similar-looking TA maps. For example:
- A→B→C with τ₁=5ps, τ₂=100ps
- A→B, A→C with τ₁=5ps, τ₂=100ps

These can look nearly identical in the raw data, especially with noise. A human might need to try both models and compare fit quality — but a CNN can potentially detect subtle pattern differences that distinguish them.

---

## 3. The Key Insight — TA as an Image {#3-the-key-insight}

### Why Treat It as an Image?

A TA map is literally a 2D grid of numbers — just like a grayscale photograph. The "brightness" at each pixel corresponds to the signal intensity.

In a photo:
- Patterns of pixels → "this is a cat"

In a TA map:
- Patterns of positive/negative signals evolving over time → "this is Class 6"

The same mathematics that lets a computer recognize faces can recognize kinetic topologies.

### The Preprocessing

Real TA data has variable size (different spectrometers have different pixel counts and time steps). We need to standardize it to a fixed size:

1. **Interpolate time axis** to 256 points (log-spaced from 0.1 to 10,000 ps)
   - Log-spacing is critical because dynamics span many orders of magnitude
   - Early dynamics (1 ps) need just as much resolution as late dynamics (1000 ps)

2. **Interpolate wavelength axis** to 64 points (linearly spaced)

3. **Normalize** to [-1, 1] by dividing by the maximum absolute value
   - This makes the CNN independent of signal strength (only shape matters)

Result: every TA map becomes a **256×64 grayscale image** with values between -1 and +1.

### What Information Is Preserved vs Lost

**Preserved:**
- Relative shape of spectra at different times
- Relative timing of rises and decays
- Which wavelength regions are positive vs negative
- Number of distinct spectral features

**Lost:**
- Absolute signal amplitude (normalized away)
- Exact time scale in picoseconds (mapped to pixel indices)
- Exact wavelength positions (mapped to pixel indices)

This is why the CNN can classify topology but cannot directly give you decay times — the absolute time information is gone.

---

## 4. Neural Networks 101 — Building Blocks {#4-neural-networks-101}

### What Is a Neural Network?

A neural network is a mathematical function that:
1. Takes an **input** (our 256×64 image)
2. Applies many **layers** of simple operations (multiply, add, threshold)
3. Produces an **output** (21 class probabilities)

The "magic" is that the multiplication factors (**weights**) are learned from data, not programmed by hand.

### A Single Neuron

The simplest unit:
```
output = ReLU(w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + bias)
```

Where:
- `x₁...xₙ` = inputs
- `w₁...wₙ` = weights (learned parameters)
- `bias` = offset (also learned)
- `ReLU` = activation function: `max(0, x)` — clips negative values to zero

A single neuron draws a straight line through the data. Many neurons together can approximate any function.

### Layers

Neurons are organized into **layers**:
- **Input layer:** the raw data (256×64 = 16,384 numbers)
- **Hidden layers:** intermediate computations (where the "thinking" happens)
- **Output layer:** 21 numbers (one per class)

### What Is "Deep" Learning?

"Deep" = many layers stacked. DeepSKAN has:
- 7 convolutional blocks (each with multiple layers inside)
- 5 fully connected layers
- Total: ~30+ layers deep

Why deep? Each layer learns increasingly abstract features:
- Layer 1: edges, gradients (simple patterns)
- Layer 3: combinations of edges → shapes (rise times, decay curves)
- Layer 5: combinations of shapes → signatures (specific kinetic profiles)
- Layer 7: global patterns → "this is Class 6"

---

## 5. Convolutional Neural Networks — Why They Work for Images {#5-cnns}

### The Convolution Operation

Instead of connecting every pixel to every neuron (which would have billions of parameters), a CNN uses a **small sliding window** (called a kernel or filter) that moves across the image:

```
Kernel (3×3):          Image patch:          Output:
[1  0  -1]            [0.2  0.3  0.4]       sum of element-wise
[1  0  -1]     ×      [0.5  0.6  0.7]   =   products = single
[1  0  -1]            [0.1  0.2  0.3]       number
```

This kernel detects **vertical edges** (because it gives positive output where left side is bright and right side is dark).

### Why Convolutions for TA Data?

Three key properties:

1. **Translation invariance:** A decay pattern at 500 nm is recognized the same way as at 700 nm. The CNN doesn't need to memorize separate patterns for each wavelength — one kernel handles all positions.

2. **Local connectivity:** A kernel size of (9,3) means each output pixel depends on 9 time points and 3 wavelength points. This is physically meaningful — you're looking at "how does the signal change over a small time window and narrow spectral range?"

3. **Hierarchical features:** Early layers detect simple patterns (is the signal rising or falling?), later layers combine them into complex patterns (is this a sequential rise-decay-rise pattern?).

### Feature Maps (Channels)

One kernel produces one **feature map** — a new 2D image where each pixel says "how strongly does this pattern appear here?" Multiple kernels (e.g., 64) each detect different patterns, giving 64 feature maps.

In our architecture:
- Input: 1 channel (the TA map)
- After block 1: 64 channels (64 different patterns detected)
- After block 2: 128 channels
- After block 5: 256 channels (very abstract features)

### MaxPooling

After convolutions, we **downsample** by taking the maximum value in each 2×2 region:
```
[1.0  0.3]
[0.7  0.5]  → max = 1.0
```

Why? 
- Reduces computation (smaller maps = faster)
- Makes the network slightly position-invariant ("the feature is somewhere in this 2×2 area")
- Progressively reduces spatial dimensions: 256×64 → 128×32 → 64×16 → ...

---

## 6. The DeepSKAN Architecture — Block by Block {#6-the-architecture}

### Full Architecture Summary

```
Input: (1, 256, 64) — 1 channel, 256 time points, 64 wavelengths

GaussianNoise (σ=0.01) — training-time augmentation

ResBlock 1: 1→64 channels,   kernel (15,1), MaxPool (2,1)   → (64, 128, 64)
ResBlock 2: 64→128 channels, kernel (11,1), MaxPool (2,1)   → (128, 64, 64)
ResBlock 3: 128→128 channels,kernel (9,3),  MaxPool (2,2)   → (128, 32, 32)
ResBlock 4: 128→128 channels,kernel (9,3),  MaxPool (2,2)   → (128, 16, 16)
ResBlock 5: 128→256 channels,kernel (5,5),  MaxPool (2,2)   → (256, 8, 8)
ResBlock 6: 256→256 channels,kernel (3,3),  MaxPool (2,2)   → (256, 4, 4)
ResBlock 7: 256→128 channels,kernel (3,3)                    → (128, 4, 4)

AdaptiveAvgPool2d(2,4) → (128, 2, 4) = 1024 features

FC1: 1024 → 1024 + ReLU + Dropout
FC2: 1024 → 512 + ReLU + Dropout
FC3: 512 → 512 + ReLU + Dropout
FC4: 512 → 256 + ReLU + Dropout
FC5: 256 → 21 (output logits, one per class)

Softmax → 21 probabilities summing to 1.0
```

### Why Asymmetric Kernels? (The Most Important Design Choice)

**Blocks 1-2 use (15,1) and (11,1) kernels — tall and narrow.**

This means: look at **15 consecutive time points** but only **1 wavelength** at a time.

**Physical motivation:** In the time dimension, dynamics span many orders of magnitude and adjacent time points are highly correlated (smooth exponential curves). A tall kernel captures the *shape* of the decay over a significant time range.

In the wavelength dimension, adjacent wavelengths are also correlated, but the spectral features are narrower (a Q-band bleach might only span 20-30 nm). So we don't need to look at many wavelengths simultaneously in the early layers.

**Blocks 3-4 use (9,3) — starting to mix.**

Now the network begins to look at how the spectrum *changes* over both time and wavelength simultaneously. This detects things like "stimulated emission redshift" or "isosbestic points."

**Blocks 5-7 use (5,5) and (3,3) — fully 2D.**

By now the feature maps are small (8×8, 4×4) and contain abstract representations. Square kernels look at the global structure — "what is the overall pattern of rises and decays across the entire map?"

### Why This Progression Works

It mirrors how a human spectroscopist analyzes TA data:
1. First look along time axis: "how many distinct decay components are there?"
2. Then look at spectral-temporal correlations: "do the spectra shift over time?"
3. Finally, the global picture: "is this consistent with a sequential or branching model?"

### The GaussianNoise Layer

During training, small random noise (σ=0.01) is added to the input. This is **data augmentation** — it makes the model robust to slight variations in noise level between different measurements.

During inference (prediction on real data), this layer is turned off.

### The Final FC Layers

After the convolutional blocks extract features (1024 numbers summarizing the TA map), fully connected (FC) layers act as the **classifier**:

- 1024 abstract features → "is feature X present AND feature Y present? → probably Class 6"
- This is essentially a vote-counting system — each neuron in the FC layers combines evidence from different features to support or reject each class.

---

## 7. Residual Connections — Why They Matter {#7-residual-connections}

### The Problem with Deep Networks

In theory, deeper = more powerful. In practice, very deep networks suffer from **vanishing gradients** — the training signal (how to improve) gets weaker as it passes backward through many layers, until the early layers barely learn at all.

### The Solution: Skip Connections

A residual block does:
```
output = F(input) + input
```

Instead of:
```
output = F(input)
```

The `+ input` is the **skip connection** (also called residual connection or shortcut).

### Why This Helps

If the optimal transformation is close to identity (doing nothing), the network only needs to learn `F(x) ≈ 0`, which is easy. Without skip connections, learning identity requires `F(x) ≈ x`, which is hard.

More importantly: during training, gradients can flow through the skip connection directly, bypassing the transformation layers. This means early layers always get a strong training signal, even in a 30-layer network.

### In DeepSKAN

Each ResBlock has 4 convolutions inside, with a skip connection from input to output:
```
Input → Conv → BN → ReLU → Conv → BN → ReLU → Conv → BN → ReLU → Conv → BN → (+Input) → ReLU → Output
        |_______________________________________________________________|
                        skip connection (1×1 conv if channels change)
```

If the number of channels changes (e.g., 64→128), the skip connection uses a 1×1 convolution to match dimensions.

---

## 8. BatchNorm and Dropout — Training Stability {#8-batchnorm-and-dropout}

### Batch Normalization (BatchNorm)

**What it does:** After each convolution, normalize the output to have mean=0, std=1 across the batch.

**Why:** Without normalization, as data flows through many layers, the values can drift (get very large or very small). This makes training unstable — the optimizer overshoots because the scale of values keeps changing between updates.

BatchNorm fixes this by keeping all intermediate values in a stable range. Training converges ~5-10x faster.

**In our code:**
```python
nn.Conv2d(64, 128, kernel_size=(11,1), bias=False)  # bias=False because BN has its own bias
nn.BatchNorm2d(128)  # normalizes the 128 feature maps
nn.ReLU()
```

### Dropout

**What it does:** During training, randomly sets some neurons to zero (with probability p=0.3). During inference, all neurons are active.

**Why:** Prevents **overfitting** — without dropout, the network can memorize the training data (especially problematic when training on synthetic data that doesn't perfectly represent reality).

With dropout, no single neuron can be relied upon, forcing the network to distribute knowledge across many neurons — making it more robust.

**In DeepSKAN:** Dropout (p=0.3) is applied before each FC layer. This means 30% of the 1024/512/256 features are randomly zeroed during each training step.

---

## 9. The Training Process — How the CNN Learns {#9-training}

### Overview

Training = iteratively adjusting the millions of weights so the network's predictions match the correct labels.

```
For each batch of 64 training samples:
  1. Forward pass: run the input through the network → get predictions
  2. Compute loss: how wrong are the predictions? (cross-entropy)
  3. Backward pass: compute how each weight contributed to the error (backpropagation)
  4. Update weights: adjust each weight slightly to reduce the error (gradient descent)
```

Repeat for thousands of batches across many epochs.

### The Numbers

- **Parameters (weights):** ~2.4 million in the original DeepSKAN
- **Training samples per epoch:** 100,000 synthetic TA maps
- **Batch size:** 64 (process 64 samples at once for efficiency)
- **Epochs:** 40 (see all training data 40 times)
- **Learning rate:** 0.0001 (how big each weight update is)
- **Total weight updates:** 100,000/64 × 40 = 62,500 updates

### What Happens During Training (What We Observed)

```
Epoch  1: accuracy  12% (random guessing for 21 classes would be 4.8%)
Epoch  5: accuracy  48% (learning fast — low-hanging fruit)
Epoch 15: accuracy  70% (most easy classes learned)
Epoch 30: accuracy  74% (diminishing returns)
Epoch 40: accuracy  75% (converged — more epochs won't help much)
```

### Overfitting Warning Signs

If training accuracy keeps rising but validation accuracy stalls or drops:
```
Epoch 30: train_acc=85%, val_acc=73%  ← gap growing = overfitting
```

This means the network is memorizing the synthetic training data rather than learning general patterns. Our Dropout helps prevent this.

---

## 10. Synthetic Data Generation — The Physics Engine {#10-synthetic-data}

### Why Synthetic Data?

There are NO large labeled datasets of real TA spectra with known kinetic models. You can't collect 100,000 TA measurements with verified topologies. So we **simulate** them.

### How Each Sample Is Generated

For each of the 100,000 training samples:

1. **Pick a random class** (balanced — each of 21 classes appears equally often)

2. **Generate random rate constants:**
   - For each active pathway in the topology: k = 1/τ, where τ is drawn log-uniformly from [1, 5000] ps
   - Log-uniform means τ=1ps and τ=1000ps are equally likely

3. **Solve the kinetic ODE:**
   - Using `scipy.integrate.solve_ivp` (Runge-Kutta 4th/5th order)
   - Initial condition: species A fully populated, all others zero
   - Include Gaussian IRF (instrument response function) convolution
   - Get concentration profiles C(t) for each species

4. **Generate random EADS (spectra):**
   - Each species gets a spectrum made of 1-6 random Gaussians
   - Random center wavelength, width, and amplitude
   - This is where the sim-to-real gap comes from — real spectra have structured features (Q-bands, vibronic progressions), random Gaussians don't

5. **Build the 2D TA map:**
   - ΔA(λ,t) = Σ C_i(t) × ε_i(λ) − ε_GS(λ) (ground state bleach)

6. **Add noise:**
   - Gaussian white noise (random level 0-5% of signal)
   - Structured noise: baseline drift, scatter spike, coherent artifact near t=0
   - This makes the model robust to real-world imperfections

7. **Normalize:** divide by max(|ΔA|) → values in [-1, 1]

8. **Reshape:** to (1, 256, 64) — the CNN input format

### The Label

Each sample gets a class label (0-20) corresponding to its kinetic topology. The CNN learns to map (image → label).

---

## 11. Loss Function and Optimization {#11-loss-and-optimization}

### Cross-Entropy Loss

For classification, we use **cross-entropy loss**:

$$L = -\sum_{c=1}^{21} y_c \log(\hat{p}_c)$$

Where:
- $y_c$ = 1 if the true class is $c$, 0 otherwise (one-hot encoding)
- $\hat{p}_c$ = the network's predicted probability for class $c$

**Intuition:** If the true class is 6 and the network says p(class 6) = 0.9, loss is low. If p(class 6) = 0.01, loss is very high. The loss punishes confident wrong answers severely.

**At initialization (random weights):** all classes get probability ~1/21 ≈ 0.048. Loss = -log(0.048) = 3.04. That's why we see "loss ≈ 3.04" at epoch 0 — it's the mathematical baseline for 21 classes.

### Adam Optimizer

We use the **Adam** optimizer (Adaptive Moment Estimation):
- Learning rate: 0.0001
- It automatically adapts the step size for each parameter
- Combines the benefits of momentum (smooth updates) and RMSprop (scale-aware)

### Gradient Clipping

We clip gradients to max norm 1.0. This prevents **exploding gradients** — situations where a bad batch causes enormously large weight updates that destabilize the network.

### Learning Rate Scheduler

`ReduceLROnPlateau`: if validation loss doesn't improve for 3 epochs, multiply the learning rate by 0.1. This helps the optimizer settle into a good minimum in the later stages of training.

---

## 12. Inference — From Data to Prediction {#12-inference}

### What Happens When You Run the Trained Model on Real Data

1. **Load CSV** → (313 wavelengths, 258 time points)
2. **Preprocess:** interpolate to (256, 64), normalize to [-1,1]
3. **Add batch dimension:** (1, 256, 64) → (1, 1, 256, 64)
4. **Forward pass through CNN** (milliseconds on GPU, ~1 second on CPU)
5. **Get 21 raw scores** (logits)
6. **Softmax** converts to probabilities summing to 1.0
7. **Highest probability = prediction**

### Reading the Output

```
Class  6: probability 0.59   ← PREDICTED (59% confident)
Class 10: probability 0.14
Class  7: probability 0.09
...
```

This means: "I'm 59% sure this is Class 6 (A→B→C), 14% it could be Class 10, 9% Class 7..."

### What Does Confidence Mean?

- **>80%:** strong prediction — the model sees clear distinguishing features
- **50-80%:** moderate — the data is consistent with multiple models
- **<40%:** low — the model is guessing; don't trust it blindly

For our data:
- CBK-H: 59% → moderate confidence, the sequential model is the most likely
- CBK-D: 40% → low confidence, but notably ALL top candidates are branched models

---

## 13. What Can Go Wrong — Failure Modes {#13-failure-modes}

### The Sim-to-Real Gap (Main Failure)

The CNN was trained on synthetic data with random Gaussian spectra. Real molecular spectra have:
- Sharp Q-band bleach features
- Vibronic progression structure
- Stimulated emission bands
- Specific spectral correlations between species

The CNN never saw these patterns during training. It's like training a face recognizer on cartoons and testing on photographs.

### Preprocessing Destroys Information

When we normalize and interpolate to 256×64:
- The absolute time scale (in ps) is lost — only relative positions remain
- Amplitude information is lost — a strong signal looks the same as a weak one
- Wavelength calibration is approximate

This is fine for classification (which only needs shape), but when the subsequent GA fitting step uses this preprocessed data, it fails because it needs the physics.

### Low Confidence ≠ Wrong

A 40% confidence prediction is NOT the same as "wrong." It means the data doesn't strongly discriminate between models. In our case:
- Class 19 (40%) + Class 8 (22%) + Class 17 (17%) = 79% for BRANCHED models
- The individual class is uncertain, but the CATEGORY (branched) is fairly clear

### Class Ambiguity

Some of the 21 classes produce inherently similar TA maps. For example, with certain rate constants, a branched model can look almost identical to a sequential model. This is a fundamental physics limit — not a failure of the CNN.

---

## 14. Our Specific Results — CBK-H and CBK-D {#14-our-results}

### CBK-H (Protonated)

```
Top-1: Class 6 (59%) — A→B→C (sequential, 2 transfers)
Top-2: Class 10 (14%) — A→B→C with A also going to D
Top-3: Class 7 (9%) — A→B→C→GS (sequential, 3 transfers)
```

**Interpretation:** The CNN sees a clear sequential pattern in CBK-H. All top-3 predictions involve sequential elements. This matches our expectation from the classical analysis.

### CBK-D (Deuterated)

```
Top-1: Class 19 (40%) — A→B, A→C, B→C, B→D (complex branching)
Top-2: Class 8 (22%) — A→B→C with B also going to D
Top-3: Class 6 (17%) — A→B→C (sequential)
```

**Interpretation:** The CNN detects DIFFERENT visual patterns in CBK-D. The dominant prediction is branching, not sequential. The top-3 are split between branched (62%) and sequential (17%).

### The Scientific Hypothesis

If CBK-D truly has a branched topology while CBK-H is sequential, then:
- The "anomalous isotope effect" (D faster than H) isn't anomalous at all
- It's an artifact of forcing a wrong model (sequential) onto a branched system
- Deuteration doesn't just change the rates — it changes the PATHWAY

This is what makes DeepSKAN scientifically interesting: it generates hypotheses that a human expert might not consider.

### The Caveat

40% confidence is low. This hypothesis needs verification:
- Fit the branched model (Class 19) to CBK-D data explicitly
- Compare fit quality (residuals) between sequential and branched
- Look for spectral signatures of branching (e.g., isosbestic point violations)

---

## 15. Quick Reference — Numbers to Remember {#15-quick-reference}

### Architecture
- Input: 256×64 grayscale image (1 channel)
- Output: 21 class probabilities
- Depth: 7 residual blocks + 5 FC layers
- Parameters: ~2.4 million
- Early kernels: (15,1), (11,1) — time-domain features
- Late kernels: (5,5), (3,3) — spatial features
- Regularization: Dropout (30%), BatchNorm, GaussianNoise

### Training
- Data: 100,000 synthetic samples per epoch (generated on-the-fly)
- Classes: 21 kinetic topologies (4 species, balanced sampling)
- Optimizer: Adam, LR=0.0001
- Loss: Cross-entropy
- Epochs: 40
- Hardware: T4 GPU (Kaggle/Lightning.ai)
- Final accuracy: 72% validation (random = 4.8%)
- Training time: ~4-6 hours total

### Our Predictions
- CBK-H: Class 6 (sequential), 59% confidence
- CBK-D: Class 19 (branching), 40% confidence
- Key insight: isotope may change TOPOLOGY, not just rates

### Key Vocabulary for Q&A
- **Epoch:** one pass through all training data
- **Batch:** subset of samples processed together (64)
- **Overfitting:** model memorizes training data, fails on new data
- **Sim-to-real gap:** synthetic training data ≠ real experimental data
- **Softmax:** converts raw scores to probabilities (sum = 1)
- **Cross-entropy:** loss function for classification
- **Residual connection:** skip connection that helps deep networks train
- **BatchNorm:** normalizes intermediate values for stable training
- **Dropout:** randomly disables neurons during training (regularization)
- **Feature map:** output of one convolutional filter (a "pattern detector")
- **Global Analysis (GA):** classical method to fit kinetic models to TA data
- **EADS:** Evolution-Associated Difference Spectra (spectral fingerprint of each state)

---

## Common Q&A at the Poster

**Q: "Why not just use Global Analysis directly?"**
A: GA requires you to CHOOSE the model first. If you choose wrong, you get wrong lifetimes. DeepSKAN suggests the model without human bias.

**Q: "72% accuracy seems low?"**
A: Random is 4.8%, so 72% is very good. Also, many of the "errors" are between closely related topologies. Top-3 accuracy is much higher (~90%).

**Q: "Why did you use synthetic data?"**
A: There are no large labeled datasets of real TA spectra with verified kinetic models. Synthetic data is the only scalable approach. The trade-off is the sim-to-real gap.

**Q: "How do you know the branched model is correct for CBK-D?"**
A: We don't — it's a hypothesis. The CNN suggests it based on pattern recognition. Validation would require fitting the branched model explicitly and comparing residuals.

**Q: "What would you do differently?"**
A: Train on realistic spectra (from SVD of real data), use physics-informed methods (PINN), or use an agentic AI that reads literature to suggest models.

**Q: "Is this publishable?"**
A: The method + the hypothesis (isotope changes topology) is interesting. Needs validation with branched-model fitting and ideally additional measurements to confirm.
