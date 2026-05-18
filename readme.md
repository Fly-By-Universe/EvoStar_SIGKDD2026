# EvoStar: Structure-Guided Semi-Supervised Clustering via Evolving Star Network

> **EvoStar** is a highly scalable semi-supervised clustering framework that innovatively leverages a dynamically evolving, sparse hierarchical "star" topology to naturally integrate structural cues and human supervision. Driven by a lightweight pruning-grafting mechanism and active query prioritization, EvoStar achieves superior and robust clustering performance under tight constraint budgets, featuring exceptional **$O(n \log n)$** time and **$O(n)$** space complexity.

---

## 🌟 Key Highlights

* **Aggressive Early Gain:** EvoStar achieves a rapid performance surge in the very early stages of the semi-supervised learning process. On large-scale datasets like MNIST, CoverType, and HAR, it converges to optimal solutions with minimal human queries.
* **Exceptional Scalability:** Departing from traditional dense similarity graphs and global matrix transformations, EvoStar operates with **$O(n \log n)$** time and **$O(n)$** space complexity, effortlessly scaling to massive datasets.
* **Theoretical Guarantees:** The query selection is driven by a rigorously defined  **Representative Decay Model** . Theoretical analysis under mean-field approximations proves that prioritizing higher-level representative nodes yields exponentially growing expected semantic gains.
* **Monotonic & Robust Optimization:** EvoStar maintains a robust, monotonic improvement trajectory across iterations. Evaluated on 12 diverse benchmark datasets (spanning from high-dimensional biomedical data to time-series signals), it consistently avoids catastrophic performance degradation and exhibits strong robustness against annotation noise.

---

## 🧠 Core Algorithm: How EvoStar Works

EvoStar shifts the paradigm from "static graph inference" to "dynamic topology evolution." The framework operates in three core phases:

### Phase 1: Unsupervised Hierarchical Initialization

EvoStar begins by constructing a sparse hierarchical star topology in an unsupervised manner:

1. **Adaptive Density Peaks:** Automatically identifies representative density peaks based on local data distributions.
2. **Star Unit Construction:** Aggregates data points toward local representatives using nearest-neighbor links, forming multi-level "Star Units."

### Phase 2: Active Query Prioritization

Instead of uniform random querying, EvoStar employs a highly efficient active learning strategy. EvoStar is Guided by this theoretical foundation, EvoStar prioritizes querying nodes situated at  **higher hierarchy levels** , possessing **larger degrees** (more immediate descendants), and exhibiting **greater  distance** from their parents (indicating higher risk probability). This ensures every user annotation maximizes clustering boundary rectification.

### Phase 3: Lightweight Evolving Mechanism (Pruning & Grafting)

Upon receiving user feedback (Must-Link or Cannot-Link), EvoStar avoids costly global recomputations by executing localized topological updates:

* **Pruning:** If a structural error is detected, the framework immediately severs the anomalous parent-child edge, detaching the incorrect subtree.
* **Grafting:** The detached subtree is then grafted onto the safest, nearest alternative star center based on local homogeneity probabilities.
* **Cultivation:** Through these micro-level topological rearrangements, constraint information implicitly propagates across the hierarchy, dynamically reshaping the clusters without the need for external inference models.

---

## 📊 Performance at a Glance

* **Metrics:** Core evaluations are measured by Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).
* **Baselines:** EvoStar consistently outperforms state-of-the-art kernel-based and matrix-based semi-supervised clustering methods under identical or tighter constraint budgets.
