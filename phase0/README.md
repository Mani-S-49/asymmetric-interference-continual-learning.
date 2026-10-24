# Phase 0 — Research Onboarding & Continual Learning Foundations

## Project

**Cross-Lingual Continual Learning with Interference-Aware Replay**

Phase 0 is the research onboarding stage. The goal is to understand the continual-learning foundations, cross-lingual continual learning, cross-lingual interference, replay methods, and language-similarity priors before implementing the main experimental framework.

---

# 1. Phase 0 Goals

By the end of Phase 0, I should be able to:

* Explain continual learning.
* Explain catastrophic forgetting.
* Explain forward transfer and backward transfer.
* Explain cross-lingual continual learning.
* Explain why language order can affect continual-learning performance.
* Explain experience replay.
* Explain why replay sample selection matters.
* Explain cross-lingual interference.
* Explain why interference can be asymmetric between languages.
* Explain the motivation for using language similarity/typology as a prior.
* Reproduce a small continual-learning experiment showing forgetting.
* Explain how the Phase 0 concepts connect to the later research phases.

---

# 2. Environment Checklist

* [x] VS Code project opened
* [x] Python virtual environment created
* [x] Python version verified
* [x] NVIDIA GPU detected
* [x] CUDA-enabled PyTorch installed
* [x] PyTorch CUDA availability verified
* [x] GPU computation tested
* [x] Phase 0 directory created
* [x] Reading-notes directory created
* [x] Experiments directory created
* [x] Results directory created

Current hardware:

* GPU: NVIDIA GeForce RTX 2050
* VRAM: 4 GB
* PyTorch: 2.14.1+cu130
* CUDA runtime: 13.0

---

# 3. Phase 0 Directory Structure

```text
phase0/
│
├── README.md
│
├── reading_notes/
│
├── experiments/
│
└── results/
```

---

# 4. Mandatory Papers

## Paper 1 — Cross-Lingual Continual Learning

**M'hamdi, Ren & May (2023)**

Status: [ ] Not started

Focus:

* Cross-lingual continual learning problem
* Continual-learning setup
* Catastrophic forgetting
* Forward/backward transfer
* CFT
* CBT
* MFT
* MBT
* Experimental protocol

Notes:

`reading_notes/01_mhamdi_2023.md`

---

## Paper 2 — Massively Multilingual Continual Learning

**Winata et al. (2023)**

Status: [ ] Not started

Focus:

* Massively multilingual continual learning
* Language streams
* Catastrophic forgetting
* LR-ADJUST
* Experimental design
* 51-language setting

Notes:

`reading_notes/02_winata_2023.md`

---

## Paper 3 — Cross-Lingual Interference

**Alastruey et al. (2025)**

**Interference Matrix: Quantifying Cross-Lingual Interference in Transformer Encoders**

Status: [ ] Not started

Focus:

* Interference matrix
* Directional interference
* Asymmetric interference
* Language pairs
* Script effects
* Relationship between language similarity and interference
* Connection to the research hypothesis

Notes:

`reading_notes/03_alastruey_2025.md`

---

## Paper 4 — COMPASS

**Flynn (2026)**

**COMPASS: COntinual Multilingual PEFT with Adaptive Semantic Sampling**

Status: [ ] Not started

Focus:

* Multilingual PEFT
* Semantic sampling
* Replay/sample selection
* Continual multilingual learning
* Comparison with the proposed research direction

Notes:

`reading_notes/04_compass_2026.md`

---

## Paper 5 — GCR

**Tiwari et al. (2022)**

**GCR: Gradient Coreset Based Replay Buffer Selection for Continual Learning**

Status: [ ] Not started

Focus:

* Gradient-based replay selection
* Coresets
* Replay buffer construction
* Gradient similarity/diversity
* Connection to the GCR baseline

Notes:

`reading_notes/05_gcr_2022.md`

---

## Paper 6 — DER++

**Buzzega et al. (2020)**

**Dark Experience for General Continual Learning**

Status: [ ] Not started

Focus:

* Experience replay
* Logit replay
* Dark knowledge
* DER
* DER++
* Replay buffer
* Connection to the DER++ baseline

Notes:

`reading_notes/06_derpp_2020.md`

---

## Paper 7 — EWC

**Kirkpatrick et al. (2017)**

**Overcoming Catastrophic Forgetting in Neural Networks**

Status: [ ] Not started

Focus:

* Catastrophic forgetting
* Stability vs plasticity
* Parameter importance
* Fisher information
* Elastic Weight Consolidation
* Classical continual learning

Notes:

`reading_notes/07_ewc_2017.md`

---

## Paper 8 — URIEL / lang2vec

**Littell et al. (2017)**

Status: [ ] Not started

Focus:

* Language representations
* Typological features
* Geographic features
* Genetic features
* Language vectors
* lang2vec
* Using language similarity as a prior

Notes:

`reading_notes/08_uriel_2017.md`

---

# 5. Paper Reading Template

For every paper, answer:

## 1. Research Problem

What problem is the paper trying to solve?

## 2. Motivation

Why is this problem important?

## 3. Main Idea

What is the central idea of the paper?

## 4. Method

How does the proposed method work?

## 5. Dataset / Languages

What datasets, languages, tasks, or models were used?

## 6. Experimental Setup

How were the experiments conducted?

## 7. Metrics

What metrics were used?

## 8. Main Results

What did the experiments demonstrate?

## 9. Important Equations

Record equations that are necessary for understanding the method.

## 10. Limitations

What limitations or open questions remain?

## 11. What I Learned

Write the most important lessons in your own words.

## 12. Connection to My Project

How does this paper affect my research design?

## 13. Questions

What do I still not understand?

---

# 6. Reading Order

The recommended order is:

```text
01 M'hamdi et al. (2023)
        ↓
02 Winata et al. (2023)
        ↓
03 Alastruey et al. (2025)
        ↓
04 COMPASS
        ↓
05 GCR
        ↓
06 DER++
        ↓
07 EWC
        ↓
08 URIEL / lang2vec
```

The first three papers are the highest-priority papers because they establish the research problem and motivate the interference-aware direction.

---

# 7. Toy Continual-Learning Experiment

After the paper-reading foundation, implement a small continual-learning experiment.

The experiment should contain three sequential tasks:

```text
Task A
  ↓
Train
  ↓
Evaluate A
  ↓
Task B
  ↓
Train
  ↓
Evaluate A + B
  ↓
Task C
  ↓
Train
  ↓
Evaluate A + B + C
```

The purpose is to observe catastrophic forgetting directly.

The experiment should record:

* Accuracy on Task A after Task A
* Accuracy on Task A after Task B
* Accuracy on Task A after Task C
* Accuracy on Task B after Task B
* Accuracy on Task B after Task C
* Accuracy on Task C after Task C

Save experiment code in:

```text
experiments/
```

Save results in:

```text
results/
```

---

# 8. Phase 0 Metrics

At minimum, understand:

### Average Accuracy

Performance across the tasks/languages after continual learning.

### Backward Transfer

How learning later tasks affects previously learned tasks.

### Forward Transfer

How knowledge from earlier tasks affects learning of future tasks.

### Forgetting

Performance degradation on previously learned tasks after subsequent learning.

For the later cross-lingual experiments, also understand:

* CFT
* CBT
* MFT
* MBT

---

# 9. Phase 0 Completion Gate

Phase 0 is complete only when:

## Research Understanding

* [ ] I can define continual learning.
* [ ] I can explain catastrophic forgetting.
* [ ] I understand forward transfer.
* [ ] I understand backward transfer.
* [ ] I understand experience replay.
* [ ] I understand replay-buffer selection.
* [ ] I understand cross-lingual continual learning.
* [ ] I understand cross-lingual interference.
* [ ] I understand why interference may be asymmetric.
* [ ] I understand the motivation for a language-similarity prior.

## Paper Reading

* [ ] Paper 1 completed
* [ ] Paper 2 completed
* [ ] Paper 3 completed
* [ ] Paper 4 completed
* [ ] Paper 5 completed
* [ ] Paper 6 completed
* [ ] Paper 7 completed
* [ ] Paper 8 completed

## Engineering

* [ ] Toy continual-learning experiment implemented
* [ ] Sequential training works
* [ ] Evaluation after every task works
* [ ] Forgetting is observable
* [ ] Results are saved
* [ ] Experiment can be rerun

## Research Connection

* [ ] I can explain why replay may reduce forgetting.
* [ ] I can explain why replay selection matters.
* [ ] I can explain why language relationships may help replay selection.
* [ ] I can explain how the Phase 0 findings motivate Phase 1.

---

# 10. Rule for Phase 0

Do not start the full research experiment yet.

Do not implement:

* XNLI stream
* Llama-3.2-1B
* LoRA research model
* B0-B8
* 5 stream permutations
* 10-language experiments
* W&B full experiments
* scaling experiments

Those belong to later phases.

The purpose of Phase 0 is to establish the conceptual and experimental foundation first.

---

# 11. Current Status

**Current phase:** Phase 0

**Current step:** Environment setup completed.

**Next step:** Begin Paper 1 — M'hamdi, Ren & May (2023).

**Phase 0 status:** In progress
