# OOC-Verify — Model Freeze Record

## Version

**OOC-Verify v1.0 — Frozen Experimental Version**

Status: **FROZEN**

Freeze purpose: Establish a fixed experimental version for final evaluation and research reporting.

---

## 1. System Overview

OOC-Verify is an explainable multimodal system for detecting potentially out-of-context image-caption misinformation.

The system combines:

1. Image-caption semantic alignment
2. Multimodal visual reasoning
3. External evidence retrieval
4. Evidence analysis
5. Evidence-aware decision fusion
6. Structured explanation generation

---

## 2. Frozen AI Components

### Image-Text Alignment

Model:

`openai/clip-vit-base-patch32`

Purpose:

- Extract image and text representations
- Measure image-caption semantic alignment
- Provide an alignment signal to the verification pipeline

---

### Alignment Classifier

Model:

`Logistic Regression`

Training source:

`NewsCLIPpings-derived CLIP embeddings`

Purpose:

Convert CLIP alignment features into an alignment-based prediction.

Validation performance:

- Accuracy: 64.09%
- Precision: 65.63%
- Recall: 59.17%
- F1 Score: 62.23%
- ROC-AUC: 71.28%

---

### Multimodal Language Model

Model:

`Qwen/Qwen2.5-VL-3B-Instruct`

Purpose:

- Analyze the image together with the caption
- Determine whether the caption is supported by visible visual evidence
- Distinguish direct visual claims from contextual claims
- Generate a short reasoning explanation

Inference configuration:

- CUDA-enabled inference
- GPU/CPU offloading
- Maximum GPU memory allocation configured for the available RTX 3050 6 GB GPU

---

## 3. Evidence Retrieval

Source:

`Google News RSS`

Purpose:

Retrieve external news evidence relevant to contextual claims.

Evidence items are classified as:

- supports
- contradicts
- insufficient
- irrelevant

Evidence retrieval is not a trained machine-learning model.

---

## 4. Decision Pipeline

The frozen proposed configuration follows:

Image + Caption
        ↓
CLIP Image-Text Alignment
        ↓
Alignment Classifier
        ↓
Qwen2.5-VL Multimodal Reasoning
        ↓
Contextual Claim Detection
        ↓
External Evidence Retrieval
        ↓
Evidence Analysis
        ↓
Evidence-Aware Decision Fusion
        ↓
Prediction + Confidence + Reason + Evidence + Explanation

---

## 5. Evaluation Configurations

The system supports three experimental configurations:

### Alignment Only

CLIP → Alignment Classifier → Final Decision

MLLM and evidence retrieval are skipped.

### MLLM Only

Image + Caption → Qwen2.5-VL → Final Decision

CLIP alignment and evidence retrieval are skipped.

### Proposed OOC-Verify

CLIP → Alignment Classifier → Qwen2.5-VL → Evidence Retrieval → Evidence Analysis → Evidence-Aware Fusion

---

## 6. Frozen Decision Policy

The decision logic was finalized before the final evaluation phase.

The final evaluation must not be used to modify:

- Model selection
- Model weights
- Prompts
- Decision thresholds
- Evidence rules
- Fusion rules
- Contextual-claim rules

Any future improvement must be treated as a new system version.

---

## 7. Evaluation Policy

After this freeze point:

> Final evaluation cases are used only to measure system performance and identify limitations. They must not be used to tune or modify the frozen system.

Evaluation failures will be recorded honestly.

If improvements are required after evaluation, they will be implemented as a new version rather than modifying this frozen experimental version.

---

## 8. Known Limitations

The frozen system has the following limitations:

1. CLIP alignment measures semantic similarity and does not prove image provenance.
2. The MLLM may fail on complex contextual claims.
3. External evidence does not automatically prove that the submitted image is the image used in the reported event.
4. Exact image provenance verification is not implemented as a dedicated module.
5. Qwen2.5-VL-3B-Instruct requires GPU/CPU offloading on the available hardware.
6. Decision confidence is a heuristic confidence score and is not a statistically calibrated probability.
7. Evidence retrieval and evidence classification contain rule-based components.
8. VisualNews is not part of the current executed experimental pipeline.

---

## 9. Research Integrity

This freeze establishes a reproducible boundary between:

**Model Development / Tuning**

and

**Final Evaluation**

No evaluation result will be selectively used to modify the frozen system.

---

## 10. Versioning

Current version:

`OOC-Verify v1.0`

Status:

`FROZEN`

Future changes must increment the version, for example:

`OOC-Verify v1.1`

or

`OOC-Verify v2.0`

depending on the scope of the changes.

---

## 11. Freeze Declaration

The OOC-Verify v1.0 experimental pipeline is frozen for final evaluation.

The system configuration, model selection, prompts, decision logic, evidence analysis logic, and evaluation methodology are considered fixed after this freeze point.