# AdversarialAgentTest: Stress-Testing LLM Robustness in Financial Decision-Making

## Project Overview
AdversarialAgentTest is an **AI evaluation and robustness testing framework** developed to assess how reliably a **Large Language Model (LLM)** performs financial decision-making when exposed to **adversarial or manipulated market signals**. The project simulates real-world conditions where financial AI agents may encounter noisy, conflicting, or emotionally exaggerated inputs and evaluates whether the model’s trading decisions remain stable.

The objective is to test the **robustness, reliability, and failure boundaries** of an LLM-based trading agent by deliberately injecting adversarial noise into cryptocurrency news headlines and measuring changes in predictive performance.

---

## Dataset Source
The benchmark dataset used in this project was derived from my previous project **FinAgentEval**, where the initial cryptocurrency news dataset was collected, sentiment analyzed, and labeled for **BUY**, **SELL**, and **HOLD** decisions.

🔗 Dataset Source: https://github.com/najiyak23/FinAgentEval

---

## Objectives
- Build an adversarial stress-testing framework for financial AI agents
- Simulate noisy and manipulative market inputs
- Evaluate LLM robustness under adversarial conditions
- Measure decision degradation using statistical metrics
- Identify model failure modes and misclassification patterns

---

## Methodology
This project was developed as an **adversarial AI evaluation framework** using a previously constructed **cryptocurrency news benchmark dataset** containing financial headlines and associated **BUY**, **SELL**, and **HOLD** labels. The original dataset was imported from the baseline evaluation project and used as the ground-truth reference for robustness testing.

An **adversarial data generator** was implemented in **Python** to deliberately manipulate the original headlines by introducing **noise injection techniques**, including:
- emotional amplification (e.g., **"URGENT!!!"**)
- punctuation distortion
- sentiment flipping

These transformations simulate real-world adversarial scenarios where AI trading systems may receive misleading or exaggerated market information.

The adversarial headlines were then passed to a **Large Language Model** through **OpenRouter API** using a controlled **prompt-engineering pipeline**, where the model generated one trading decision (**BUY**, **SELL**, or **HOLD**) for each manipulated headline.

The predictions were compared against the original ground-truth labels using **statistical evaluation metrics** implemented in **Scikit-learn**, including:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The final analysis measured **robustness degradation**, defined as the change in predictive performance between the baseline model and the adversarial evaluation.

---

## Results

### Baseline Accuracy
**50%**

### Adversarial Accuracy
**50%**

### Robustness Drop
**0%**

### Key Findings
- The model showed **no significant accuracy degradation** under simple adversarial perturbations.
- This indicates **robustness to basic text manipulation and noisy market signals**.
- However, the model continued to struggle with **HOLD** predictions.
- The primary failure mode was **ambiguity handling**, not adversarial sensitivity.

---

## Tech Stack
- Python
- Pandas
- Scikit-learn
- OpenRouter API
- python-dotenv
- Git & GitHub

---

## Future Improvements
- Test stronger adversarial attacks (**contradictory multi-signal inputs**).
- Compare multiple LLMs (**GPT-4o, Gemini, local models via Ollama**).
- Add **latency and response-time benchmarking**.
- Build **automated adversarial test dashboards**.
- Expand to **live crypto market feeds and on-chain signals**.
