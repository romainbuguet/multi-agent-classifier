# Multi-Agent Classifier

This project implements a text classification system using LLM agents with different personas to analyze subtle differences in customer feedback — specifically distinguishing between constructive and unconstructive criticism. Each agent (e.g. Critic, Analyst, Customer) provides an analysis of the input text based on a unique role. A final meta-agent then synthesizes these perspectives to produce a transparent classification decision.

## Project Structure

```
├── classify.py                # Core logic for multi-agent classification
├── config.yaml                # Configuration file defining agents and label schema
├── evaluate_system.py         # Evaluation pipeline for model performance
├── generate_dataset.py        # Script for building synthetic datasets
├── app.py                     # Streamlit app to run classifications
├── requirements.txt           # Python dependencies for the project
├── README.md                  # This file
```

## Features

- **Multi-Agent Design:** Each agent analyzes feedback from a unique perspective (e.g., critic, analyst, customer).
- **Constructiveness Focus:** Classifies feedback as constructive or unconstructive.
- **Configurable Setup:** Easily modify agent definitions and behavior via `config.yaml`.
- **Evaluation Support:** Compare models accuracy.

## How It Works

1. **Data Preparation**: Use `generate_dataset.py` to prepare feedback samples.
2. **Classification**: `classifier.py` assigns labels based on the consensus or individual agent decisions.
4. **Evaluation**: Use `evaluate_system.py` to evaluate the system (accuracy).

## Results

| Classifier       | Accuracy | Correct Predictions |
|------------------|----------|----------------------|
| Baseline         | 69%      | 345 / 500            |
| Agent-Based      | 82%      | 410 / 500            |

The agent-based system clearly outperforms the baseline, showcasing the strength of diverse reasoning styles.

## License

MIT License
