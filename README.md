# Conversation Summarisation with RL
![image](https://github.com/user-attachments/assets/4a5211cf-6875-4e95-8663-116ae6e92a09)
## Overview

This repository demonstrates a two-phase training pipeline for dialogue summarisation:

1. **Supervised Fine‑Tuning**: T5-small and T5-base models are fine‑tuned on the SAMSum dialogue summary dataset.
2. **Reinforcement Learning (RL)**: The fine‑tuned checkpoints are further trained using Proximal Policy Optimization (PPO) to generate non‑toxic, plain summaries of toxic dialogues using a custom synthetic dataset.

## Features

* **Dialogue Summarisation**: Leverages the SAMSum corpus for high‑quality conversational summaries.
* **Synthetic Toxic Dataset**: Contains toxic dialogue inputs with neutral summaries to guide detoxification.
* **Reinforcement Learning**: PPO-based training via Hugging Face’s TRL library to minimise toxicity.
* **Evaluation Suite**: Automated ROUGE metric evaluation and toxicity checks using <b>"facebook/roberta-hate-speech-dynabench-r4-target"</b> model.

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/dark-horiznz/Conversation-Summarisation-with-RL.git
   cd Conversation-Summarisation-with-RL
   ```
2. **Set up a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

* **SAMSum**: Automatically loaded via:

  ```python
  from datasets import load_dataset
  samsum = load_dataset("knkarthick/samsum")
  ```
* **Toxic Conversations**: Provided in `data/toxic_conversations.json`.
  
  ```python
  from datasets import load_dataset
  samsum = load_dataset("majorSeaweed/toxic-dialogue-summarisation")
  ```

## Supervised Fine‑Tuning

Fine‑tune on the SAMSum dataset using the Hugging Face Trainer

## Reinforcement Learning (PPO)

Apply PPO to detoxify summaries on the toxic dataset

* **Reward Function**: Combines toxicity penalty (via a pretrained classifier) and fluency rewards.

## Inference

Generate summaries with the trained PPO model

## Evaluation

* **ROUGE Metrics**: Evaluate using built‑in evaluation scripts to compare against SAMSum references.
* **Toxicity Checks**: Run a toxicity classifier on output summaries to verify detoxification.

## Results

| Model                 | ROUGE‑1  | ROUGE‑2  | ROUGE‑L  |
| --------------------- | -------- | -------- | -------- |
| t5-small (supervised) | 52.1     | 26.3     | 49.7     |
| t5-small (PPO)        | **53.4** | **27.1** | **51.2** |
| t5-base (supervised)  | 54.8     | 29.0     | 52.4     |
| t5-base (PPO)         | **56.2** | **30.5** | **53.9** |

## SFT Model Comparsion
![image](https://github.com/user-attachments/assets/e60eff85-5d77-4a4a-ab48-497a255282f9)

## PPO Model Comparision (Toxicity Scores by BERT)

<b>1. T5 Base</b>

![image](https://github.com/user-attachments/assets/7c956c09-8580-4de0-baf1-537c1ae1a7ea)

<b>2. T5 Small</b>

![image](https://github.com/user-attachments/assets/e0f4890f-8a16-4d8e-a47d-9643bd3805c4)

## Contributing

Contributions welcome! Please open an issue or pull request with enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
