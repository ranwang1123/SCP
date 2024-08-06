# Towards Robustness Prompt Tuning with Fully Test-Time Adaptation for CLIP’s Zero-Shot Generalization
This repository contains the implementation of our paper: "Towards Robustness Prompt Tuning with Fully Test-Time Adaptation for CLIP’s Zero-Shot Generalization". This work focuses on enhancing the robustness of prompt tuning by leveraging fully test-time adaptation techniques for the CLIP model. Accepted by ACM MM 2024.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the necessary dependencies, follow the instructions provided in the [INSTALL.md](SCP/docs/INSTALL.md) file.

```bash
# Clone the repository
git clone https://github.com/yourusername/repository_name.git
cd repository_name

# Follow the instructions in INSTALL.md
Usage
After installing the dependencies, you can use the following commands to run the experiments:

bash
Copy code
# Example command to run the training script
python train.py --config configs/config.yaml

# Example command to run the evaluation script
python evaluate.py --checkpoint checkpoints/best_model.pth --data data/test_dataset
Datasets
We used the following datasets for our experiments:

Dataset 1: Description of dataset 1
Dataset 2: Description of dataset 2
Please refer to the paper for more details on the datasets and their preparation.

Results
Our experiments demonstrate significant improvements in robustness and generalization of the CLIP model. Detailed results and comparisons are provided in the paper. Here are some key findings:

Improved zero-shot generalization accuracy by X%
Enhanced robustness against distribution shifts
Contributing
We welcome contributions from the community. If you would like to contribute, please follow these steps:

Fork the repository
Create a new branch (git checkout -b feature-branch)
Commit your changes (git commit -am 'Add new feature')
Push to the branch (git push origin feature-branch)
Create a new Pull Request
License
This project is licensed under the MIT License - see the LICENSE file for details.
