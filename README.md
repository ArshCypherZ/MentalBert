# MentalBert

MentalBert is a text classification model designed to classify whether a given text is depressive or not. This project leverages the BERT architecture and is fine-tuned on a specific dataset to improve its performance in identifying depressive texts.

## Project Structure

```
MentalBert/
├── run.py              # Code to run the model for classification
├── finetune1.py       # Code for fine-tuning the MentalBert model using dataset1
├── finetune2.py       # Code for fine-tuning the MentalBert model using dataset2
└── datasets/
    ├── dataset1.csv   # Dataset for fine-tuning model 1
    └── dataset2.csv   # Dataset for fine-tuning model 2
```

## Files Description

- **run.py**: This script contains the implementation for running the MentalBert model. It takes input text and classifies it as depressive or non-depressive based on the fine-tuned model.
  
- **finetune1.py**: This script is used for fine-tuning the MentalBert model using the dataset located in `datasets/dataset1.csv`. The fine-tuning process adjusts the model parameters for better accuracy on the specific dataset.

- **finetune2.py**: Similar to `finetune1.py`, this script fine-tunes the MentalBert model using a different dataset located in `datasets/dataset2.csv`. 

## Requirements

Make sure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Model

To classify text using the pre-trained MentalBert model, execute:

```bash
python run.py
```

### Fine-Tuning the Model

To fine-tune the model with the first dataset, run:

```bash
python finetune1.py
```

For the second dataset, use:

```bash
python finetune2.py
```