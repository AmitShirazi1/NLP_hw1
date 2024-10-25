# MEMM Model for Part-of-Speech Tagging - NLP project

## Overview
This project implements a Maximum Entropy Markov Model (MEMM) for part-of-speech (POS) tagging. The main objective is to accurately tag words in sentences based on their context. The implementation is structured into several Python files, each serving a specific role in the process.

## Code Files

### 1. `preprocessing.py`
This file contains the `FeatureStatistics` and `Feature2id` classes that are responsible for feature extraction from the input data. The `FeatureStatistics` class accumulates statistics for each feature, while `Feature2id` assigns unique indices to each feature based on their occurrence in the training dataset. Key functions include:
- Extracting word/tag pairs and their features.
- Creating a sparse matrix representation of the features.

### 2. `optimization.py`
This file handles the optimization of the model parameters. It includes functions that apply optimization techniques to maximize the likelihood of the observed data, refining the model's performance based on the extracted features.

### 3. `main.py`
The main execution file that ties together the various components of the project. It facilitates the flow of data through preprocessing, feature extraction, and model training. Key functionalities include:
- Loading data files.
- Initiating the preprocessing steps.
- Executing the optimization and training processes.

### 4. `inference.py`
This file is responsible for applying the trained MEMM model to new data. It includes methods for predicting POS tags for unseen sentences and evaluating the model's performance on test datasets.

### 5. `generate_comp_tagged.py`
This script is used for generating tagged sentences from the provided words and tags. It serves as a utility for preparing the input data for model training and evaluation.

## Data Files
The project includes several data files necessary for training and testing the model:
- `comp1.words` and `comp1.wtag_public`
- `comp2.words` and `comp2.wtag_public`
- `test1.wtag`
- `train1.wtag` and `train2.wtag`

While the data files are essential for the functionality of the model, the primary focus of this project lies in the code implementation.

## How to Run the Project
1. Ensure you have the necessary Python libraries installed.
2. Place the data files in the designated `data` folder.
3. Execute the `main.py` file to start the training and evaluation process.

## Conclusion
This project serves as a comprehensive implementation of a MEMM for POS tagging, demonstrating the importance of feature extraction and optimization in NLP tasks. The code files provided are crucial for understanding and extending the functionality of the model.
