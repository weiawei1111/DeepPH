# Code for [DeepPH : A Multimodal Deep Learning Model for Predicting Enzyme Optimal pH Range]
This supplementary code supports the results in our paper.  

## File Descriptions

- `data_split.py`  
  Script for preprocess ing enzyme data into graph format with multimodal features for pH range prediction.

- `egnn_clean.py`  
  Defines the core EGNN architecture used in our method.

- `egnn_model_split_range.py`  
  The full model pipeline for training on range prediction tasks.

- `train_model_value_split_range.py`  
  Main training script for learning both the pH value and its interval (range).  
  Supports GPU training and logs intermediate metrics.

- `test_model_r2_split_aa_range3.pt`  
  The trained model containing learned weights.

-‘new_train_value.pkl, new_test_value.pkl and new_test_value_remove_phenv.pkl'
  They are our training dataset and two test sets.

## How to Run

1. Environment Setup  
   The proposed model was implemented using Python 3.8+, PyTorch 2.6.0 and PyTorch Geometric 2.6.1.
2. Execute the following command to train the DeepPH model: 
   'python train_model_value_split_range.py'



