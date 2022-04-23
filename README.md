## Abstract
This project aims to utilize text classification techniques to categorize tweets from Twitter as either positive or negative. The conceptually simple and empirically powerful BERT model is applied to pursue the optimal result.


## Results

| Model                             | Max Accuracy | Max F1-score |
| --------------------------------- | ------------ | ------------ |
| [Classic ML](classic_ml_baseline/)| 0.744        | 0.757        | 
| [GloVe + NN](neural_network/)     | 0.827        | 0.835        | 
| [BERT](bert/) (bert-base-uncased) | 0.894        | 0.895        | 


## Requirement 

* torch-1.10.0
* transformers-4.16.0
* datasets-1.15.1


## File Description

`run.py`:  Produce the same .csv predictions used in the best submission.

`bert_model.ipynb`: The notebook includes the pre-processing and complete BERT model.

`preprocessing.py`: Pre-defined functions for cleaning data.


## Reproductivity

It is strongly recommend to run the codes on Google Colab or GPU. 

To reproduce, files should be placed or saved in the following structure.

> **data**

>> train_pos_full.txt

>> train_neg_full.txt

>> test_data.txt

>> train_dataset.pt (save)

>> train_clean.tsv (save)


> **bert**

>> bert_model.ipynb

>> run.py

>> preprocessing.py

>> checkpoint-xxx (checkpoint bert model, save)


More specific comments are added codes by line.


### Author: Yuanjun Feng