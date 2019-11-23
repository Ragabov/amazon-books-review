# Amazon Book Reviews 3-class sentiment classifier
#### Note
This repository contains all the needed code to train/test/run a simple flask review classification service, but all the trained models and data files are not included. Check the explotary notebook for the details on the data wrangling/analysis and the training of the traditional classification models reported below.
### Introduction
This repository contains the code for replicating results described below and can be used to run a simple flask service delivering one of two classification models (Traditional Voting Ensemble or a Deep Learning Bi-GRU model)

## Service API Documentation

### 

## 1. POST/GET `/classify`

### Request
The `classify` endpoint requires a single argument `utterance` that contains the text to classify, an example of a valid GET request is as follows : 
`SERVICE_URL/classify?utterance=I loved it`

### Response
A request to the method `classify` returns a json object that contains the predicted class as in the example below : 
```
{
    "detected_class": "neutral"
}
```

The `detected_class` key can take one of three values : {`negative`, `neutral`, `positive`} 

## Experiments Results
Below are the results of the top scoring classifiers and their tuned hyper parameters values

| # model | # hyper-parameters | accuracy   | macro F1 | Micro F1  
| ------------------ | ------------------ | ------- | ------------------ | ------------------ 
| MultiNomial Naive Bayes | alpha: 0.2 - Uniform Prior | 85% | 0.69 | .87  
| Logistic Regression | L1 Regularization - balancing class weights - C: 1 - one-vs-rest | 89%| .73 | .90
| Linear Kernel Support Vector | L2 Regularization -balancing class weights - C: .2 - one-vs-rest | 90% |.73 | .90 
| Soft Voting Ensemble (MNB + LR + Calibrated Linear SV) | 100 | 90% | .74 | .90 
| Bi-LSTM GRU + Twitter-Glove-200d embeddings | hidden-size of GRU cells : 100 -  learning-rate: .001 - Early Stopping| 86% | .70 | .87

## Usage


### Requirements
* Python 3.X with pip + installing the dependencies in the `requirements.txt` file

#### Training the traditional ML models
* Run the `explotary_notebook` residing in the `explotary_analysis` directory

#### Training the Bi-GRU 
* Download a gensim compatible word2vec/glove embeddings 
* Build the data by running `python build_data_files.py` with the appropriate arguments
* Train the model by running `python training.py` with the appropriate arguments
* Test the model by running `python testing.py` with the appropriate arguments

#### Running Flask Classification Service
To run the flask service using a trained model : 
* Run `python sentiment.py` with the appropriate arguments

