# Sentiment-Analysis
# Sentiment Analysis for Arabic Tweets

This project focuses on sentiment analysis for Arabic tweets using classical machine learning models.

## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Preprocess Data](#preprocess-data)
- [Features Extraction](#features-extraction)
- [Model Selection and Training](#model-selection-and-training)
- [Results and Evaluation](#results-and-evaluation)
- [Test the Model with New Text](#test-the-model-with-new-text)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

Sentiment analysis, also known as opinion mining, is a branch of Natural Language Processing (NLP) aiming to determine sentiment or emotion in text. This project develops a sentiment analysis system for Arabic tweets using classical machine learning models.

In this project, i explore and compare various classical machine learning algorithms for sentiment analysis, including Naive Bayes, Logistic Regression, Support Vector Machines (SVM), MLP Classifier, and Random Forest Classifier. By applying these algorithms to our Arabic sentiment analysis task, we aim to identify the most effective model for accurate sentiment prediction

## Dataset Description

The project uses the [Arabic Sentiment Twitter Corpus](https://www.kaggle.com/datasets/mksaad/arabic-sentiment-twitter-corpus) available on Kaggle. The dataset consists of 45K balanced Arabic tweets (23K positive, 22K negative) collected from Twitter for sentiment analysis tasks.

## Preprocess Data

Before conducting sentiment analysis on the dataset, several preprocessing steps were applied to clean and transform the data for better analysis and model training.
1.	Cleaning tweet text
2.	Removing punctuations
3.	Normalizing Arabic text   
4.	Removing repeating characters
5.	Tokenization
6.	Stop Word Removal
   
By applying these preprocessing steps, the tweet text data was effectively cleaned and transformed, making it suitable for further feature extraction, model training, and sentiment analysis tasks.

## Features Extraction

To represent the tweet text data in a numerical format suitable for machine learning models, the TF-IDF (Term Frequency-Inverse Document Frequency) technique was employed. TF-IDF assigns weights to words based on their frequency in a particular document (tweet) and their inverse frequency across all documents (tweets).
The following steps were performed for features extraction using TF-IDF unigram:
1.	TF-IDF Vectorization
2.	Vocabulary Extraction
3.	Feature Matrix Construction
4.	Binary Encoding of Class Labels

The resulting feature matrix, along with the encoded class labels, can now be used for training machine learning models and performing sentiment analysis on Arabic tweets.

## Model Selection and Training

1.	Model Selection
Several popular machine learning algorithms were considered for sentiment analysis, including:
•	Naive Bayes Algorithm
•	Logistic Regression Algorithm
•	Support Vector Machines Algorithm
•	MLP Classifier Algorithm
•	Random Forest Classifier Algorithm

Each algorithm was chosen based on its suitability for text classification tasks and its potential to capture the sentiment information from the tweet text.

2.	Model Training
For each selected algorithm, the following steps were performed:
•	Splitting the preprocessed dataset into training and testing sets ( using an 70:30 ratio).
•	Initializing an instance of the chosen algorithm with default hyperparameters.
•	Training the model on the training set, which involves feeding the feature matrix and corresponding class labels.
•	Evaluating the trained model's performance on the testing set, using metrics such as accuracy, precision, recall, and F1-score.
•	Repeating the training and evaluation process for each selected algorithm.

## Results and Evaluation

The trained machine learning models were evaluated using various performance metrics to assess their effectiveness in sentiment analysis on Arabic tweets. The following models were considered:
1.	Naive Bayes (GaussianNB)
•	Accuracy: 0.723
•	Precision: 0.67%
•	F1-Score: 0.67%

3.	Logistic Regression
•	Accuracy: 0.767
•	Precision: 0.76%
•	F1-Score: 0.76%

5.	Support Vector Machines (SVM)
•	Accuracy: 0.524

7.	MLP Classifier (Neural Network)
•	Accuracy: 0.771

9.	Random Forest Classifier
•	Accuracy: 0.781


Models	                    Accuracy
Naive Bayes	                  0.723
Logistic Regression   	      0.767
Support Vector Machines      	0.524
MLP Classifier	              0.771
Random Forest Classifier      0.781

Based on the evaluation results, it can be observed that the Random Forest Classifier achieved the highest accuracy among the models with an accuracy of 0.781. 
## Test the Model with New Text

## Test the model with now text

To provide a user-friendly interface for sentiment analysis, a GUI was developed using the Tkinter library in Python. The GUI allows users to enter text and obtain sentiment analysis results using the trained model.
![image](https://github.com/rahaf-fahad/Sentiment-Analysis/assets/95524346/e3a0566e-2a62-458b-aa3a-3d835072ebe7)
![image](https://github.com/rahaf-fahad/Sentiment-Analysis/assets/95524346/e1c714fe-1121-4f8b-afb3-359dac614f22)



## Conclusion

the project demonstrated the effectiveness of machine learning algorithms in sentiment analysis and showcased the potential applications of sentiment analysis in understanding public sentiment from Arabic tweets. The insights gained from this project can be valuable for various domains, including social media monitoring, market research, and public opinion analysis.


## References

- [Reference 1](https://www.kaggle.com/code/wassimchouchen/arabic-sentiment-analysis-using-mnb-improved)
- [Reference 2](https://www.kaggle.com/code/mksaad/sentiment-analysis-in-arabic-tweets-using-sklearn)
- [Reference 3](https://www.kaggle.com/code/wahajalghamdi/arabic-language-twitter-sentiment-analysis/notebook#Features-Extraction-from-tweets-text-with-TFIDF-unigram)
- [Reference 4](https://www.kaggle.com/datasets/mksaad/arabic-sentiment-twitter-corpus)
- [Reference 5](https://huggingface.co/blog/sentiment-analysis-python#2-how-to-use-pre-trained-sentiment-analysis-models-with-python)
