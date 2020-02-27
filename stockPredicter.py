#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# essential libraries
from finml import *
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

# for logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# suppress warnings for deprecated methods from TensorFlow
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

################################################################

# Example of string operations
import pandas as pd
example_data = {'alphabets':['a,b,c', 'd,e,f', 'a,z,x', 'a,s,p']} 
example_df = pd.DataFrame(example_data) 

# Chain two string operations
example_df['alphabets'].str.upper().str.split(",")


# ### Data exploration and transformation
# The dataset has the following three columns:
# 1. date: This column contains the date of the observation.  
#   
#   
# 2. headlines: This column contains the concatenation of headlines for that date. The headlines are separated by the `<end>` string. For example, if there are three headlines `h1`, `h2`, and `h3` on a given day, the headline cell for that day will be the string `h1<end>h2<end>h3`.  
#   
#   
# 3. returns: This column contains the daily returns.



# Loads the dataset in a `Pandas` dataframe and plots the time series of the daily Apple returns (returns on the y-axis and dates on the x-axis).


df = pd.read_csv('AAPL_returns.csv', parse_dates=[0]) 
plt.figure(figsize=(20,5))
plt.bar(x=df['date'], height=df['RET'])
plt.xlabel("Dates")
plt.ylabel("Daily Returns")
plt.title("Apple Returns")
plt.legend(['Returns'], fontsize=12)

plt.show()



# Plots the time series of daily headline frequencies (the number of headlines per day on the y-axis and the corresponding date on the x-axis).
from matplotlib import pyplot as plt


df = pd.read_csv('Assignment4-data.csv', parse_dates=[1]) 
df['headline_freq'] = df['headlines'].str.split("<end>").str.len()


plt.figure(figsize=(20,5))
plt.bar(x=df['date'], height=df['headline_freq'])
plt.xlabel("Dates")
plt.ylabel("Frequency")
plt.title("Frequency of Headlines")
plt.legend(['Frequency of Headlines'], fontsize=12)
plt.show()



# Count the number of days on which the stock had positive and non-positive returns, respectively.

df = pd.read_csv('Assignment4-data.csv')

# Creates return_direction column 
df["returns_direction"] = np.where(df["returns"]>0, 1, 0)

# Counts the number of days the stock had positive and non-positive returns
num_days_return = df['returns_direction'].value_counts()
# 1 means positive
# 0 means non-positive
print("The number of days with positive days: {} ".format(num_days_return[1]))
print("The number of days with non-positive days: {} ".format(num_days_return[0]))


# Calculates the tf-idf metric for the following word and headline(s) pairs:
# 1. Word "apple" in headlines with date 2008-01-07. Store this value in a variable called `aaple_tfidf`.
# 2. Word "samsung" in headlines with date 2008-01-17. Store this value in a variable called `samsung_tfidf`.
# 3. Word "market" for news headlines with dates 2008-03-06. Store this value in a variable called `market_tfidf`.
# 
# Please write a Python code that calculates the metrics from the `df` dataframe.

import string

df = pd.read_csv('data.csv')
df = df.head(n=100)


# a loop that separates (or splits) each word in the document from one another, creating a list of all words
raw_document_words = [doc.split() for doc in df['headlines']]

document_words = []
for word_list in raw_document_words:
    temp_list = []
    for word in word_list:
        temp_list.append(word.translate(str.maketrans('', '', string.punctuation)))
    document_words.append(temp_list)
    
#turn our list into a sorted array (alphabetical order) which is a set (i.e. no duplicate words)
vocab = sorted(set(sum(document_words, [])))
#create a dictionary that takes each word as a key and their alphabetical order as a value
vocab_dict = {k: i + 1 for i, k in enumerate(vocab)}
#create a mxn TF matrix and initialise it with zeros. mxn because we have m (100) headlines and n words in our vocab
tf = np.zeros((len(df['headlines']), len(vocab)), dtype=int)
#for each word in our list of words
for i, doc in enumerate(document_words):
    for word in doc:
        if word == 'apple' or word == 'samsung' or word == 'market':
            tf[i, vocab_dict[word] - 1] += 1


idf = np.log(tf.shape[0]/tf.astype(bool).sum(axis=0))

tf_idf = tf * idf

for i in range(len(df)):
    if df.iloc[i].date == '2008-01-07':
        aaple_tfidf = tf_idf[i][vocab_dict['apple'] - 1]
    if df.iloc[i].date == '2008-01-17':
        samsung_tfidf = tf_idf[i][vocab_dict['samsung'] - 1]
    if df.iloc[i].date == '2008-03-06':
        market_tfidf = tf_idf[i][vocab_dict['market'] - 1]



print("NOTE: the following tf-idf scores are with a small amount of data cleaning (removing punctuation from words)")
print("The apple tf-idf is: {}".format(aaple_tfidf))
print("The samsung tf-idf is: {}".format(samsung_tfidf))
print("The market tf-idf is: {}".format(market_tfidf))


# Builds and train a **one**-layer neural network with two units (neurons) to explain return directions based on financial news.


df = pd.read_csv('data.csv')

# Creates return_direction column 
df["returns_direction"] = np.where(df["returns"]>0, 1, 0)

X_headlines = df.headlines.values
y = df.returns_direction
data = split_by_threshold(X_headlines, y, test_size=0.4,)
(train_texts, y_train, test_texts, y_test) = data
(X_train, X_test, vectorizer, k_best_selector) = ngram_vectorize(data,)


data = (X_train, y_train, X_test, y_test)
model = build_model(X_train.shape[1:], layers=1, units=2)
model = train_model(data, model)
evaluate(model, data)


# Explore the effects of different splits between the training and testing data on the performance of a given neural network model. 

df = pd.read_csv('data.csv')

# Creates return_direction column 
df["returns_direction"] = np.where(df["returns"]>0, 1, 0)

X_headlines = df.headlines.values
y = df.returns_direction
data = split_by_threshold(X_headlines, y, test_size=0.1,)
(train_texts, y_train, test_texts, y_test) = data
(X_train, X_test, vectorizer, k_best_selector) = ngram_vectorize(data,)
data = (X_train, y_train, X_test, y_test)


model = build_model(X_train.shape[1:], layers=2, units=3);
model = train_model(data, model)
evaluate(model, data)
model = build_model(X_train.shape[1:], layers=3, units=5);
model = train_model(data, model)
evaluate(model, data)


data = split_by_threshold(X_headlines, y, test_size=0.4,)
(train_texts, y_train, test_texts, y_test) = data
(X_train, X_test, vectorizer, k_best_selector) = ngram_vectorize(data,)
data = (X_train, y_train, X_test, y_test)


model = build_model(X_train.shape[1:], layers=2, units=3);
model = train_model(data, model)
evaluate(model, data)
model = build_model(X_train.shape[1:], layers=3, units=5);
model = train_model(data, model)
evaluate(model, data)


# | Num. Layers/Num. Units| Train/Test split | Precision  | Recall | Accuracy |
# | --------- |:---------:| -----:| -----:| -----:|
# | 2/3 | 90/10 | 0.5175097276264592 | 1.0 | 0.5175097276264592 |
# | 3/5 | 90/10 | 0.4897959183673469 | 0.7218045112781954 | 0.4669260700389105 |
# | 2/3 | 60/40 | 0.5223735408560312 | 1.0 | 0.5223735408560312 |
# | 3/5 | 60/40 | 0.5256975036710719 | 0.6666666666666666 | 0.5116731517509727 |
# 
# 
# As the Train/Test split goes from 90/10 to 60/40 the precision improves and the recall drops. This is most likely because in the 90 Train/10 Test Splits, the model is being overfitted since the model performs much better in the training set compared to the test set. This means the model is built too closely to the training set and has picked up the nuances of the dataset which is most likely just noise. The 60/40 does a better job of generalising the model since it isn't giving the model too much data for the model to fit to. Although it is typically thought that you want to give as much Testing and Training data as possible there are a lot of trade-offs and 60/40 seems to work well.  
# 
# With the 3/5 networks, the recall falls from 0.722 to 0.667 as the model is getting better classifying which is also seen in the precison and accuracy too. 
# 



# Runs a logistic regression with the same independent and dependent variables as used for the above neural network models.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv')

# Creates return_direction column 
df["returns_direction"] = np.where(df["returns"]>0, 1, 0)

X_headlines = df.headlines.values
y = df.returns_direction
data = split_by_threshold(X_headlines, y, test_size=0.4,)
(train_texts, y_train, test_texts, y_test) = data
(X_train, X_test, vectorizer, k_best_selector) = ngram_vectorize(data,)
data = (X_train, y_train, X_test, y_test)


clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
y_prediction = clf.predict(X_test)

recall = recall_score(y_test, y_prediction)
precision = precision_score(y_test, y_prediction)
accuracy = accuracy_score(y_test, y_prediction)

print("Precision is: {}".format(precision))
print("Recall is: {}".format(recall))
print("Accuracy is: {}".format(accuracy))



# The Logistic Regression is a decent classifier. It is better than some of the lower order neural networks (1/2, 2/3) as they just classified True for all instances whereas the logistic regression is genuinely starting to understanding the dataset and makes classifications. The logistic regression is limited by the fact that it has a single sigmoid function so can not truly understand the complexities of the natural language like a multi-layer/unit neural network can. 
# 
# A pro of the Logistic regression is that it does not need to be trained the way that a neural network does so can output results much faster.
# 
# Of the neural networks tested, the 3 layers/5 units with 60 training/40 testing performed the best because it was able to get the closest to understanding the complexity of the data and although it took longer than the Logistic Regression, it was able to perform slightly better so it is the preferred classifier to use.
# 
# 
# 
# | Classifiers | Precision  | Recall | Accuracy |
# | --------- | -----:| -----:| -----:|
# | Best Neural Network Tested (3/5, 60/40) | 0.5256975036710719 | 0.6666666666666666 | 0.5116731517509727 |
# | Logistic Regression | 0.5227606461086637 | 0.6629422718808193 | 0.5077821011673151 |
# 



# Everything so far was explaining stock returns with contemporaneous financial news that were released on the same date. To explore how well a neural network can **predict** the direction of **future** returns based on our text data

df1 = pd.read_csv('AAPL_returns.csv', parse_dates=['date']) 

df1['returns_pred'] = df1['RET'].shift(-1, fill_value=0)



df2 = pd.read_csv('data.csv', parse_dates=['date']) 

df = pd.merge(df1, df2, on='date')

# Creates return_direction column 
df["returns_direction"] = np.where(df["returns_pred"]>0, 1, 0)



X_headlines = df.headlines.values
y = df.returns_direction
data = split_by_threshold(X_headlines, y, test_size=0.4,)
(train_texts, y_train, test_texts, y_test) = data
(X_train, X_test, vectorizer, k_best_selector) = ngram_vectorize(data,)
data = (X_train, y_train, X_test, y_test)


model = build_model(X_train.shape[1:], layers=3, units=5);
model = train_model(data, model)
evaluate(model, data)




