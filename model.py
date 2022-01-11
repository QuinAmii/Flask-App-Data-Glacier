#Importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv('/content/hiring.csv')


dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)
X = dataset.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
  word_dict = {'one' :1, 'two' :2, 'three' :3, 'four' :4, 'five' :5, 'six' :6, 'seven' :7, 'eight' :8, 'nine' :9, 'ten' :10, 'eleven' :11, 'twelve' :12,
               'zero':0, 0: 0}
  return word_dict[word]

  X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]


#spilt train and test set
#since our dataset is small we will train avaliable data

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#fitting model with train dataset
regressor.fit(X, y)

#Saving model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))


#loading model to compare results
model = pickle.load(open('model.pkl', 'rb'))








































from flask import Flask, request, jsonify, render_template 
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

    @app.route('/predict', methods=['POST'])
  
def predict():
    ...

    @app.route('/predict_api', methods=['POST'])
