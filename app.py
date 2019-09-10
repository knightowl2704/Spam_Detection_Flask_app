from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open("/home/piyush/Desktop/Flask_a/svm.pkl",'rb'))
vectorizer = pickle.load(open("/home/piyush/Desktop/Flask_a/vectorizer.pickle", 'rb'))
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods = ["POST"])
def predict():
    message = request.form['message']
    data = [message]

    x = vectorizer.transform(data)

    y = model.predict(x)
    if str(y[0]) == 'ham':
        return "Not Spam"
    else:
        return "Spam"


app.run()
