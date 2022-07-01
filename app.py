import numpy as np
import pandas as pd
import pickle
import sklearn
import joblib
from flask import Flask, render_template, request

#------------------ Purification ------------------- 
import re
import string 
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import remove_stopwords
wnl = WordNetLemmatizer()

def purification(text):
    #converting to lower case
    text = text.lower()

    # removing html tags
    pattern_html = re.compile('<.*?>')
    text=pattern_html.sub(" ", text) 

    #removing urls
    pattern_urls = re.compile('https?://\S +|www\.\S+')
    text = pattern_urls.sub(' ', text)

    #removing punctuations
    for i in string.punctuation:
        text = text.replace(i, "")
    
    #rectifying Spelling mistakes
    text = TextBlob(text).correct()

    #lemmatizing text
    words = word_tokenize(str(text)) ###
    lemm_string = []
    for word in words:
        lemm_string.append(wnl.lemmatize(word, pos='v'))
    text =  ' '.join(lemm_string)

    #removing Stop words
    text = remove_stopwords(text) 
    
    return text

##------------------- unpickling----------------- 
model = pickle.load(open('model.pkl', 'rb'))
# purification = pickle.load(open('purification.pkl', 'rb'))
transformer = pickle.load(open('transformer.pkl', 'rb'))


#-----------------------Flask-----------------------
app = Flask(__name__)  # important
# server=app.server

@app.route('/')  # for pointing homepage
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():

    text = request.form.get('text_input') #-----------str
    text_p = purification(text)
    text_t = transformer.transform([text_p])
    result = model.predict(text_t)[0]
    
    if result==1:
        prediction = 'It is Spam'
    else:
        prediction = "It's not Spam Enjoy"

    return render_template('predict.html', prediction = prediction)
if __name__ == "__main__":
    app.run(debug=True)
