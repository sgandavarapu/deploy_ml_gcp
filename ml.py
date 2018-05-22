# -*- coding: utf-8 -*-

import requests
import os
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
#from urlparse import urlparse
#import schedule
import datetime
import newspaper
import time
import string
import re
from flask import Flask, jsonify, request, render_template
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#from vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from oauth2client.client import GoogleCredentials
from google.cloud import storage

app = Flask(__name__)
app.config['PORT'] = os.getenv("PORT", "8085")
app.config['DEBUG'] = os.getenv("DEBUG", "true") == "true"

@app.route('/score-transaction',methods=['POST'])
def predict():

    #read model pickel file from gcs

    project_id = 'ingo-risk-ml'
    credentials = GoogleCredentials.get_application_default()

    client = storage.Client(project=project_id)
    bucket = client.get_bucket('ingo-risk-ml.appspot.com')

    model_file = bucket.blob('risk_model.pkl')#.read_from()
    model_file.download_to_filename('risk_model.pkl')

    #load the model and predict
    risk_model = joblib.load('risk_model.pkl')

    if request.method == 'POST':
        data = request.form['transaction']

    risk_score = risk_model.predict_proba([data['payee_checks'],data['maker_history']])

    result = {}
    result['trandetid'] = data['trandetid']
    result['risk_score'] = risk_score*10000

    return render_template("result.html",result = result)


@app.route('/mnca-classify-url',methods=['POST'])
def classify_url():
    """Function that takes the url string of a news article and returns the
    classifciation """
    if request.method == 'POST':

        #url = request.form.query
        url = request.form['query']
        #url = request.form.get('url')

        link = newspaper.Article(url)
        link.download()
        link.parse()

        article = {}
        article["title"] = link.title
        article["text"] = link.text

    #return jsonify(classify_article(article))
    result = classify_article(article)
    return render_template("result.html",result = result)


if __name__ == '__main__':
  if app.config['DEBUG']:
      app.run(host='127.0.0.1', port=int(app.config['PORT']), debug=True)
  else:
    app.run(host='0.0.0.0', port=int(app.config['PORT']), debug=False)
