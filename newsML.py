import datetime as dt
import json
import pickle
import re
import warnings

import nltk
import numpy as np
import pandas as pd
from flask import Flask, Response
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')

df = pd.read_csv(r'dataset_news.csv', sep='/')

df['text'] = df.text.values.astype(str)
df = df[df['title'].map(lambda x: len(str(x)) >= 42)]
df['clean_text'] = df['text']
df = df.drop_duplicates(subset='title', keep='first')

stop_words = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()


def preprocess(dataframe, column_name):
    for index, row in dataframe.iterrows():
        filter_sentence = ''

        sentence = row[column_name]
        sentence = re.sub(r'[^\w\s]', '', sentence)  # cleaning

        words = nltk.word_tokenize(sentence)  # tokenization

        words = [w for w in words if not w in stop_words]  # stopwords removal

        for word in words:
            filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()

        dataframe.loc[index, column_name] = filter_sentence


preprocess(df, 'clean_text')

with open('Logistic_Regression.pkl', 'rb') as handle:
    model = pickle.load(handle)

today = dt.datetime.today()
df['date'] = df.date.fillna(today)
df['date'] = pd.to_datetime(df['date'], utc=True)
date = pd.to_datetime(df['date']).apply(lambda x: x.date())
df['date'] = date

X = df['clean_text']

predicted = model.predict(X)
prob = model.predict_proba(X)[:, 1] * 100  # probabilitatea sa fie fake este pt 1 si sa fie true este 0
df['probability'] = np.around(prob, decimals=2)

df = df.drop('text', 1)
df = df.drop('clean_text', 1)
df = df.drop('author', 1)

print(df.sample(frac=1))
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv(r"E:\Master E Business\first\output.csv", index=False)

app = Flask(__name__)
CORS(app)


@app.route("/news")
def news():
    return Response(json.dumps(df.to_dict(orient="records"), default=str, indent=4, ensure_ascii=False),
                    mimetype="application/json")


if __name__ == '__main__':
    app.run(debug=True)
