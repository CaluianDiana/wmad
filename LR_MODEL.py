import re
import warnings

import matplotlib.pyplot as plt
import nltk
import pandas as pd  # data processing
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

df_train = pd.read_csv('./data_set/train2.csv')
print(df_train)
df_train = df_train.fillna(' ')
df_train['total'] = df_train['title'] + ' ' + df_train['author'] + df_train['text']
df_train.total = df_train.total.astype(str)
print(df_train)

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

print(stop_words)


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


preprocess(df_train, 'total')

df_train = df_train[['total', 'label']]

text = df_train['total']
labels = df_train['label']

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2')),
    # ('clf', LogisticRegression(C=1e5)),
    # ('clf', KNeighborsClassifier()),
    # ('clf', MultinomialNB()),
    # ('clf', DecisionTreeClassifier()),
    ('clf', SVC(probability=True)),
    # de cautat
])

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(text.values.astype('str'), labels, test_size=0.2, random_state=7)

pipeline.fit(x_train, y_train)

prediction = pipeline.predict(x_test)

print(classification_report(y_test, prediction, target_names=['Fake', 'Real']))

print("Accuracy of  Classifier: {}%".format(round(accuracy_score(y_test, prediction) * 100, 2)))

print("\nConfusion Matrix of Logistic Regression Classifier:\n")
print(confusion_matrix(y_test, prediction))
cm = confusion_matrix(y_test, prediction)
sns.heatmap(cm, annot=True)
plt.show()

print("\nCLassification Report of Logistic Regression Classifier:\n")
print(classification_report(y_test, prediction, target_names=['Fake', 'Real']))
