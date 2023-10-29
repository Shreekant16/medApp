import pandas as pd
import numpy as np
import random
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import gensim.downloader as api
import gensim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


d1 = pd.read_csv("dataset.csv")
d1 = d1.fillna(0)

symptoms = []
disease = d1.Disease
starting = ["I am having ", "I'm facing ", "I am going from "]

for j in range(0, 4920):
    sent = random.choice(starting)
    for i in range(1, 18):
        if d1[f"Symptom_{i}"][j] != 0:
            sent += d1[f"Symptom_{i}"][j]
            sent += ", "
    symptoms.append(sent)

dataset = pd.DataFrame({"symptoms": symptoms, "disease": disease})

nltk.download("punkt")
nltk.download("stopwords")
stopword = stopwords.words("english")


def process_text(text):
    text = re.sub(r",", "", text)
    stemmer = PorterStemmer()

    clean_text = []
    for word in word_tokenize(text):
        if word not in stopword:
            clean_text.append(stemmer.stem(word))
            # clean_text += " "
    return clean_text


Sentences = []
for i in range(0, 4920):
    process = process_text(dataset.symptoms[i])
    Sentences.append(process)

labels = [i for i in range(0, 41)]
map = {}
for d, i in zip(dataset.disease.unique(), labels):
    map[d] = i

dataset = pd.DataFrame({"sentences": Sentences, "label": dataset.disease})
dataset.label = dataset.label.map(map)

vectorizer = gensim.models.Word2Vec(sentences=dataset.sentences)

vectors = []
vec_size = vectorizer.vector_size
for i in range(0, 4920):
    sent = dataset.sentences[i]
    vec = np.zeros(vec_size, )
    count = 0
    for word in sent:
        if word in vectorizer.wv.index_to_key:
            vec += vectorizer.wv[word]
            count += 1
    final_vec = vec / count
    vectors.append(final_vec)

dataset = pd.DataFrame({"vectors": vectors, "labels": dataset.label})
x = dataset.vectors.to_list()
y = dataset.labels.to_list()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(x_train, y_train)
pickle.dump(classifier, open("trained_model.pkl", "wb"))
y_pred = classifier.predict(x_test)

vectorizer.save("encoder.kv")


