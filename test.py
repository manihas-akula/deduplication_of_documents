

import os
from bs4 import BeautifulSoup
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import dot
import numpy as np
import pandas as pd

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

documents_text = []
documents_name = []

def clean_doc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [ps.stem(word) for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens)
    #print(tokens)
    return tokens


directory = 'dataset'
list_of_files = os.listdir(directory)
for file in list_of_files:
    subfiles = os.listdir(directory+'/'+file)
    for sub in subfiles:
        path = directory+'/'+file+'/'+sub
        with open(path) as f:
            data = f.read()
        f.close()    
        soup = BeautifulSoup(data, 'html.parser')
        #text = str(soup.find_all(text=True))
        text = soup.get_text()
        text = re.sub(r"\s+", " ", text, flags=re.UNICODE)
        text = clean_doc(text.lower())
        documents_text.append(text.strip())
        documents_name.append(path)




tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(documents_text)
feature_names = np.asarray(tfidf.get_feature_names())
table = pd.DataFrame(feature_matrix.toarray(), columns=tfidf.get_feature_names())
print(table)

feature_matrix = np.asarray(table)
print(feature_matrix.shape)
print(feature_names.shape)
documents_name = np.asarray(documents_name)
print(documents_name.shape)

transaction_data = {}
for i in range(len(feature_matrix)):
    print(feature_matrix[i])
    maxElement = np.amax(feature_matrix[i])
    result = np.where(feature_matrix[i] == np.amax(feature_matrix[i]))
    index = result[0]
    transaction_data[feature_names[index[0]]+" "+documents_name[i]] = maxElement

sort = sorted(transaction_data.items(), key=lambda item: item[1], reverse=True)
index = 0
top_words = []
words = []
for word, value in sort:
    arr = word.split(" ")
    print(arr[1]+" "+arr[0]+" "+str(value))
    top_words.append(value*100)
    words.append(arr[0])
    index = index + 1
    if index > 9:
        break

plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Words')
plt.ylabel('Words TFIDF')
#plt.xticks(top_words, words)
plt.plot(words,top_words, 'ro-', color = 'indigo')
plt.legend(['Top 10 words'], loc='upper left')
plt.title('Top 10 Words Graph')
plt.show()

def nearDuplicate(vec1, vec2):
    vector1 = np.asarray(vec1)
    vector2 = np.asarray(vec2)
    return dot(vector1, vector2)/(norm(vector1)*norm(vector2))


for i in range(len(feature_matrix)):
    similarity = 0
    file = ''
    for j in range(len(feature_matrix)):
        if i != j:
            sim = nearDuplicate(feature_matrix[i],feature_matrix[j])
            if sim > similarity:
                similarity = sim
                file = documents_name[i]+" "+documents_name[j]
    if similarity > 0.70:
        print(file+" "+str(similarity))        
            
         



        
