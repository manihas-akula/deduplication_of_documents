
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
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
import cv2

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

main = tkinter.Tk()
main.title("Near De-Duplication Web Document") #designing main screen
main.geometry("1300x1200")

global filename
global feature_matrix
global feature_names

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

def upload(): #function to upload tweeter profile
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");

    global documents_text
    global documents_name
    documents_text.clear()
    documents_name.clear()

    list_of_files = os.listdir(filename)
    for file in list_of_files:
        subfiles = os.listdir(filename+'/'+file)
        for sub in subfiles:
            path = 'dataset'+'/'+file+'/'+sub
            print(path)
            with open(path) as f:
                data = f.read()
            f.close()    
            soup = BeautifulSoup(data, 'html.parser')
            #text = str(soup.find_all(text=True))
            textdata = soup.get_text()
            textdata = re.sub(r"\s+", " ", textdata, flags=re.UNICODE)
            textdata = clean_doc(textdata.lower())
            documents_text.append(textdata.strip())
            documents_name.append(path)
    text.insert(END,"Total Web Documents found in dataset is : "+str(len(documents_name))+"\n\n");            

def preprocess():
    global feature_matrix
    global documents_name
    global feature_names
    tfidf = TfidfVectorizer()
    feature_matrix = tfidf.fit_transform(documents_text)
    feature_names = np.asarray(tfidf.get_feature_names())
    table = pd.DataFrame(feature_matrix.toarray(), columns=tfidf.get_feature_names())
    

    feature_matrix = np.asarray(table)
    text.insert(END,"Total words found in all documents are : "+str(len(feature_names))+"\n\n\n")
    documents_name = np.asarray(documents_name)
    text.insert(END,str(table))

def topWords():
    text.delete('1.0', END)
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
        text.insert(END,arr[1]+" "+arr[0]+" "+str(value*100)+"\n")
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
                
def nearDuplicates(vec1, vec2):
    vector1 = np.asarray(vec1)
    vector2 = np.asarray(vec2)
    return dot(vector1, vector2)/(norm(vector1)*norm(vector2))    

def nearDuplicate():
    text.delete('1.0', END)
    for i in range(len(feature_matrix)):
        similarity = 0
        file = ''
        for j in range(len(feature_matrix)):
            if i != j:
                sim = nearDuplicates(feature_matrix[i],feature_matrix[j])
                if sim > similarity:
                    similarity = sim
                    file = documents_name[i]+" ARE NEAR DUPLICATES "+documents_name[j]
        if similarity > 0.70:
            text.insert(END,file+" with Similarity Score : "+str(similarity)+"\n")     
    

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F) #getting Gabor features
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def getFeatures(path):
    img = cv2.imread(path,0)
    img = cv2.resize(img,(128,128))
    filters = build_filters()
    gabor_features = process(img,filters)
    gabor_features = cv2.resize(gabor_features, (128,128))
    im2arr = np.array(gabor_features)
    im2arr = im2arr.reshape(128,128)
    return im2arr    


def nearImage():
    X = []
    Y = []
    duplist = []
    filename = filedialog.askdirectory(initialdir=".")
    for root, dirs, directory in os.walk(filename):
        for i in range(len(directory)):
            name = directory[i]
            print(str(root)+"/"+name)
            im2arr = getFeatures(str(root)+"/"+name)
            X.append(im2arr)
            Y.append(name)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(X.shape)

    XX = np.ndarray(shape=(len(X), 16384), dtype=np.float32)
    for i in range(len(X)):
        temp = X[i].reshape(-1)
        XX[i] = temp

    for i in range(len(XX)):
        similarity = 0
        file = ''
        img_name1 = ''
        img_name2 = ''
        for j in range(len(XX)):
            if i != j:
                sim = nearDuplicates(XX[i],XX[j])
                if sim > similarity and Y[i] not in duplist and Y[j] not in duplist:
                    duplist.append(Y[i])
                    duplist.append(Y[j])
                    similarity = sim
                    file = Y[i]+" ARE NEAR DUPLICATES "+Y[j]
                    img_name1 = Y[i]
                    img_name2 = Y[j]
        if similarity > 0.90:
            text.insert(END,file+" with Similarity Score : "+str(similarity)+"\n")
            im1 = cv2.imread('images/'+img_name1)
            im2 = cv2.imread('images/'+img_name2)
            cv2.imshow(img_name1+' & '+img_name2+' are near duplicates', cv2.resize(im1,(500,500)))
            cv2.imshow(img_name2+' & '+img_name1+' are near duplicates', cv2.resize(im2,(500,500)))
            cv2.waitKey(0)
    

font = ('times', 16, 'bold')
title = Label(main, text='A Strategy for Near-Deduplication Web Documents Considering Both Domain &Size of the Document')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Web Documents Dataset", command=upload)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=330,y=550)
preprocessButton.config(font=font1) 

topButton = Button(main, text="Top 10 Words & Graph", command=topWords)
topButton.place(x=520,y=550)
topButton.config(font=font1) 

nearButton = Button(main, text="Find Near De-Duplication Documents", command=nearDuplicate)
nearButton.place(x=50,y=600)
nearButton.config(font=font1)

imageButton = Button(main, text="Find Near De-Duplication Images", command=nearImage)
imageButton.place(x=350,y=600)
imageButton.config(font=font1)

main.config(bg='sea green')
main.mainloop()
