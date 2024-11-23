#https://www.kaggle.com/code/juanagsolano/spam-email-classifier-from-enron-dataset(the original reference)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import math
snow = nltk.stem.SnowballStemmer('english')

def preprocesamiento_words(sentence):       
    #Minisculas
    sentence=sentence.lower() 
    #Remoción de HTML
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)
    #Normalizacion URLs
    cleanr = re.compile('(http|https)://[^\s]*')
    sentence = re.sub(cleanr, 'httpaddr', sentence)
    #Removing Punctuations
    sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)
    #Normalizacion de Direcciones de Correo Electronico
    cleanr = re.compile('[^\s]+@[^\s]+.com')
    sentence = re.sub(cleanr, 'emailaddr', sentence)
    #Normalizacion de Numeros
    cleanr = re.compile('[0-9]+')
    sentence = re.sub(cleanr, 'number', sentence)
    #Normalizacion de $
    cleanr = re.compile('[$]+')
    sentence = re.sub(cleanr, 'dollar', sentence)
    #Remoción de no-palabras (caracteres no alfanumericos)
    cleanr = re.compile('[^a-zA-Z0-9]')
    sentence = re.sub(cleanr, ' ', sentence)
    #Remoción de 'subject'
    cleanr = re.compile('subject')
    sentence = re.sub(cleanr, ' ', sentence)
    #Remoción de stop-words y lematización
    words = [snow.stem(word) for word in sentence.split() if word not in stopwords.words('english')]   # Stemming and removing stopwords
    return words


text_list = []
counter = 0
df=pd.read_csv("enron_spam_data.csv")
df["cls"]=np.where(df["Spam/Ham"]=="spam",1,0)
df["Message"]=df["Subject"]+df["Message"]
df=df.sample(frac=1,random_state=3)
cls=[]
for (sentence,output) in zip(df['Message'],df["cls"]):
    if pd.isnull(sentence):
        continue
    text_list.append(preprocesamiento_words(sentence))
    cls.append(output)
    counter += 1
    if counter==4000:
        break
print("...finished preprocess")

email_process = []
counter = 0
for row in text_list:
    sequ = ''
    for word in row:
        sequ = sequ + ' ' + word
    email_process.append(sequ)
    counter += 1

#tokenization
from sklearn.feature_extraction.text import TfidfVectorizer
def email_tokenization(data,features=2000):
  count_vect = TfidfVectorizer(max_features=features)
  count_matrix = count_vect.fit_transform(data)
  count_array = count_matrix.toarray()
  tokens = pd.DataFrame(data=count_array,columns = count_vect.get_feature_names_out())
  voca = count_vect.vocabulary_
  return (tokens, voca)

tokens, voca = email_tokenization(email_process, features=2000)
tokens["cls"]=cls

#save as csv and vocab
tokens.to_csv("spam_ham.csv")
with open("vocab.txt","wt") as f:
    for idx,item in enumerate(voca.keys()):
        f.write(str(idx)+": "+item+"\n")
f.close()
