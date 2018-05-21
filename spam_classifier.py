# import packages
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import pickle
import codecs

# import file with stop_words
stopwords = codecs.open('stopwords.txt', 'r', 'utf-8').read().split(',')
my_stop_words=stopwords[0].split('\n')
my_stop_words=[my_stop_words[i].replace('\r','') for i in range(len(my_stop_words))]

#some data preparation/loading
fake=pd.read_csv('fake_account.csv',sep='\t')
fake.columns=['user_id','text']
fake['spam']=1
fake.drop('user_id',axis=1,inplace=True)
legit=pd.read_csv('legitimate_account.csv',sep='\t')
legit=legit[[legit.columns[-1],legit.columns[0]]]
legit.columns=['text','spam']
legit['spam']=0
data=pd.concat([fake,legit])
data=data.dropna()
data.reset_index(inplace=True,drop=True)
X=data['text']
y=data['spam']

#Before feeding network, let's implement tf idf
tf = TfidfVectorizer(stop_words=set(my_stop_words))
X=tf.fit_transform(X)
#Let's divide the data between training and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)


#let's train the mlp : very simple network here but should give better results than NB classifier
clf = MLPClassifier(hidden_layer_sizes = (6,2), activation = 'tanh',learning_rate = 'adaptive', max_iter = 200)
clf.fit(X_train,y_train)
#Now check results
predictions=clf.predict(X_test)
print(confusion_matrix(predictions,y_test))


#If results are good, i.e accuracy on positives is above 60% and accuracy on negatives above 95%:
#->we save the model

joblib.dump(clf, 'MLP_spam.pkl') 
with open("tokenizer_MLP.pickle", "wb") as handle: 
	pickle.dump(tf, handle, protocol = pickle.HIGHEST_PROTOCOL)"""
