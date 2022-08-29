#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries that we are using in this project

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.preprocessing import LabelEncoder
import re 
import seaborn as sns
from textwrap import wrap
from textblob import TextBlob
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
 

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# ## Load Data

# In[2]:



df=pd.read_csv("UpdatedSpeech.csv",encoding="cp1251")
df.head() #Show the first 5 data values in the dataset


# ## Data Preprocessing

# In[3]:


df.info() #To get the information about dataset like any missing values or what is the data-type of the variables


# In[4]:


df['Disease'].value_counts()


# In[5]:


# To get the unique disease that we are using in our dataset
df['Disease'].unique()


# #### we get to know that there are 11 classes of Diseases on which we will train our model 

# In[6]:


### "No Disease" target values is missplet in some records as "No DIsease" so we replace "No DIsease" with "No Disease"

df['Disease'].replace("No DIsease","No Disease",inplace=True)


# In[7]:


## Replace "No Disease" with "Some Disease"

df['Disease'].replace("No Disease","Some Disease",inplace=True)


# In[8]:


df['Disease'].value_counts()


# In[9]:


plt.figure(figsize=(16,10))
sns.countplot(df['Disease'])


# In[10]:


df['Disease'].unique()


# ## Explorating Data Analysis (EDA)

# ### Word Cloud- Word clouds are the visual representations of the frequency of different words present in a document. It gives importance to the more frequent words which are bigger in size compared to other less frequent words.

# In[11]:


## Important symptoms keywords for Hepatitis E

imp_words = ' '.join(list(df[df['Disease'] == 'Hepatitis E']['Symptoms']))
imp_wc = WordCloud(width = 512,height = 512).generate(imp_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(imp_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[12]:


## Important symptoms keywords for Tuberculosis

imp_words = ' '.join(list(df[df['Disease'] == 'Tuberculosis']['Symptoms']))
imp_wc = WordCloud(width = 512,height = 512).generate(imp_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(imp_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[13]:


## Important symptoms keywords for Diabetes

imp_words = ' '.join(list(df[df['Disease'] == 'Diabetes']['Symptoms']))
imp_wc = WordCloud(width = 512,height = 512).generate(imp_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(imp_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[14]:


## Important symptoms keywords for Common Cold

imp_words = ' '.join(list(df[df['Disease'] == 'Common Cold']['Symptoms']))
imp_wc = WordCloud(width = 512,height = 512).generate(imp_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(imp_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[15]:


## Important symptoms keywords for Hepatitis B

imp_words = ' '.join(list(df[df['Disease'] == 'Hepatitis B']['Symptoms']))
imp_wc = WordCloud(width = 512,height = 512).generate(imp_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(imp_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[16]:


## Important symptoms keywords for Pneumonia

imp_words = ' '.join(list(df[df['Disease'] == 'Pneumonia']['Symptoms']))
imp_wc = WordCloud(width = 512,height = 512).generate(imp_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(imp_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[17]:


## Important symptoms keywords for Typhoid

imp_words = ' '.join(list(df[df['Disease'] == 'Typhoid']['Symptoms']))
imp_wc = WordCloud(width = 512,height = 512).generate(imp_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(imp_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[18]:


## Important symptoms keywords for Migraine

imp_words = ' '.join(list(df[df['Disease'] == 'Migraine']['Symptoms']))
imp_wc = WordCloud(width = 512,height = 512).generate(imp_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(imp_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[19]:


## Important symptoms keywords for Dengue

imp_words = ' '.join(list(df[df['Disease'] == 'Dengue']['Symptoms']))
imp_wc = WordCloud(width = 512,height = 512).generate(imp_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(imp_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[20]:


## Important symptoms keywords for Hypothyroidism

imp_words = ' '.join(list(df[df['Disease'] == 'Hypothyroidism']['Symptoms']))
imp_wc = WordCloud(width = 512,height = 512).generate(imp_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(imp_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# ## Tokenization and normalizing text

# In[21]:


## Spelling Correction on sentences 

def spell_correct(text):
    textBlb = TextBlob(text)
    textCorrected = textBlb.correct()   # Correcting the text
    textCorrected=str(textCorrected)
    return textCorrected


# In[22]:


df['text']=df['Symptoms'].apply(spell_correct)


# In[23]:


df


# In[24]:


waste_words=['ah','uh','hmm','oh','uhh','ahh','ohhh','ohh','shhh','mm','mmm','hmmm','hm','yeah','ya','yup','nope','im','naw','noo','yess']
stopwords = set(stopwords.words('english'))


# In[25]:


stopwords


# In[26]:


## Tokenization, nomalization and removing punctuation, stopwords function

def tokenize(text):
    
    text=re.sub(r'[^\w\s]','', text) #Remove Punctuation
    text=text.lower() #Lower the text

    #tokenize text
    tokens = word_tokenize(text)
    
    tokens=[word for word in tokens if word not in stopwords] #Remove stopwords from the sentences
    tokens=[word for word in tokens if word not in waste_words] #Remove some grabage words from the sentences like Ah,oh etc
     
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    

    # iterate through each token
    clean_tokens =''
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok,pos='v').strip()
        #print(clean_tok)
        clean_tokens=clean_tokens+clean_tok+' '

    return clean_tokens


# In[27]:


# Processing text messages and get the tokenized text 
df['text'] = df['text'].apply(tokenize)
df


# ## Train Test split 

# In[28]:


x_train,x_test,y_train,y_test=train_test_split(df['text'],df['Disease'],test_size=0.2,random_state=12)


# In[29]:


y_train.unique()


# In[30]:


y_test.unique()


# In[31]:


x_train


# ## Feature Extraction from text data 

# In[32]:


tfidf=TfidfVectorizer() #Initialization of Term frequency and Inverse document frequency vectorizer

x_trainTF=tfidf.fit_transform(x_train) #Extract feature from train text data set


# In[33]:


print(x_trainTF) #tfidf vectorizer score for each attribute


# In[34]:


x_trainTF.shape  #316 attributes are extracted from the text train dataset by using these attributes we will develop Models


# In[35]:


#Extract feature from the test dataset

x_testTF = tfidf.transform(x_test)


# In[36]:


x_testTF.shape #Same 316 attributes are extracted for the test dataset


# ## Model Building to predict Disease

# ### Random Forest 

# In[37]:


# training the ensemble classifier 
from sklearn.ensemble import RandomForestClassifier #import library for random forest


#Initialize random forest classifier
randomForestClassifier = RandomForestClassifier(n_estimators=100,criterion='entropy')


#Fit random forest algo on our train dataset
randomForestClassifier.fit(x_trainTF.todense(), y_train)


# Testing our build model against tesing dataset 
y_pred = randomForestClassifier.predict(x_testTF.todense())  #Predict the test dataset by giving test symptoms
print('Confusion Matrix')
print()
print(confusion_matrix(y_test, y_pred))
print()
print('Classification Report')
print()
print(classification_report(y_test, y_pred))

ac_rf=accuracy_score(y_test,y_pred)
print(ac_rf)


# ### Naive Bayes

# In[101]:


# training the Naive Bayes classifier 
from sklearn.naive_bayes import MultinomialNB

NB_clf = MultinomialNB()

NB_clf.fit(x_trainTF, y_train)


# testing against testing set 
y_pred = NB_clf.predict(x_testTF) 
print('Confusion Matrix')
print()
print(confusion_matrix(y_test, y_pred))
print()
print('Classification Report')
print()
print(classification_report(y_test, y_pred))
print()


ac_nb=accuracy_score(y_test,y_pred)
print(ac_nb)


# ### SVM

# In[103]:


# training the SVM classifier 
from sklearn.svm import SVC

svm = SVC()

svm.fit(x_trainTF, y_train)

# testing against testing set 
y_pred = svm.predict(x_testTF) 
print('Confusion Matrix')
print()
print(confusion_matrix(y_test, y_pred))
print()
print('Classification Report')
print()
print(classification_report(y_test, y_pred))
print()


ac_svm=accuracy_score(y_test,y_pred)
print(ac_svm)


# ## Stochastic Gradient Descent Classifier

# In[104]:


from sklearn.linear_model import SGDClassifier

sgd=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)

sgd=sgd.fit(x_trainTF,y_train)

y_pred=sgd.predict(x_testTF)

print('Confusion Matrix')
print()
print(confusion_matrix(y_test, y_pred))
print()
print('Classification Report')
print()
print(classification_report(y_test, y_pred))
print()


ac_sgd=accuracy_score(y_test,y_pred)
print(ac_sgd)


# ## Gradient Boosting Classifier

# In[41]:


from sklearn.ensemble import GradientBoostingClassifier


GBC = GradientBoostingClassifier()

GBC.fit(x_trainTF, y_train)

# testing against testing set 
y_pred = GBC.predict(x_testTF) 
print('Confusion Matrix')
print()
print(confusion_matrix(y_test, y_pred))
print()
print('Classification Report')
print()
print(classification_report(y_test, y_pred))
print()


ac_gb=accuracy_score(y_test,y_pred)
print(ac_gb)


# In[42]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(x_trainTF, y_train)

# testing against testing set 
y_pred = knn.predict(x_testTF) 
print('Confusion Matrix')
print()
print(confusion_matrix(y_test, y_pred))
print()
print('Classification Report')
print()
print(classification_report(y_test, y_pred))
print()


ac_knn=accuracy_score(y_test,y_pred)
print(ac_knn)


# ### Comparison of above model accuracy 

# In[105]:


Accuracy=pd.DataFrame({"Model":["Random Forest","Naive Bayes","SVM","SGD","Gradient Boosting","KNN"],"Accuracy":[ac_rf,ac_nb,ac_svm,ac_sgd,ac_gb,ac_knn]})
Accuracy


# In[106]:


plt.figure(figsize=(15,8))
sns.barplot(x=Accuracy.Model,y=Accuracy.Accuracy)


# ### Best 3 models that we get are SGD,KNN and SVM we will consider one of them after doing hyper-parameter tunning 

# ## Hyper-parameter Tunning, GridsearchCV and Cross validation

# ### SVM Hyperparameter Tunning

# In[107]:


svm.get_params() #Parametes we have in SVM algorithm


# In[108]:


encoder=LabelEncoder()
y_train_int=encoder.fit_transform(y_train)


# In[109]:


param_grid={'C': [0.1, 1, 10 ,100, 1000],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
        }

cv=KFold(n_splits=3,random_state=None,shuffle=False) #Cross validation

gs_svm = GridSearchCV(
        estimator=SVC(kernel='rbf'),
        param_grid=param_grid,
        cv=cv, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

grid_result = gs_svm.fit(x_trainTF, y_train_int)
best_params = grid_result.best_params_
print(best_params)
print(grid_result.best_score_)


# ## Creating Pipeline 
# #### We can automate all of this `fitting`, `transforming`, and `predicting`, by chaining these estimators together into a single estimator object. That single estimator is known as `Pipeline`. To create this pipeline, we just need a list of `(key, value)` pairs, where the key is a string containing what we want to name the step, and the value is the estimator object.

# In[110]:


from sklearn.pipeline import Pipeline
pipeline_svm = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm_clf', SVC(C=100,gamma=0.1,kernel='rbf'))
    
])

# train classifier
pipeline_svm=pipeline_svm.fit(x_train, y_train)

# evaluate all steps on test set
predicted = pipeline_svm.predict(x_test)

print('Confusion Matrix')
print()
print(confusion_matrix(y_test, predicted))
print()
print('Classification Report')
print()
print(classification_report(y_test, predicted))
print()
svm_ac=accuracy_score(y_test,predicted)
print(svm_ac)


# ### KNN best value of K and  hyperparameter Tunnig
# 
# 

# One of the challenges in a k-NN algorithm is finding the best 'k' i.e. the number of neighbors to be used in the majority vote while deciding the class. Generally, it is advisable to test the accuracy of your model for different values of k and then select the best one from them.

# In[49]:


# calculating the accuracy of models with different values of k
mean_acc = np.zeros(15)
for i in range(1,16):
    #Train Model and Predict  
    knn = KNeighborsClassifier(n_neighbors = i).fit(x_trainTF,y_train)
    y_pred= knn.predict(x_testTF)
    mean_acc[i-1] = accuracy_score(y_test, y_pred)

print(mean_acc)

loc = np.arange(1,16,step=1.0)
plt.figure(figsize = (10, 6))
plt.plot(range(1,16), mean_acc)
plt.xticks(loc)
plt.xlabel('Number of Neighbors ')
plt.ylabel('Accuracy')
plt.show()


# ### From above we get to know that at k=4 and k=5 gives best accuracy

# In[50]:


grid_params = { 'n_neighbors' : [2,4,5],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}


cv=KFold(n_splits=3,random_state=None,shuffle=False) #Cross validation

gs_knn = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=grid_params,
        cv=cv, scoring='accuracy', verbose=0, n_jobs=-1)

grid_result = gs_knn.fit(x_trainTF, y_train_int)
best_params = grid_result.best_params_
print(best_params)
print(grid_result.best_score_)


# In[72]:


pipeline_knn = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('knn_clf', KNeighborsClassifier(metric='minkowski',n_neighbors=4,weights='distance'))
    
])

# train classifier
pipeline_knn.fit(x_train, y_train)

# evaluate all steps on test set
predicted = pipeline_knn.predict(x_test)

print('Confusion Matrix')
print()
print(confusion_matrix(y_test, predicted))
print()
print('Classification Report')
print()
print(classification_report(y_test, predicted))
print()
knn_ac=accuracy_score(y_test,predicted)
print(knn_ac)


# ### SGD hyperparameter tunning

# In[58]:


grid_params = { 'loss' : ['hinge','squared_hinge','modified_huber'],
               'penalty' : ['l1','l2','elasticnet'],
               'alpha' : [1,0.5,0.1,0.001,0.0001,0.00001],
              'max_iter':[5,10,20,50,100]}


cv=KFold(n_splits=3,random_state=None,shuffle=False) #Cross validation

gs_sgd = GridSearchCV(
        estimator=SGDClassifier(),
        param_grid=grid_params,
        cv=cv, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

grid_result = gs_sgd.fit(x_trainTF, y_train_int)
best_params = grid_result.best_params_
print(best_params)
print(grid_result.best_score_)


# In[111]:


pipeline_sgd = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('sgd_clf', SGDClassifier(loss='hinge',alpha=0.0001,max_iter=50,penalty='l2'))
    
])

# train classifier
pipeline_sgd.fit(x_train, y_train)

# evaluate all steps on test set
predicted = pipeline_sgd.predict(x_test)

print('Confusion Matrix')
print()
print(confusion_matrix(y_test, predicted))
print()
print('Classification Report')
print()
print(classification_report(y_test, predicted))
print()
sgd_ac=accuracy_score(y_test,predicted)
print(sgd_ac)


# In[112]:


Accuracy_pipeline=pd.DataFrame({"Model":["SVM","SGD","KNN"],"Accuracy":[svm_ac,sgd_ac,knn_ac]})
Accuracy_pipeline


# In[113]:


plt.figure(figsize=(12,8))
sns.barplot(x=Accuracy_pipeline.Model,y=Accuracy_pipeline.Accuracy)


# ### Best accuracy models is SVM on our dataset with highest accuracy of 80% .Hence we take SVM model as our final model for prediction of any new symptoms data

# ## Get Pickle file of our final model SVM 

# In[114]:


import pickle


# Save to file in the specified location
pkl_filename = "C:\\Users\\Rahul\\Desktop\\Disease_Prediction\\disease_prediction.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(pipeline_svm, file)


# In[ ]:





# In[ ]:




