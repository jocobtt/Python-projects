
# coding: utf-8

# In[103]:


import nltk 


# In[104]:


import pandas as pd 


# In[105]:


res = pd.read_excel('/Users/JTBras/Stat 330/training_data.xlsx')


# In[106]:


res['Segment'] = res['Segment'].str.replace('\r','')


# In[107]:


res['Segment'] = res['Segment'].str.replace('\t','')


# In[108]:


res = res[res['Segment'] != res['Segment'].shift(-1)]


# In[109]:


res.head()


# In[110]:


res.groupby('Code').describe()


# In[111]:


#tokenize, stopwords and punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 


# In[112]:


stop_words = set(stopwords.words("english"))


# In[113]:


res['Segment'] = res.Segment.astype(str)


# In[132]:


#res.dropna(axis=0,how='all')


# In[115]:


responses = res['Segment']


# In[116]:


import string


# In[117]:


responses = responses.apply(lambda x:''.join([i for i in x 
                                                  if i not in string.punctuation]))


# In[118]:


res["token"] = responses.fillna("").map(nltk.word_tokenize)


# In[119]:


res['token'] = res['token'].apply(lambda x: [item for item in x if item not in stop_words])


# In[120]:


res['token'] = res['token'].astype(str)


# In[121]:


res.head()


# In[122]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# In[127]:


def train(classifier,X,y):
    X_train, X_test,y_train,y_test = train_test_split(res['token'],res['Code'],random_state=0)
    classifier.fit(X_train,y_train)
    print("Accuracy: %s" % classifier.score(X_test,y_test))
    return classifier


# In[128]:


trial = Pipeline([('vectorizer',TfidfVectorizer()),('classifier',MultinomialNB()),
])


# In[129]:


train(trial,res.token,res.Code)


# In[130]:


trial2 = Pipeline([('vectorizer',TfidfVectorizer()),('classifier',MLPClassifier()),
])


# In[131]:


train(trial2,res.token,res.Code)

