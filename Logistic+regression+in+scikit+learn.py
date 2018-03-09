
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[4]:


import os


# In[5]:


os.chdir('/Users/JTBras/Downloads')


# In[6]:


admin = pd.read_csv('UCLAAdmissions.csv',sep=',')


# In[5]:


admin.head()


# In[6]:


admin.shape


# In[7]:


admin.info()


# In[8]:


sns.heatmap(admin.corr())


# In[9]:


sns.lmplot(x='gre',y='admit',data=admin,logistic=True,y_jitter=.03,hue='rank')


# In[10]:


sns.lmplot(x='gpa',y='admit',data=admin,logistic=True,y_jitter=.03,hue='rank')


# In[11]:


sns.pairplot(admin,hue='rank')


# In[26]:


admin.head()


# In[24]:


X = [['gre','gpa','rank']]
y = ['admit']


# In[27]:


#X = pd.get_dummies(X)


# In[28]:


from sklearn.model_selection import train_test_split


# In[32]:


from sklearn.linear_model import LogisticRegression
from patsy import dmatrices


# In[33]:


y, X = dmatrices('admit ~ gre + gpa + C(rank)',admin,return_type='dataframe')


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1,random_state=101)


# In[34]:


mod = LogisticRegression(fit_intercept=False,C=1e9)


# In[39]:


md = mod.fit(X_train,y_train)


# In[40]:


mod.coef_


# In[42]:


pred = mod.predict(X_test)


# In[43]:


from sklearn.metrics import classification_report


# In[44]:


print(classification_report(y_test,pred))


# In[1]:


from sklearn.metrics import confusion_matrix


# In[2]:


print(confusion_matrix(y_test,pred))

