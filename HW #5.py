
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


food = pd.read_csv('FoodFat.csv',sep=",")


# In[7]:


food.head()


# In[9]:


food.shape


# In[11]:


food.info()


# In[12]:


sns.pairplot(food,hue='FoodType')


# In[11]:


sns.heatmap(food.corr())


# In[6]:


import statsmodels.formula.api as sm
import statsmodels.api as smf


# In[7]:


mod = sm.ols('FatScore ~ Protein+Sugars+Carbs+Sodium+ServingSize+C(FoodType,Treatment(reference="Burger"))',data=food).fit()


# In[15]:


print(mod.summary())


# In[16]:


plt.hist(mod.resid_pearson)


# In[17]:


plt.scatter(mod.fittedvalues,mod.resid_pearson)
plt.plot([mod.fittedvalues.min(),mod.fittedvalues.max()],[0,0])
plt.xlabel('Fitted Values')
plt.ylabel('Std. resids')


# In[21]:


ncv = 200 
ntest = 12
bias = np.repeat(np.nan,ncv)
rpmse = np.repeat(np.nan,ncv)
for cv in range(0,ncv):
    #split 
    testobs = np.random.choice(food.shape[0],size=ntest,replace=False)
    testset = food.iloc[testobs, :]
    trainset = food.drop(testobs)
    #fit model to training set 
    trainmod = sm.ols('FatScore ~ Protein+Sugars+Carbs+Sodium+ServingSize+C(FoodType,Treatment(reference="Burger"))',data=trainset).fit()
    #predict from test set 
    testpreds = trainmod.predict(testset)
    #rpmse and bias
    bias[cv] = np.mean(testpreds-testset['FatScore'])
    rpmse[cv] = np.sqrt(np.mean((testpreds-testset['FatScore'])**2))


# In[23]:


plt.hist(bias)


# In[24]:


plt.hist(rpmse)


# In[10]:


fig = plt.figure(figsize=(12,8))
fig = smf.graphics.plot_partregress_grid(mod, fig=fig)


# In[11]:


fig = plt.figure(figsize=(12, 8))
fig = smf.graphics.plot_ccpr_grid(mod, fig=fig)


# In[12]:


#in scikit learn


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[7]:


food.info()


# In[19]:


print(food["FoodType"].unique())


# In[17]:


X = food[['Carbs','Sodium','Sugars','Protein','FoodType','ServingSize']]
y = food['FatScore']


# In[18]:


X = pd.get_dummies(X)


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1,random_state=101)


# In[20]:


lm = LinearRegression()


# In[21]:


lm.fit(X_train,y_train)


# In[22]:


print(lm.intercept_)


# In[23]:


coeff_food = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_food


# In[24]:


pred = lm.predict(X_test)


# In[25]:


plt.scatter(y_test,pred)

