
# coding: utf-8

# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
get_ipython().magic('matplotlib inline')


# In[3]:


water = pd.read_table("https://mheaton.byu.edu/Courses/StatResearch/330inPython/AgWater/AgricultureWater.txt",delim_whitespace=True)


# In[4]:


water.head(10)


# In[32]:


"""can use this to split data if they were together 
water['swc'] = water['swc_cwsi'].apply(lambda swc_cwsi: swc_cwsi.split(' ')[0]) 
water['cwsi'] = water['swc_cwsi'].apply(lambda swc_cwsi: swc_cwsi.split(' ')[1])
"""


# In[33]:


water.plot.scatter('cwsi','swc')


# In[6]:


water.corr()


# In[7]:


sns.lmplot(x='cwsi',y='swc',data=water)


# In[8]:


#correlation heat map
sns.heatmap(water.corr())


# In[13]:


import statsmodels.formula.api as smf


# Linear across the different beta's- make linear by using polynomial regression

# In[14]:


#fit nonlinear regression model-quadratic I()=this is collective 
#still linear in coefficients
mod = smf.ols('swc~cwsi+I(cwsi**2)',data=water).fit()


# In[15]:


print(mod.summary())


# lose interpretation- to interpret you plot on top of it

# In[ ]:


#check assumptions
#check bic and aic to see which model is best - x^2 or x^3 etc. 


# In[8]:


#normality in residuals
sns.distplot(mod.resid_pearson,bins=10)


# In[9]:


#equal variance
plt.scatter(mod.fittedvalues,mod.resid_pearson)
plt.plot([mod.fittedvalues.min(),mod.fittedvalues.max()],[0,0])
plt.xlabel('Fitted Values')
plt.ylabel('Std. resids')


# In[17]:



#smf.het_breuschpagan(mod.resid_pearson,mod.model.exog)


# In[45]:


#predict 
pred_df = pd.DataFrame(dict(cwsi=np.linspace(0,1,num=1000)))


# In[46]:


pred_df.head()


# In[48]:


predic = mod.predict(pred_df)


# In[49]:


plt.scatter(water['cwsi'],water['swc'])
plt.plot(pred_df['cwsi'],predic)
plt.xlabel('cwsi')
plt.ylabel('swc')


# cross validation

# In[19]:


ncv = 100
ntest = 8
bias = np.repeat(np.NaN,ncv)
rmse = np.repeat(np.NaN,ncv)
for cv in range(0,ncv):
    ## Split dataset into training and test
    testobs = np.random.choice(water.shape[0],size=ntest,replace=False)
    testset = water.iloc[testobs,:]
    trainset = water.drop(testobs)
    
    ## Fit model to trainingset
    trainmod = smf.ols('swc~cwsi+I(cwsi**2)',data=trainset).fit()
    
    ## Predict testset
    testpreds = trainmod.predict(testset)
    
    ## Calculate bias & RMSE
    bias[cv] = np.mean(testpreds-testset['swc'])
    rmse[cv] = np.sqrt(np.mean((testpreds-testset['swc'])**2))


# In[20]:


plt.hist(bias)


# In[21]:


plt.hist(rmse)

