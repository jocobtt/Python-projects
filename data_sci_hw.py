
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[2]:


import os
import statsmodels.stats.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


os.chdir('/Users/JTBras/Downloads/Linear Regression assignment')


# In[4]:


htwt_1 = pd.read_csv('height_weight1.csv',sep=',')


# In[10]:


htwt_1.head()


# In[11]:


htwt_1.corr()


# In[17]:


sns.lmplot(x='height',y='weight',data=htwt_1)


# # Model that includes intercept:

# In[20]:


mod = smf.ols('weight~height',data=htwt_1).fit()


# In[21]:


mod.summary()


# In[23]:


plt.hist(mod.resid_pearson)


# In[25]:


sm.het_breuschpagan(mod.resid_pearson,mod.model.exog)[1]


# In[33]:


plt.scatter(mod.fittedvalues,mod.resid_pearson)
plt.plot([mod.fittedvalues.min(),mod.fittedvalues.max()],[0,0],color='red')
plt.ylabel('Std. Residuals')
plt.xlabel('Fitted Values')
plt.show()


# In[26]:


mod.rsquared


# # model that doesn't include intercept:

# In[69]:


no_mod = smf.ols(formula='weight~height -1',data=htwt_1).fit()


# In[70]:


no_mod.summary()


# In[42]:


plt.hist(no_mod.resid_pearson)


# In[43]:


sm.het_breuschpagan(no_mod.resid_pearson,no_mod.model.exog)


# In[44]:


plt.scatter(no_mod.fittedvalues,no_mod.resid_pearson)
plt.plot([no_mod.fittedvalues.min(),no_mod.fittedvalues.max()],[0,0],color='red')
plt.ylabel('Std. Residuals')
plt.xlabel('Fitted Values')
plt.show()


# In[45]:


no_mod.rsquared


# We would want to use R squared to see how well our model fits our data. From the R-squared value we can see that the model that doesn't include the intercept fits the data significantly better

# # part 2:

# In[5]:


htwt2 = pd.read_csv('height_weight2.csv',sep=',')


# In[6]:


htwt2.head()


# In[7]:


sns.lmplot(x='height',y='weight',data=htwt2)


# In[7]:


no_mod2 = smf.ols('weight ~ height ',data=htwt2).fit()


# In[8]:


no_mod2.summary()


# In[46]:


plt.hist(no_mod2.resid_pearson)


# In[47]:


sm.het_breuschpagan(no_mod2.resid_pearson,no_mod2.model.exog)


# In[48]:


plt.scatter(no_mod2.fittedvalues,no_mod2.resid_pearson)
plt.plot([no_mod2.fittedvalues.min(),no_mod2.fittedvalues.max()],[0,0],color='red')
plt.ylabel('Std. Residuals')
plt.xlabel('Fitted Values')
plt.show()


# In[56]:


dist = no_mod2.get_influence().cooks_distance


# In[25]:


htwt2.info()
htwt2.weight


# Our model appears to be linear, normal and independent. Equal variance though seems to be an issue but this issue doesn't improve after transforming our variables. Not being equal variant will affect our prediction point and interval and make it harder to get accurate results.

# In[9]:


#cross validate
ncv=100
bias = np.repeat(np.nan,ncv)
rpmse = np.repeat(np.nan,ncv)
coverage = np.repeat(np.nan,ncv)
width = np.repeat(np.nan,ncv)
ntest=10


# In[65]:


for cv in range(ncv):
    #choose obs to split into test set
    test_obs = np.random.choice(htwt2.shape[0],size=ntest,replace=False)
    #split into test and training sets
    test_set = htwt2.iloc[test_obs,:]
    train_set = htwt2.drop(test_obs)
    #use train_set to fit a model
    train_mod = smf.ols(formula='weight~height-1',data=train_set).fit()
    #predict test set

    testpreds = train_mod.get_prediction(test_set)
    #calculate bias
    bias[cv] = np.mean(testpreds.predicted_mean-test_set['weight'])
    #calculate rpmse
    rpmse[cv] = np.sqrt(np.mean((testpreds.predicted_mean-test_set['weight'])**2))
    #calculate coverage
    cis = testpreds.conf_int(obs=True)
    conditions = [cis[:,0]<=test_set['weight'].values,cis[:,1]>=test_set['weight']]
    coverage[cv] = np.mean(np.all(conditions,axis=0))
    width[cv] = np.mean(cis[:,1]-cis[:,0])




# In[63]:


#summary of how our model predicts


# In[66]:


print(np.mean(width))
print(np.mean(bias))
print(np.mean(rpmse))
print(np.mean(coverage))


# From the above values we can conclude that our predictions are on average 15.688 units off of the true value of weight given their height. Also our bias is .03 so we are .03 below the true value on average. Our prediction interval varies about 62.28 pounds with 95% confidence. Also we cover about 94.4% of the true data on average with our model. So we can conclude that our model does a very good job at predicting a given point and producing an informative interval given a person's weight.

# In[11]:


ranges = pd.DataFrame(dict(height=np.linspace(htwt2['height'].min(),htwt2['height'].max(),num=len(htwt2.weight))))


# In[22]:


preds = no_mod2.predict(ranges)


# In[13]:


pred_range = no_mod2.get_prediction(ranges)


# In[14]:


intt = pred_range.conf_int(.05)


# In[20]:


np.mean(intt[:,0]), np.mean(intt[:,1])


# # Problem 3:

# In[28]:


cars = pd.read_csv('car.csv',sep=',')


# In[7]:



cars.Make.unique()


# In[29]:


cars.Make = cars.Make.map({'Ford': 0, 'BMW': 1,'Toyota':2})


# In[9]:


cars.head()


# In[10]:


cars.describe()


# In[67]:


cars.info()


# In[30]:


sns.pairplot(data=cars,hue='Make')


# In[45]:


sns.lmplot(x='Miles',y='Price',data=cars)


# In[65]:


sns.heatmap(cars.corr())


# In[66]:


#fit model#1


# In[76]:


mod_1 = smf.ols('Price ~ Miles+ C(Type)+C(Age)+C(Make)',data=cars).fit()


# In[77]:


mod_1.summary()


# In[3]:


#try model w/ drop age?


# In[78]:


drop_mod = smf.ols('Price ~ Miles + C(Type) +C(Make)',data=cars).fit()


# In[79]:


drop_mod.summary()


# In[9]:


#log transform == best model


# In[31]:


trans_mod = smf.ols('np.log(Price) ~ 1+ Miles +Age+C(Type)+C(Make)',data=cars).fit()


# In[32]:


trans_mod.summary()


# This is our best model because our R^2 value is high- so the model fits the data well. We also can see that all of the values that are being inclucded into our model are significant so we know that this is our best model.

# In[22]:


#predict using best_mod- make=1, 7 years old, 67,000 miles


# In[33]:


pred_df = pd.DataFrame(dict(Age=[7],Make=[1],Type=[3],Miles=[67000]))


# In[34]:


predds = trans_mod.predict(pred_df)


# In[35]:


np.exp(predds)


# We would get a prediction of $14,656.17 for the price of a given a car of make 1 that is 7 years old with 67,000 miles on it.
