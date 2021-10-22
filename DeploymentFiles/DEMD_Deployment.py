#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import numpy as np
import pickle


# In[5]:


df=pd.read_excel('Concrete_Data.xls')


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.corr()


# In[12]:


(df.corr() > 0.6).sum()


# In[15]:


df.columns = ['Cement','Blast Furnace','Fly Ash','Water','Superplasticizer','Coarseaggregate','Fineaggregate','Age','Concrete_compressive_strength']


# In[16]:


from sklearn.model_selection import train_test_split as TTS

train , valid = TTS(df, train_size = 0.8)


# In[57]:


train[predictors]


# In[30]:


predictors=df.columns.to_list()
predictors.remove('Concrete_compressive_strength')


# # Linear Regression

# In[31]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[32]:


lr.fit(train[predictors],train["Concrete_compressive_strength"])


# In[34]:


predictions = lr.predict(valid[predictors])


# In[35]:


from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(valid['Concrete_compressive_strength'], predictions)
RMSE = np.sqrt(MSE)
RMSE_percent = (RMSE/np.mean(valid['Concrete_compressive_strength'],))*100


# In[36]:


RMSE_percent


# # Decision Tree Regression

# In[37]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(min_samples_split = 2)
dt.fit(train[predictors],train["Concrete_compressive_strength"])


# In[38]:


predictions_dt = dt.predict(valid[predictors])


# In[39]:


MSE = mean_squared_error(valid['Concrete_compressive_strength'], predictions_dt)
RMSE = np.sqrt(MSE)
RMSE_percent = (RMSE/np.mean(valid['Concrete_compressive_strength'],))*100


# In[40]:


RMSE_percent


# # Random Forest Regression

# In[63]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(train[predictors],train["Concrete_compressive_strength"])


# In[64]:


predictions_rf = rf.predict(valid[predictors])


# In[65]:


MSE = mean_squared_error(valid['Concrete_compressive_strength'], predictions_rf)
RMSE = np.sqrt(MSE)
RMSE_percent = (RMSE/np.mean(valid['Concrete_compressive_strength'],))*100


# In[66]:


RMSE_percent


# # XGboost Regression

# In[45]:


from xgboost import XGBRegressor


# In[50]:


XGBC = XGBRegressor(objective ='reg:squarederror')
model_no_tune = XGBC.fit(train[predictors], train['Concrete_compressive_strength'])
y_pred = model_no_tune.predict(valid[predictors])
MSE = mean_squared_error(valid['Concrete_compressive_strength'], y_pred)
RMSE = np.sqrt(MSE)
RMSE_percent = (RMSE/np.mean(valid['Concrete_compressive_strength']))*100


# In[51]:


RMSE_percent


# In[68]:


# KNN Regression


# In[52]:


from sklearn import neighbors


# In[54]:


rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(train[predictors], train['Concrete_compressive_strength'])  #fit the model
    pred=model.predict(valid[predictors]) #make prediction on test set
    error = np.sqrt(mean_squared_error(valid['Concrete_compressive_strength'],pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


# In[55]:


min(rmse_val)


# In[56]:


model = neighbors.KNeighborsRegressor(n_neighbors = 2)
model.fit(train[predictors], train['Concrete_compressive_strength'])  #fit the model
pred=model.predict(valid[predictors]) #make prediction on test set
error = np.sqrt(mean_squared_error(valid['Concrete_compressive_strength'],pred))
error


# In[ ]:





# In[72]:


knn = neighbors.KNeighborsRegressor(n_neighbors = 2)
knn.fit(train[predictors], train['Concrete_compressive_strength'])


# In[74]:


valid[predictors]


# In[84]:


#Saving model to disk
pickle.dump(knn,open('modelknn.pkl','wb'))

#Loading model to compare the results
model=pickle.load(open('modelknn.pkl','rb'))
print(model.predict([valid[predictors].loc[119].to_list()]))


# In[ ]:




