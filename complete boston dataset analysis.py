#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[3]:


boston=load_boston()


# In[4]:


type(boston)


# In[5]:


boston.keys()


# In[6]:


boston.DESCR


# In[7]:


boston.feature_names


# In[8]:


boston.target


# In[9]:


data=boston.data


# In[10]:


data=pd.DataFrame(data=data,columns=boston.feature_names)
data.head()


# In[11]:


data['price']=boston.target
data.head()


# In[12]:


data.describe()


# In[13]:


data.info()


# In[14]:


data.isnull().sum()


# # data visualisation

# In[15]:


sns.pairplot(data)


# # distributing features

# In[16]:


rows=2
cols=7
fig,ax=plt.subplots(nrows=rows,ncols=cols,figsize=(16,4))
col=data.columns
index=0

for i in range(rows):
    for j in range(cols):
        sns.distplot(data[col[index]],ax=ax[i][j])
        #col index means 1st feature
        index=index+1
         
    
plt.tight_layout()


# # correlation mtrix

# In[17]:


corrmat=data.corr()
corrmat


# In[18]:


fig,ax=plt.subplots(figsize=(18,10))
sns.heatmap(corrmat,annot=True,annot_kws={'size': 12})


# In[19]:


corrmat.index.values


# In[20]:


def getcorrelatedfeature(corrdata,threshlod):
    feature=[]
    value=[]
    
    for i,index in enumerate(corrdata.index):
        if abs(corrdata[index])>threshlod:
            feature.append(index)
            value.append(corrdata[index])
            
            df=pd.DataFrame(data=value,index=feature,columns=['Corr Value'])
            return df


# In[21]:


threshold=0.50
corr_value=getcorrelatedfeature(corrmat['price'],threshold)
corr_value


# In[22]:


corr_value.index.values


# In[23]:


correlated_data=[corr_value.index]
correlated_data


# # shuffle and split

# In[24]:


x=boston.data
y=boston.target


# In[25]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# # train model
# 

# In[26]:


model=LinearRegression()
model.fit(X_train,Y_train)


# In[27]:


Y_predict=model.predict(X_test)
#x_predict=model.predict(Y_test)
Y_test


# # checking errors

# In[28]:


from sklearn.metrics import r2_score


# In[29]:


score=r2_score(Y_test,Y_predict)
mae=mean_absolute_error(Y_test,Y_predict)
mse=mean_squared_error(Y_test,Y_predict)

print('r2_score',score)
print('mae', mae)
print('mse', mse)

