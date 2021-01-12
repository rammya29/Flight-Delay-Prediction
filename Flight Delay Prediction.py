#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


# In[2]:


df = pd.read_csv('flights.csv', low_memory=False)
df


# In[3]:


df = df[0:100000]


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.value_counts('DIVERTED')


# In[7]:


sns.jointplot(data=df, x="SCHEDULED_ARRIVAL", y="ARRIVAL_TIME")


# In[8]:


df.corr()


# In[9]:


df[df.columns[1:]].corr()['ARRIVAL_DELAY'][:].sort_values(ascending=False)


# In[10]:


df=df.drop(['YEAR','FLIGHT_NUMBER','AIRLINE','DISTANCE','TAIL_NUMBER','TAXI_OUT', 'SCHEDULED_TIME','DEPARTURE_TIME','WHEELS_OFF','ELAPSED_TIME', 'AIR_TIME','WHEELS_ON','DAY_OF_WEEK','TAXI_IN','CANCELLATION_REASON'], axis=1)


# In[11]:


df.columns


# In[12]:


df.shape


# In[13]:


df.isna().sum()


# In[14]:


df=df.fillna(df.mean())


# In[15]:


df.isna().sum()


# In[19]:


plt.subplots(figsize=(20,15))
sns.heatmap(df.corr(), annot=True, cmap="PuBuGn" ,fmt='g')


# In[20]:


result=[]
for row in df['ARRIVAL_DELAY']:
    if row > 15:
        result.append(1)
    else:
        result.append(0)


# In[21]:


df['result'] = result


# In[22]:


df.head()


# In[23]:


df['result'].value_counts()


# In[24]:


df=df.drop(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ARRIVAL_TIME', 'ARRIVAL_DELAY'],axis=1)
df.columns


# In[25]:


df = df.values
X, y = df[:,:-1], df[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[26]:


scaled_features = StandardScaler().fit_transform(X_train, X_test)


# In[27]:


clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)


# In[28]:


pred = clf.predict_proba(X_test)


# In[29]:


auc_score = roc_auc_score(y_test, pred[:,1])
auc_score


# In[30]:


print('AUC Score of Model is: {} %'.format(round(auc_score*100,2)))


# In[31]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result1 = confusion_matrix(y_test, pred[:,1])
print("Confusion Matrix:")
print(result1)
print('')
result2 = classification_report(y_test, pred[:,1])
print("Classification Report:",)
print (result2)
result3 = accuracy_score(y_test,pred[:,1])
print("Accuracy:",result3)


# In[32]:


print("Accuracy Achieved:",round(metrics.accuracy_score(y_test, pred[:,1])*100,2),'%')
print(' ')
print("Precision Achieved:",round(metrics.precision_score(y_test, pred[:,1])*100,2),'%')
print(' ')
print("Recall Achieved:",round(metrics.recall_score(y_test, pred[:,1])*100,2),'%')


# In[ ]:




