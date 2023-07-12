#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[4]:


df = pd.read_csv('KNN_Project_Data')


# **Check the head of the dataframe.**

# In[5]:


df.head()


# In[3]:





# In[6]:


sns.pairplot(df, hue = 'TARGET CLASS')


# In[4]:





# In[7]:


from sklearn.preprocessing import StandardScaler


# ** Create a StandardScaler() object called scaler.**

# In[8]:


scaler = StandardScaler()


# ** Fit scaler to the features.**

# In[11]:


scaler.fit(X = df.drop('TARGET CLASS', axis = 1 ))


# In[7]:





# **Use the .transform() method to transform the features to a scaled version.**

# In[13]:


X = scaler.transform(X = df.drop('TARGET CLASS', axis = 1 ))


# **Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[14]:


tdf = pd.DataFrame(X, columns=df.columns[:-1])
tdf.head()


# In[9]:





# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# In[17]:


from sklearn.neighbors import KNeighborsClassifier


# **Create a KNN model instance with n_neighbors=1**

# In[18]:


MyKNN = KNeighborsClassifier(n_neighbors= 1)


# **Fit this KNN model to the training data.**

# In[19]:


MyKNN.fit(X_train, y_train)


# In[14]:





# **Use the predict method to predict values using your KNN model and X_test.**

# In[21]:


y_predict = MyKNN.predict(X_test)


# ** Create a confusion matrix and classification report.**

# In[23]:


from sklearn.metrics import confusion_matrix, classification_report


# In[26]:


print(classification_report(y_test,y_predict))


# In[17]:





# In[28]:


print(confusion_matrix(y_test, y_predict))


# In[27]:





# In[29]:


err_rates = []
for idx in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = idx)
    knn.fit(X_train, y_train)
    pred_idx = knn.predict(X_test)
    err_rates.append(np.mean(y_test != pred_idx))


# In[19]:


#you will get diff 


# In[33]:


plt.style.use('ggplot')
plt.subplots(figsize = (10,6))
plt.plot(range(1,40), err_rates, color = 'blue', marker = 'o', markerfacecolor = 'red')
plt.xlabel('K-value')
plt.ylabel('Error Rate')
plt.title('Error Rate vs K-value')


# In[20]:


df_err = pd.DataFrame(data=error_rate, columns=['Error'])
df_err.plot()


# In[34]:


MyKNN = KNeighborsClassifier(n_neighbors = 31)
MyKNN.fit(X_train,y_train)
y_predict = MyKNN.predict(X_test)

print('WITH K=31')
print('')
print(confusion_matrix(y_test,y_predict))
print('')
print(classification_report(y_test,y_predict))


# In[21]:


knn = KNeighborsClassifier(n_neighbors=24)
knn.fit(X, y)
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

