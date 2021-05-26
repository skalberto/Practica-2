#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[28]:


import numpy as np
from numpy import set_printoptions


# In[6]:


sNames = ['preg','plas','pres','skin','test','mass','pedi','age','class']


# In[10]:


df = pd.read_csv('diabetes.csv',names=sNames)


# In[12]:


# Obtenemos los data como array
data = df.values


# In[37]:


# Dejamos fuera los nombres de las columnas
X = data[1:,0:8]
len(X)


# In[46]:


# Dejamos fuera el nonbre de la columna class
Y= data[1:,8]
len(Y)


# In[41]:


# Preparamos una reescalamiento de los datos entre [0-1] tecnica MinMaxScaler(.)
from sklearn.preprocessing import MinMaxScaler
Escala = MinMaxScaler(feature_range=(0,1))


# In[48]:


# Calculamos la nueva escala
NewEscala = Escala.fit_transform(X)


# In[43]:


set_printoptions(precision=3)


# In[49]:


print(NewEscala)


# In[57]:


# Obtenemos otra escala -> Standarizar -> media = 0 y desv. stand = 1
from sklearn.preprocessing import StandardScaler
Escala_N = StandardScaler().fit(X)


# In[59]:


NewEscala_N = Escala_N.transform(X)


# In[60]:


NewEscala_N


# In[61]:


# Obtenemos otra escala -> Normalizacion
from sklearn.preprocessing import Normalizer
Escala_G = Normalizer().fit(X)


# In[62]:


NewEscala_G = Escala_G.transform(X)


# In[63]:


NewEscala_G


# In[100]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 0.25)


# In[102]:


from sklearn.model_selection import KFold,cross_val_score


# In[104]:


# Aplicamos Clasificador KNN
from sklearn.neighbors import KNeighborsClassifier


# In[105]:


nFolder = 10


# In[107]:


kFold = KFold(n_splits=10)


# In[108]:


model = KNeighborsClassifier()


# In[109]:


res = cross_val_score(model,X,Y,cv=kFold)


# In[110]:


res


# In[111]:


res.mean()


# In[112]:


# Un 73% aprox de certeza en la clasificacion


# In[113]:


# Aplicamos Clasificador Naive Bayes
from sklearn.naive_bayes import GaussianNB


# In[114]:


kFold = KFold(n_splits=10)


# In[115]:


model = GaussianNB()


# In[116]:


res = cross_val_score(model,X,Y,cv=kFold)


# In[117]:


res


# In[118]:


res.mean()


# In[119]:


# un 76 % aprox de certeza en la clasificacion


# In[120]:


# Aplicamos Clasificador Arbol Decision
from sklearn.tree import DecisionTreeClassifier


# In[121]:


kFold = KFold(n_splits=10)


# In[122]:


model = DecisionTreeClassifier()


# In[123]:


res = cross_val_score(model,X,Y,cv=kFold)


# In[124]:


res


# In[125]:


res.mean()


# In[126]:


# un 69 % aprox de certeza en la clasificacion


# In[132]:


#Graficando los rendimientos


# In[147]:


from matplotlib import pyplot as plt


# In[153]:


# Graficando el Rendimiento de los modelos
# KNN -> Vecinos mas cercanos
# NB -> Naive Bayes
# AD -> Arboles de Decision
aModelos = {'KNN':73, 'NB':76,'AD':69}
sNam = list(aModelos.keys())
sVal = list(aModelos.values())
  
fig = plt.figure(figsize = (5,3))
 
# Grafico de Barras..
plt.bar(sNam,sVal,color='blue',width = 0.4)
plt.xlabel("Modelos")
plt.ylabel("Certeza Modelo")
plt.title("Resultado de Rendimiento")
plt.show()


# In[149]:


# El mejor resultado lo obtuvo Naive Bayes con este setdata de diabetes.csv


# In[ ]:




