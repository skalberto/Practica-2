#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


import pandas as pd


# In[ ]:


ds = pd.read_csv('diabetes.csv')


# In[ ]:


df = pd.DataFrame(ds)


# In[ ]:


df.columns=['Partos','Glucosa','Presion','Piel','Insulina','IMC','Pedi','Edad','Clase']


# In[ ]:


df


# In[ ]:


df.describe()


# In[ ]:


from pandas import set_option


# In[ ]:


set_option('display.width', 100)
set_option('precision', 2)


# In[ ]:


df.describe()


# In[ ]:


Clases = df.groupby('Clase').size()


# In[ ]:


Clases


# In[ ]:


Pearson = df.corr(method='pearson')


# In[ ]:


Pearson


# In[ ]:


import seaborn as sns


# In[ ]:


sns.heatmap (Pearson,annot = True)


# In[ ]:


df_aux = df.copy(deep = True)
df_aux[['Glucosa','Presion','Piel','Insulina','IMC']] = 
df_aux[['Glucosa','Presion','Piel','Insulina','IMC']].replace(0,np.NaN)


# In[ ]:


df_aux


# In[ ]:


Resumen_NaN = df_aux.isnull().sum()


# In[ ]:


Distribuciones = df_aux.hist(figsize = (10,7))


# In[ ]:


df_aux['Glucosa'].fillna(df_aux['Glucosa'].mean(), inplace = True)
df_aux['Presion'].fillna(df_aux['Presion'].mean(), inplace = True)
df_aux['Piel'].fillna(df_aux['Piel'].median(), inplace = True)
df_aux['Insulina'].fillna(df_aux['Insulina'].median(), inplace = True)
df_aux['IMC'].fillna(df_aux['IMC'].median(), inplace = True)


# In[ ]:


Distribuciones_ajustadas = df_aux.hist(figsize = (10,7))


# In[ ]:


df_aux


# In[ ]:


# Grafiquemos la función de densidad para este DataSet
# DataFrame -> Array Numpy, dejamos fuera columna de Clase
aDF = df_aux.values[:,:-1]


# In[ ]:


aDF


# In[ ]:


aDF = pd.DataFrame(aDF)
aDF.plot(kind='density')


# In[ ]:


# La funcion de densidad no se aprecia correctamente pues
# el DataFrame no está normalizado.
# Normalizemos los datos y veamos que sucede.

from sklearn import preprocessing

aDF = df_aux.values[:,:-1]
normal = preprocessing.MinMaxScaler()
aScaler = normal.fit_transform(aDF)


# In[ ]:


# Ahora el data frame está normalizado entre (0,1)
aScaler


# In[ ]:


aDF = pd.DataFrame(aScaler)
aDF.plot(kind='density',subplots=True, layout=(3,4), sharex=False)


# In[ ]:


# Verificar si existen Outliers en dataset original. Dejo fuera la Columna Clase (0/1)
for nDataSet in range(0,8):
 aDF = df_aux.values[nDataSet::,:-1]
 aDF = pd.DataFrame(aDF)
 aDF.plot(kind= 'box',figsize=(20,10))


# In[ ]:


import seaborn
seaborn.set()
plt.scatter(df_aux['Glucosa'],df_aux['Insulina'])
plt.xlabel('Glucosa')
plt.ylabel('Insulina')


# In[ ]:


import seaborn
seaborn.set()
plt.scatter(df_aux['Glucosa'],df_aux['Presion'])
plt.xlabel('Glucosa')
plt.ylabel('Presion')


# In[ ]:


# Como probar si una distribución de datos es normal
from numpy.random import seed, randn
from statsmodels.graphics.gofplots import qqplot
# Configuro la semilla aleatoria
seed(1993)
# Genero 100 muestras
data = randn(100)
# Represento el Q-Q plot
qqplot(data,line='s')
plt.show()


# In[ ]:


#aData = df_aux.values[0::,:-1]
aData = df_aux.values[0::,0] # DataSet Original sin normalización - Descriptor 0 -> Parto
aData


# In[239]:


# Como probar si una distribución de datos es normal
from numpy.random import seed, randn
from statsmodels.graphics.gofplots import qqplot
# Configuro la semilla aleatoria
seed(1993)
for nDe in range(8): # son 8 los descriptores que hay que analizar
    aDF = pd.DataFrame(aScaler) # DataSet Normalizado
    aData = aDF.values[0::,nDe]
    with plt.rc_context():
         plt.rc("figure", figsize=(4,2))
         qqplot(aData,line='s')
plt.show()


# In[210]:


# Prueba de Shapiro-Wilk
from numpy.random import seed, randn
from scipy.stats import shapiro

seed(1993)
for nDe in range(8): # son 8 los descriptores que hay que analizar
    aDF = pd.DataFrame(aScaler) 
    aData = aDF.values[0::,nDe]
    # Prueba de Shapiro-Wilk
    stat, p = shapiro(aData)
    print('Estadisticos = %.3f, p = %.3f' % (stat,p))
    # Interpretación
    alpha = 0.05
    if p > alpha:
       print('La muestra parece Gaussiana (no se rechaza la hipótesis nula H0')
    else:
       print('La muestra no parece Gaussiana(se rechaza la hipótesis nula H0')


# In[212]:


# Prueba de D' Agostino K-Squared
from numpy.random import seed, randn
from scipy.stats import normaltest

for nDe in range(8): # son 8 los descriptores que hay que analizar
    aData = df_aux.values[0::,nDe]
    # Prueba de D' Agostino K-Squared 
    stat, p = normaltest(aData) 
    print('Estadisticos = %.3f, p = %.3f' % (stat,p))
    # Interpretación
    alpha = 0.05
    if p > alpha:
       print('La muestra parece Gaussiana (no se rechaza la hipótesis nula H0')
    else:
       print('La muestra no parece Gaussiana (se rechaza la hipótesis nula H0')


# In[215]:


aDF_Aux


# In[216]:


df_aux


# In[217]:


df_aux


# In[231]:


iris = pd.read_csv('iris.csv')


# In[232]:


iris = pd.DataFrame(iris)


# In[235]:


iris.info()


# In[238]:


iris.drop('Id',axis=1)


# In[240]:


# Como probar si una distribución de datos es normal
from numpy.random import seed, randn
from statsmodels.graphics.gofplots import qqplot
for nDe in range(4): # son 4 los descriptores que hay que analizar
    aData = iris.values[0::,nDe]
    with plt.rc_context():
         plt.rc("figure", figsize=(4,2))
         qqplot(aData,line='s')
plt.show()


# In[247]:


setosa = iris[iris.Species == 0]     # DataFrame planta iris->sotosa
versicolor = iris[iris.Species == 1] # DataFrame planta iris->versicolor


# In[256]:


setosa.head()


# In[246]:


# Como probar si una distribución de datos es normal
from numpy.random import seed, randn
from statsmodels.graphics.gofplots import qqplot
aData = setosa.values[0::,1]
with plt.rc_context():
     plt.rc("figure", figsize=(4,2))
     qqplot(aData,line='s')
plt.show()


# In[257]:


iris = iris.drop('Id',axis=1)


# In[258]:


iris


# In[259]:


setosa = iris[iris.Species == 0]     # DataFrame planta iris->sotosa
versicolor = iris[iris.Species == 1] # DataFrame planta iris->versicolor


# In[265]:


versicolor.head()


# In[267]:


iris.groupby('Species')['SepalWidthCm'].describe()


# In[269]:


versicolor


# In[270]:


from scipy import stats


# In[271]:


stats.levene(setosa['SepalWidthCm'],versicolor['SepalWidthCm'])


# In[274]:


stats.shapiro(versicolor['SepalWidthCm'])


# In[282]:


df_aux.info()


# In[283]:


iris


# In[286]:


X = iris.values[:,:-1] # Separamos los descriptores


# In[288]:


y = iris.values[:,4] # Separamos las Clases


# In[290]:


len(y) # Verificamos tamaño dataset


# In[292]:


from sklearn.neighbors import KNeighborsClassifier


# In[293]:


from sklearn import metrics


# In[296]:


# Vamos a probar que KNN clasifica 100% bien con todos el set de prueba
# despues haremos una prediccion con KNN (k=5)
iris_knn = KNeighborsClassifier(n_neighbors=1).fit(X,y)


# In[297]:


y_pred = iris_knn.predict(X)


# In[298]:


print(np.all(y_pred==y))


# In[299]:


iris_knn.score(X,y)


# In[301]:


# Ahora probamos cin KNN (k=5) vecinos
iris_knn = KNeighborsClassifier(n_neighbors = 5).fit(X,y)


# In[302]:


iris_knn


# In[303]:


iris_knn.score(X,y)


# In[304]:


y_pred=iris_knn.predict(X) 


# In[305]:


y_pred


# In[320]:


# Ahora probamos cin KNN (k=3) vecinos
iris_knn = KNeighborsClassifier(n_neighbors = 3).fit(X,y)


# In[321]:


iris_knn.score(X,y)


# In[322]:


# Vamos a utilizar PCA para hacer una reducción a 2D de los descriptores
from sklearn.decomposition import PCA


# In[323]:


X_2D = PCA(2).fit_transform(X)


# In[327]:


fig,ax = plt.subplots(1,2,figsize=(8,2))
ax[0].scatter(X_2D[:,0], X_2D[:,1], c=y, cmap=plt.cm.Paired)
ax[0].set_title('Data Set Original')
ax[1].scatter(X_2D[:,0], X_2D[:,1], c=y_pred, cmap=plt.cm.Paired)
ax[1].set_title('Data Set Prediccion')
plt.show()


# In[332]:


from sklearn.model_selection import train_test_split


# In[333]:


X_train, X_test, y_train, y_test = train_test_split(X,y)


# In[336]:


X_train[0:5], X_test[0:5], y_train[0:5], y_test[0:5]


# In[337]:


X_train.shape, X_test.shape


# In[338]:


iris_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)
y_pred = iris_knn.predict(X_test)
y_train_pred = iris_knn.predict(X_train)


# In[342]:


# n_neighbors=5, Training cross-validation score
iris_knn.score(X_train,y_train)


# In[346]:


# n_neighbors=5 Test cross-validation score
iris_knn.score(X_test,y_test)


# In[349]:


#DataFrame diabetes original sin normalizar
aData = df_aux


# In[356]:


aData.head()


# In[351]:


X = df_aux.values[:,:-1] # Separamos los descriptores


# In[352]:


y = df_aux.values[:,8] # Separamos las Clases


# In[359]:


X


# In[355]:


y[0::10] # mostramos solo las 10 primeras clases...


# In[360]:


X_train, X_test, y_train, y_test = train_test_split(X,y) 


# In[361]:


X_train.shape, X_test.shape


# In[362]:


diabetes_knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_pred = diabetes_knn.predict(X_test)
y_train_pred = diabetes_knn.predict(X_train)


# In[365]:


y_pred


# In[366]:


diabetes_knn.score(X_train, y_train)


# In[367]:


diabetes_knn.score(X_test, y_test)


# In[370]:


print(metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))


# In[374]:


X_train, X_test, y_train, y_test = train_test_split(X,y) 


# In[375]:


kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)


# In[378]:


x_pred = kmeans.predict(X_test)
#y_pred = kmeans.predict(X_test)


# In[379]:


x_pred


# In[380]:


print(metrics.classification_report(x_pred, y_pred, target_names=['No Diabetes', 'Diabetes']))


# In[381]:


from sklearn.linear_model import LinearRegression


# In[397]:


model = LinearRegression()
model.fit(X_train,y_train) 


# In[403]:


prediccion = model.predict(X_test)


# In[399]:


prediccion[0:5] # los 5 primeros...


# In[386]:


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# In[407]:


# Calculo de estadisticos
print(mse(y_test,prediccion)) # Error Medio Cuadratico
r2_score(y_test,prediccion)   # R2 score


# In[ ]:




