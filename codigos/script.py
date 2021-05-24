# By Alberto Caro
# Practica 2
#-----------------------------------------------------------
import numpy as np
import pandas as pd

ds = pd.read_csv('diabetes.csv')
df = pd.DataFrame(ds)
df.columns=['Partos','Glucosa','Presion','Piel','Insulina','IMC','Pedi','Edad','Clase']

df.describe()

from pandas import set_option

set_option('display.width', 100)
set_option('precision', 2)

df.describe()

Clases = df.groupby('Clase').size()

Pearson = df.corr(method='pearson')

import seaborn as sns

sns.heatmap (Pearson,annot = True)

df_aux = df.copy(deep = True)
df_aux[['Glucosa','Presion','Piel','Insulina','IMC']] = 
df_aux[['Glucosa','Presion','Piel','Insulina','IMC']].replace(0,np.NaN)

df_aux

Resumen_NaN = df_aux.isnull().sum()

Distribuciones = df_aux.hist(figsize = (10,7))

df_aux['Glucosa'].fillna(df_aux['Glucosa'].mean(), inplace = True)
df_aux['Presion'].fillna(df_aux['Presion'].mean(), inplace = True)
df_aux['Piel'].fillna(df_aux['Piel'].median(), inplace = True)
df_aux['Insulina'].fillna(df_aux['Insulina'].median(), inplace = True)
df_aux['IMC'].fillna(df_aux['IMC'].median(), inplace = True)


Distribuciones_ajustadas = df_aux.hist(figsize = (10,7))

# Grafiquemos la función de densidad para este DataSet
# DataFrame -> Array Numpy, dejamos fuera columna de Clase

aDF = df_aux.values[:,:-1]
aDF = pd.DataFrame(aDF)
aDF.plot(kind='density')

# La funcion de densidad no se aprecia correctamente pues
# el DataFrame no está normalizado.
# Normalizemos los datos y veamos que sucede.

from sklearn import preprocessing

aDF = df_aux.values[:,:-1]
normal = preprocessing.MinMaxScaler()
aScaler = normal.fit_transform(aDF)

aDF = pd.DataFrame(aScaler)
aDF.plot(kind='density',subplots=True, layout=(3,4), sharex=False)

# Verificar si existen Outliers en dataset original. Dejo fuera la Columna Clase (0/1)
for nDataSet in range(0,8):
 aDF = df_aux.values[nDataSet::,:-1]
 aDF = pd.DataFrame(aDF)
 aDF.plot(kind= 'box',figsize=(20,10))

import seaborn
seaborn.set()
plt.scatter(df_aux['Glucosa'],df_aux['Insulina'])
plt.xlabel('Glucosa')
plt.ylabel('Insulina')

import seaborn
seaborn.set()
plt.scatter(df_aux['Glucosa'],df_aux['Presion'])
plt.xlabel('Glucosa')
plt.ylabel('Presion')


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

#aData = df_aux.values[0::,:-1]
aData = df_aux.values[0::,0] # DataSet Original sin normalización - Descriptor 0 -> Parto
aData

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

iris = pd.read_csv('iris.csv')

iris = pd.DataFrame(iris)

iris.info()

iris.drop('Id',axis=1)

# Como probar si una distribución de datos es normal
from numpy.random import seed, randn
from statsmodels.graphics.gofplots import qqplot
for nDe in range(4): # son 4 los descriptores que hay que analizar
    aData = iris.values[0::,nDe]
    with plt.rc_context():
         plt.rc("figure", figsize=(4,2))
         qqplot(aData,line='s')
plt.show()

setosa = iris[iris.Species == 0]     # DataFrame planta iris->sotosa
versicolor = iris[iris.Species == 1] # DataFrame planta iris->versicolor

setosa.head()

# Como probar si una distribución de datos es normal
from numpy.random import seed, randn
from statsmodels.graphics.gofplots import qqplot
aData = setosa.values[0::,1]
with plt.rc_context():
     plt.rc("figure", figsize=(4,2))
     qqplot(aData,line='s')
plt.show()

iris = iris.drop('Id',axis=1)

setosa = iris[iris.Species == 0]     # DataFrame planta iris->sotosa
versicolor = iris[iris.Species == 1] # DataFrame planta iris->versicolor

versicolor.head()

iris.groupby('Species')['SepalWidthCm'].describe()

from scipy import stats

stats.levene(setosa['SepalWidthCm'],versicolor['SepalWidthCm'])

stats.shapiro(versicolor['SepalWidthCm'])

df_aux.info()

X = iris.values[:,:-1] # Separamos los descriptores
y = iris.values[:,4] # Separamos las Clases

len(y) # Verificamos tamaño dataset

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Vamos a probar que KNN clasifica 100% bien con todos el set de prueba
# despues haremos una prediccion con KNN (k=5)
iris_knn = KNeighborsClassifier(n_neighbors=1).fit(X,y)

y_pred = iris_knn.predict(X)
print(np.all(y_pred==y))

iris_knn.score(X,y)

# Ahora probamos cin KNN (k=5) vecinos
iris_knn = KNeighborsClassifier(n_neighbors = 5).fit(X,y)

iris_knn.score(X,y)
y_pred=iris_knn.predict(X) 

# Ahora probamos cin KNN (k=3) vecinos
iris_knn = KNeighborsClassifier(n_neighbors = 3).fit(X,y)

iris_knn.score(X,y)

# Vamos a utilizar PCA para hacer una reducción a 2D de los descriptores
from sklearn.decomposition import PCA

X_2D = PCA(2).fit_transform(X)

fig,ax = plt.subplots(1,2,figsize=(8,2))
ax[0].scatter(X_2D[:,0], X_2D[:,1], c=y, cmap=plt.cm.Paired)
ax[0].set_title('Data Set Original')
ax[1].scatter(X_2D[:,0], X_2D[:,1], c=y_pred, cmap=plt.cm.Paired)
ax[1].set_title('Data Set Prediccion')
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y)
X_train[0:5], X_test[0:5], y_train[0:5], y_test[0:5]
X_train.shape, X_test.shape

iris_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)
y_pred = iris_knn.predict(X_test)
y_train_pred = iris_knn.predict(X_train)

# n_neighbors=5, Training cross-validation score
iris_knn.score(X_train,y_train)

# n_neighbors=5 Test cross-validation score
iris_knn.score(X_test,y_test)

#DataFrame diabetes original sin normalizar
aData = df_aux

aData.head()

X = df_aux.values[:,:-1] # Separamos los descriptores
y = df_aux.values[:,8] # Separamos las Clases

y[0::10] # mostramos solo las 10 primeras clases...

X_train, X_test, y_train, y_test = train_test_split(X,y) 

X_train.shape, X_test.shape

diabetes_knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_pred = diabetes_knn.predict(X_test)
y_train_pred = diabetes_knn.predict(X_train)

diabetes_knn.score(X_train, y_train)
diabetes_knn.score(X_test, y_test)

print(metrics.classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

X_train, X_test, y_train, y_test = train_test_split(X,y) 

kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)

x_pred = kmeans.predict(X_test)

print(metrics.classification_report(x_pred, y_pred, target_names=['No Diabetes', 'Diabetes']))

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train) 
prediccion = model.predict(X_test)

prediccion[0:5] # los 5 primeros...

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
# Calculo de estadisticos
print(mse(y_test,prediccion)) # Error Medio Cuadratico
r2_score(y_test,prediccion)   # R2 score





