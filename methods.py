## Librerías usadas
# Para manipulación de datos 
import numpy as np
import pandas as pd

# Para el manejo de fechas
import calendar
from datetime import datetime

# Para visualización
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
pd.options.display.max_rows = None
pd.options.display.max_columns = None

# Funciones soportadas
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform

# Librerías para modelos
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Funciones Scoring
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Lectura de datos y forma de los mismos
data = pd.read_csv('orders.csv', delimiter=',')
data.shape

# Para empezar el data cleaning, observamos los missing values
data.isnull().sum()

# Obteniendo valores únicos de cada variable
data.nunique()

#order_id duplicados
data_duplicate = data[data.duplicated(['order_id'])]
print(data_duplicate)

#Tenemos 58 datos duplicados por order_id
data_duplicate.count()

#Miremos algunos de los datos repetidos
re1 = data['order_id']==15663723
data_re1 = data[re1]
print(data_re1)

re2 = data['order_id']==15357422
data_re2 = data[re2]
print(data_re2)

#Eliminamos filas con duplicaciones en order_id
order_id_duplicate = data_duplicate['order_id']
data.drop(data[data.duplicated(['order_id'])].index, inplace = True)
data.nunique()

#Eliminamos la columna de order_id, no aporta datos significativos en el análisis.
data = data.drop(["order_id"], axis = 1)

#Observamos la data
data.head()

#Tipo de dato de cada variable
data.dtypes

#Información general del problema en un pastel
labels = 'taken', 'Not Taken'
sizes = [data.taken[data['taken']==1].count(), data.taken[data['taken']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Porción de personas que aceptan y no aceptan pedidos", size = 20)
plt.show()

#Creamos nuevas variables desde la variable 'created_at'
data["date"] = data.created_at.apply(lambda x : x.split()[0].split("T")[0])
data["time"] = data.created_at.apply(lambda x : x.split()[0].split("T")[1])
data["hour"] = data.created_at.apply(lambda x : x.split()[0].split("T")[1].split(":")[0]).astype("int")
data["weekday"] = data.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
data["month"] = data.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])

#Diagrama de barras para día de la semana
sns.countplot(x='weekday', hue = 'taken',data = data)

#Porcentajes de ordenes no tomadas por día
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
per = []

for x in weekdays : 
    day = data.loc[data.weekday==x]
    day_no = day[day['taken']==0]
    per = (len(day_no.index)*100)/len(day.index)
    print(x, per)

#Diagrama de barras para mes del año
sns.countplot(x='month', hue = 'taken',data = data)

#Diagrama de barras para hora del día
sns.countplot(x='hour', hue = 'taken',data = data)

#Relaciones basadas en atributos continuos, para distancia del usuario
sns.boxplot(y='to_user_distance',x = 'taken', hue = 'taken',data = data)

#Relaciones basadas en atributos continuos, para altura del usuario
sns.boxplot(y='to_user_elevation',x = 'taken', hue = 'taken',data = data)

#Relaciones basadas en atributos continuos, para valor ganado
sns.boxplot(y='total_earning',x = 'taken', hue = 'taken',data = data)

# Arreglamos los datos por tipo de datos. 
continuous_vars = ['to_user_distance', 'to_user_elevation', 'total_earning','hour']
cat_vars = ['weekday']
data = data[['taken'] + continuous_vars + cat_vars]
data.head()

# Dividimos datos en entrenamiento y testeo. 20% testeo y 80% entrenamiento
data_train = data.sample(frac=0.8,random_state=200)
data_test = data.drop(data_train.index)
print(len(data_train))
print(len(data_test))

# codificación de las variables categóricas
lst = ['weekday']
remove = list()
for i in lst:
    if (data_train[i].dtype == np.str or data_train[i].dtype == np.object):
        for j in data_train[i].unique():
            data_train[i+'_'+j] = np.where(data_train[i] == j,1,-1)
        remove.append(i)
data_train = data_train.drop(remove, axis=1)
data_train.head()

#Reescalamiento de variables
minVec = data_train[continuous_vars].min().copy()
maxVec = data_train[continuous_vars].max().copy()
data_train[continuous_vars] = (data_train[continuous_vars]-minVec)/(maxVec-minVec)
data_train.head()

# esta función arroja el score para decidir el mejor modelo
def best_model(model):
    print(model.best_score_)    
    print(model.best_params_)
    print(model.best_estimator_)
def get_auc_scores(y_actual, method,method2):
    auc_score = roc_auc_score(y_actual, method); 
    fpr_df, tpr_df, _ = roc_curve(y_actual, method2); 
    return (auc_score, fpr_df, tpr_df)

# Ajustar regresión logística
param_grid = {'C': [0.1,0.5,1,10,50,100], 'max_iter': [250], 'fit_intercept':[True],'intercept_scaling':[1],
              'penalty':['l2'], 'tol':[0.00001,0.0001,0.000001]}
log_primal_Grid = GridSearchCV(LogisticRegression(solver='lbfgs'),param_grid, cv=10, refit=True, verbose=0)
log_primal_Grid.fit(data_train.loc[:, data_train.columns != 'taken'],data_train.taken)
best_model(log_primal_Grid)

# Ajustar regresión logística con un kernel polinomial de grado 2
param_grid = {'C': [0.1,10,50], 'max_iter': [300,500], 'fit_intercept':[True],'intercept_scaling':[1],'penalty':['l2'],
              'tol':[0.0001,0.000001]}
poly2 = PolynomialFeatures(degree=2)
data_train_pol2 = poly2.fit_transform(data_train.loc[:, data_train.columns != 'taken'])
log_pol2_Grid = GridSearchCV(LogisticRegression(solver = 'liblinear'),param_grid, cv=5, refit=True, verbose=0)
log_pol2_Grid.fit(data_train_pol2,data_train.taken)
best_model(log_pol2_Grid)

# Ajustar support vector machine con  RBF (Radial basis function) Kernel
# Este modelo no lo pude correr por falta de poder de cómputo.
param_grid = {'C': [0.5,100,150], 'gamma': [0.1,0.01,0.001],'probability':[True],'kernel': ['rbf']}
SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
SVM_grid.fit(data_train.loc[:, data_train.columns != 'taken'],data_train.taken)
best_model(SVM_grid)

# Regresión logística
log_primal = LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=250, multi_class='warn',n_jobs=None, 
                                penalty='l2', random_state=None, solver='lbfgs',tol=1e-05, verbose=0, warm_start=False)
log_primal.fit(data_train.loc[:, data_train.columns != 'taken'],data_train.taken)

# Regresión logística con kernel polinomial grado 2
poly2 = PolynomialFeatures(degree=2)
data_train_pol2 = poly2.fit_transform(data_train.loc[:, data_train.columns != 'taken'])
log_pol2 = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=300, multi_class='warn', n_jobs=None, 
                              penalty='l2', random_state=None, solver='liblinear',tol=0.0001, verbose=0, warm_start=False)
log_pol2.fit(data_train_pol2,data_train.taken)

#Predicciones para regreción logística
print(classification_report(data_train.taken, log_primal.predict(data_train.loc[:, data_train.columns != 'taken'])))

print(classification_report(data_train.taken,  log_pol2.predict(data_train_pol2)))

#Datos para hacer la gráfica
y = data_train.taken
X = data_train.loc[:, data_train.columns != 'taken']
X_pol2 = data_train_pol2
auc_log_primal, fpr_log_primal, tpr_log_primal = get_auc_scores(y, log_primal.predict(X),log_primal.predict_proba(X)[:,1])
auc_log_pol2, fpr_log_pol2, tpr_log_pol2 = get_auc_scores(y, log_pol2.predict(X_pol2),log_pol2.predict_proba(X_pol2)[:,1])

#Support vector machine con kernel RBF (Radial Basis Function)
param_grid = {'C': [0.5,100,150], 'gamma': [0.1,0.01,0.001],'probability':[True],'kernel': ['rbf']}
SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
SVM_grid.fit(data_train.loc[:, data_train.columns != 'taken'],data_train.taken)
best_model(SVM_grid)

SVM_RBF = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf', max_iter=-1, probability=True, 
              random_state=None, shrinking=True,tol=0.001, verbose=False)
SVM_RBF.fit(data_train.loc[:, data_train.columns != 'taken'],data_train.taken)

#SVM con pol kernel
param_grid = {'C': [0.5,1,10,50,100], 'gamma': [0.1,0.01,0.001],'probability':[True],'kernel': ['poly'],'degree':[2,3] }
SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
SVM_grid.fit(data_train.loc[:, data_train.columns != 'taken'],data_train.taken)
best_model(SVM_grid)

SVM_POL = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,  decision_function_shape='ovr', degree=2, gamma=0.1, kernel='poly',  max_iter=-1,
              probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)
SVM_POL.fit(data_train.loc[:, data_train.columns != 'taken'],data_train.Exited)

#Random Forest
param_grid = {'max_depth': [3, 5, 6, 7, 8], 'max_features': [2,4,6,7,8,9],'n_estimators':[50,100],'min_samples_split': [3, 5, 6, 7]}
RanFor_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, refit=True, verbose=0)
RanFor_grid.fit(data_train.loc[:, data_train.columns != 'taken'],data_train.taken)
best_model(RanFor_grid)

RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=8, max_features=6, max_leaf_nodes=None,min_impurity_decrease=0.0,
                            min_impurity_split=None,min_samples_leaf=1, min_samples_split=3,min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0,warm_start=False)
RF.fit(data_train.loc[:, data_train.columns != 'taken'],data_train.taken)

#Extrem Gradient Boosting
param_grid = {'max_depth': [5,6,7,8], 'gamma': [0.01,0.001,0.001],'min_child_weight':[1,5,10], 'learning_rate': [0.05,0.1, 0.2, 0.3], 'n_estimators':[5,10,20,100]}
xgb_grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, refit=True, verbose=0)
xgb_grid.fit(data_train.loc[:, data_train.columns != 'taken'],data_train.taken)
best_model(xgb_grid)

XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bytree=1, gamma=0.01, learning_rate=0.1, max_delta_step=0,max_depth=7,
                    min_child_weight=5, missing=None, n_estimators=20,n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,reg_alpha=0, 
                    reg_lambda=1, scale_pos_weight=1, seed=None, silent=True, subsample=1)
XGB.fit(data_train.loc[:, data_train.columns != 'taken'],data_train.taken)