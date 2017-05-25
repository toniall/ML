
# coding: utf-8
#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Outros imports
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV,train_test_split,\
                                    StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.externals import joblib


# Definindo a lista de features
features_list = ['poi',
                 'exercised_stock_options',
                 'total_stock_value',
                 'bonus',
                 'salary',
                 'deferred_income',
                 'long_term_incentive',
                 'restricted_stock',
                 'total_payments',
                 'shared_receipt_with_poi',
                 'loan_advances',
                 'expenses',
                 'from_poi_to_this_person',
                 'other',
                 'from_this_person_to_poi',
                 'director_fees',
                 'to_messages',
                 'deferral_payments',
                 'from_messages',
                 'restricted_stock_deferred'
                ]

# Carregando o conjunto de dados
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Removendo outliers
data_dict.pop('TOTAL', None)
data_dict.pop('LOCKHART EUGENE E', None)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', None)

# Salvando o conjunto de dados
my_dataset = data_dict

# Extraindo as features e os labels do conjunto de dados
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Criando os conjuntos de treinamento e testes:
features_train, features_test, labels_train, labels_test = \
		train_test_split(features, labels, test_size=0.3, random_state=42)

# Opcoes para ajustes dos parametros
CRITERION = ['gini','entropy']
SPLITTER = ['best', 'random']
MIN_SAMPLES_SPLIT = [2,4,8,16]
CLASS_WEIGHT = ['balanced', None]
MIN_SAMPLES_LEAF = [1,2,4,8,16]
MAX_DEPTH = [None,1,2,4,8,16]
SCALER = [None, preprocessing.StandardScaler()]
SELECTOR__K = [10, 13, 15, 18, 'all']
REDUCER__N_COMPONENTS = [2, 4, 6, 8, 10]

# Definicoes do Pipeline
## Normalização dos dados, utilizando [StandardScaler]
## Selecao das Features mais importantes, utilizando [SelectKBest]
## Reducao de dimensionalidade dos dados, utilizando [PCA]
## Algoritmo decision Tree

pipe = Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('selector', SelectKBest()),
        ('reducer', PCA(random_state=42)),
        ('classifier', DecisionTreeClassifier())
    ])

#Definicao dos parametros
param_grid = {
    'scaler': SCALER,
    'selector__k': SELECTOR__K,
    'reducer__n_components': REDUCER__N_COMPONENTS,
    'classifier__criterion': CRITERION,
    'classifier__splitter': SPLITTER,
    'classifier__min_samples_split': MIN_SAMPLES_SPLIT,
    'classifier__class_weight': CLASS_WEIGHT,
    'classifier__min_samples_leaf': MIN_SAMPLES_LEAF,
    'classifier__max_depth': MAX_DEPTH,
}

## Validacao cruzada, utilizando [StratifiedShuffleSplit]
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

## Otimizacao, utilizando [GridSearchCV]
grid_search = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)

## Ajustando 
grid = grid_search.fit(features_train,labels_train)

# Melhor classificador
best_clf = grid_search.best_estimator_

# Melhor Score
grid.best_score_

#
grid.score(features_test, labels_test)

# Salvando o melhor classificador
joblib.dump(best_clf, 'best_clf.pkl')

# Salvando melhor classificador, conjunto de dados e a lista de features
dump_classifier_and_data(best_clf, my_dataset, features_list)













