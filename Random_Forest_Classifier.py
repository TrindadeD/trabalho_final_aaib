# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 18:19:21 2022

@author: analo
"""

import pandas as pd
import numpy as np
import os
import tsfel
import seaborn as sns

from random import shuffle, seed
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
sns.set()

seed(3000)

# Root directory
directory = 'C:\\Users\\analo\\Desktop\\projeto final aaib\\newdata'


# Activities -> 3 diferent movements with football ball
# Rejection class (can be mistaken as an activity) -> other
activities = ['dentro_pe', 'fora_pe', 'pentear', 'rejeicao']
codes = ['1', '2', '3', '4']


"""
1) Directory iteration
2) Randomize
3) Split in training and testing sets

"""

# Train set : x_train, y_train
# Test set: x_test , y_test
# x : samples , y : labels
# 3 Classes (3 movements) : '1', '2', '3' 
# Rejection class : '4'

files = []
labels = []

# List of files directories
for i in range(len(activities)):    
    path = os.path.join(directory, activities[i]) # path to each folder
    for filename in os.listdir(path):
        dir = os.path.join(path, filename)
        files += [dir] 
        labels += [codes[i]]

        
# Change the file order inside each folder
dentro_pe = files[0:25]  # dentro_pe folder
shuffle(dentro_pe)

fora_pe = files[25:50] # fora_pe folder
shuffle(fora_pe)

pentear = files[50:75] # pentear folder
shuffle(pentear)

rejeicao = files[75:100] # rejeição folder
shuffle(rejeicao)


# Split files inside each folfer into train (80%) and test (20%) sets
# Total samples per folder = 25
# TRAIN
x_train = dentro_pe[:20] + fora_pe[:20] + pentear[:20] + rejeicao[:20] 
y_train = labels[:20] + labels[25:45] + labels[50:70] + labels[75:95] 

# TEST
x_test = dentro_pe[20:25] + fora_pe[20:25] + pentear[20:25] + rejeicao[20:25] 
y_test = labels[20:25] + labels[45:50] + labels[70:75] + labels[95:100]


"""
FEATURES FILE
"""

# configuration file -> file with selected features
cfg_file = tsfel.load_json('C:\\Users\\analo\\Desktop\\projeto final aaib\\features.json') 

"""
FEATURE EXTRACTION
"""

# 1) Read al the csv files to a pandas dataframe 
# 2) Remove 'timestamp' column
# 3) Extract features from each file


"""
TRAIN
"""

df_1 = pd.read_csv(x_train[0], names=['timestamp', 'accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz'])
df_1 = df_1.drop(['timestamp'], 1)

x_train_features = tsfel.time_series_features_extractor(cfg_file, df_1)


for i in range(1, len(x_train), 1):
    df_1 = pd.read_csv(x_train[i], names=['timestamp', 'accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz'])
    df_1 = df_1.drop(['timestamp'], 1)
    df_features_1 = tsfel.time_series_features_extractor(cfg_file, df_1)
    x_train_features = pd.concat([x_train_features, df_features_1], axis=0, ignore_index=True)
 

    
"""
TEST
"""
    
df_2 = pd.read_csv(x_test[0],  names=['timestamp', 'accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz'])
df_2 = df_2.drop(['timestamp'], 1)
x_test_features = tsfel.time_series_features_extractor(cfg_file, df_2)
for i in range(1, len(x_test), 1):
    df_2 = pd.read_csv(x_test[i], names=['timestamp', 'accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz'])
    df_2 = df_2.drop(['timestamp'], 1)
    df_features_2 = tsfel.time_series_features_extractor(cfg_file, df_2)
    x_test_features = pd.concat([x_test_features, df_features_2], axis=0, ignore_index=True)
    
    
    
"""
FEATURE SELECTION
"""

# Highly correlated features are removed (Pearson Method)
corr_features = tsfel.correlated_features(x_train_features)
x_train_features.drop(corr_features, axis=1, inplace=True)
x_test_features.drop(corr_features, axis=1, inplace=True)

print(corr_features)
# Remove low variance features (with a similar value for all samples)
selector = VarianceThreshold()
X_train = selector.fit_transform(x_train_features)
X_test = selector.transform(x_test_features)
print(X_train.shape)
"""
CLASSIFICATION
"""

# list to array
y_train = np.array(y_train)

classifier = RandomForestClassifier(n_estimators = 100, min_samples_split=10)
#classifier = DecisionTreeClassifier(max_depth=5)

# Train the classifier -> learn
classifier.fit(X_train, y_train.ravel())

# Predict Test Data
y_predict = classifier.predict(X_test)

y_train = np.array(y_train)

#Get the classification report
accuracy = accuracy_score(y_test, y_predict)*100
print('Accuracy: ' + str(accuracy) + '%')
#print(classification_report(y_test, y_predict, target_names = activities))


y_test = np.array(y_test)
# Confusion Matrix
cm = confusion_matrix(y_test, y_predict)
df_cm = pd.DataFrame(cm, index=[i for i in activities], columns = [i for i in activities])
plt.figure()
ax = sns.heatmap(df_cm, cbar = True, cmap="BuGn", annot = True, fmt = 'd')
plt.setp(ax.get_xticklabels(), rotation = 90)
plt.ylabel('True label', fontweight = 'bold', fontsize = 18)
plt.xlabel('Predicted label', fontweight = 'bold', fontsize = 18)
plt.show()


"""
SAVE MODEL
"""

from joblib import dump
dump(classifier, "randomforest.json")

