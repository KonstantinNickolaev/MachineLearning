import pandas as pd

data = pd.read_csv('titanic.csv', index_col='PassengerId')
passengers = data['Name'].count()

# # 1 task
# print('TASK 1\n')
# sex_mail = data['Sex'].value_counts()
# print(sex_mail)
# print(str(sex_mail[0]) + ' ' + str(sex_mail[1]))
# file_1 = open('1.txt','w')
# file_1.write(str(sex_mail[0]) + ' ' + str(sex_mail[1]))
# file_1.close()
#
# # 2 task
# print('\nTASK 2\n')
# survived = data['Survived'].value_counts()
# print(survived)
# print('Passengers on board: ' + str(passengers) + '\n')
# surv = round(100*float(survived[1])/passengers,2)
# print('Passengers survived in %: ' + str(surv) + '\n')
# print('Passengers died %: ' + str(round(100*float(survived[0])/passengers,2))+ '\n')
# file_2 = open('2.txt','w')
# file_2.write(str(surv))
# file_2.close()
#
# # 3 task
# print('TASK 3\n')
# #pclass = data['Pclass'].values
# #print(pclass)
# pclass = data['Pclass'].value_counts(sort=False)
# print(pclass)
# first_class = pclass[1]
# print('\nFirst class in %: ' + str(100*first_class/passengers) + '\n')
# first_class = round(100*float(first_class)/passengers,2)
# #print(first_class)
# file_3 = open('3.txt','w')
# file_3.write(str(first_class))
# file_3.close()
#
# # 4 task
# print('TASK 4\n')
# age_mean = round(data['Age'].mean(),2)
# print('Age mean: ' + str(age_mean) + '\n')
# age_median = data['Age'].median()
# print('Age median: ' + str(age_median) + '\n')
# file_4 = open('4.txt','w')
# file_4.write(str(age_mean) + ' ' + str(age_median))
# file_4.close()
#
#
# # 5 task
# print('TASK 5\n')
#
# pirson = data.corr()
# print(round(pirson['SibSp']['Parch'],2))
# pirson = round(pirson['SibSp']['Parch'],2)
# file_5 = open('5.txt','w')
# file_5.write(str(pirson))
# file_5.close()
#
#
# # 6 task
print('TASK 6\n')
print('For new commit')

# 7 task
print('TASK TREE\n')
import numpy as np
from sklearn.tree import DecisionTreeClassifier
data = data[['Pclass','Fare','Age','Sex','Survived']]
data['Sex'] = data['Sex'].str.replace('female','1')
data['Sex'] = data['Sex'].str.replace('male','0')
data = data.dropna()
y = np.array(data['Survived'])
data = data[['Pclass','Fare','Age','Sex']]
X = data.to_numpy()
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)
importances = clf.feature_importances_
print(importances)
# np.isnan(X)
file_tree = open('tree.txt','w')
file_tree.write('Fare Sex')
file_tree.close()