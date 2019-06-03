import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt #visualization
import seaborn as sns #modern visualization

#data visualization
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
matches=pd.read_csv('finaldata.csv')
matches.shape
matches["city"].fillna( method ='ffill', inplace = True)

matches["winner"].fillna( method ='ffill', inplace = True)

matches["player_of_match"].fillna( method ='ffill', inplace = True)
matches["umpire1"].fillna( method ='ffill', inplace = True)
matches["umpire2"].fillna( method ='ffill', inplace = True)
#matches=matches.drop('umpire3',1)

#splitting data n preprocessing
le = preprocessing.LabelEncoder()
X = matches.iloc[:, [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16]].values  #Feature Matrix
Y = matches["winner"]          #Target Variable
#matches.head()
le = preprocessing.LabelEncoder()
for i in range(0,15):
    X[:,i]=le.fit_transform(X[:,i])
Y=le.fit_transform(Y)

#visualization
matches.head()
matches.describe()
matches.info()
matches['season'].unique()
sns.countplot(x='season',data=matches)
plt.show()

data=matches.winner.value_counts()
sns.barplot(y=data.index,x=data,orient='h')


top_players=matches.player_of_match.value_counts()[:10]
fig, ax=plt.subplots()
ax.set_ylabel("Count")
ax.set_title("Top players of the match")
sns.barplot(x=top_players.index,y=top_players,orient='v')
toss_win=matches['toss_winner']==matches['winner']
toss_win.groupby(toss_win).size()
sns.countplot(toss_win)
fig, ax = plt.subplots()
ax.set_title("Winning by Runs - Team Performance")

sns.boxplot(y = 'winner', x = 'win_by_runs', data=matches[matches['win_by_runs']>0], orient = 'h');
plt.show()
fig, ax = plt.subplots()
ax.set_title("Winning by Wickets - Team Performance")
#top_players.plot.bar()
sns.boxplot(y = 'winner', x = 'win_by_wickets', data=matches[matches['win_by_wickets']>0], orient = 'h');
plt.show()

sns.countplot(x='venue',data=matches)
plt.xticks(rotation='vertical')
plt.show()

toss=matches.toss_decision.value_counts()
x=toss.field+toss.bat
x_field=(toss.field/x)*100
x_bat=(toss.bat/x)*100
values=np.array([x_field,x_bat])
labels=np.array(toss.index)
plt.pie(values,labels=labels,colors=['lightblue','yellow'],startangle=90,autopct='%1.1f%%')
plt.title("Toss decision")
plt.show()


sns.countplot(x='season',hue='toss_decision',data=matches)
plt.xticks(rotation='vertical')
plt.show()


num_win=(matches.win_by_wickets>0).sum()
num_loss=(matches.win_by_wickets==0).sum()
total=float(num_win+num_loss)
values=np.array([(num_win/total)*100,(num_loss/total)*100])
labels=np.array(['WIN','LOSS'])
plt.pie(values,labels=labels,colors=['lightblue','yellow'],startangle=90,autopct='%1.1f%%')
plt.title("Winning on batting second")
plt.show()


#prediction random forest
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X_train, y_train)
y_pred_rf = classifier_rf.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
cm_rf = confusion_matrix(y_test, y_pred_rf)
accuracy_rf=accuracy_score(y_test,y_pred_rf)
print("Accuracy on 2007-2018=%0.2f" %accuracy_rf)


#random forest on ipl2019
ipl_2019=pd.read_csv('IPL_2019.csv')
X_2019 = matches.iloc[:, [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16]].values  #Feature Matrix
Y_2019 = matches["winner"]
le_2019 = preprocessing.LabelEncoder()
for j in range(0,15):
    X_2019[:,j]=le_2019.fit_transform(X_2019[:,j])
Y_2019=le_2019.fit_transform(Y_2019)
y_pred_2019_rf = classifier_rf.predict(X_2019)

accuracy_2019_rf=accuracy_score(Y_2019, y_pred_2019_rf)
print("--------------------------------------------------------------")
print("Accuracy on latest dataset of 2019=%0.3f" %(accuracy_2019_rf*100))

from sklearn.model_selection import cross_val_score,ShuffleSplit
cv_rf = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
acc_2019_rf=cross_val_score(classifier_rf, X_2019, Y_2019, cv=cv_rf)
acc_2019_rf=acc_2019_rf*100
x_line=[1,2,3,4,5,6,7,8,9,10]
plt.plot(x_line,acc_2019_rf,'-o')
plt.xlabel("Number of splits")
plt.ylabel("Accuracy")
plt.title("Accuracy using Cross-validation in Random Forest")
plt.show()
print("-------------------------------------------------")
print("Accuracy on 2019=%0.2f" %(np.mean(acc_2019_rf)))


precision_2019=precision_score(Y_2019, y_pred_2019_rf,average='weighted')
print("Precision on latest dataset of 2019=%0.3f" %(precision_2019))
recall_2019=recall_score(Y_2019, y_pred_2019_rf,average='weighted')
print("Recall on latest dataset of 2019=%0.3f" %(recall_2019))
fscore_2019=f1_score(Y_2019, y_pred_2019_rf,average='weighted')
print("F1score on latest dataset of 2019=%0.3f" %(fscore_2019))

precision_2019=precision_score(Y_2019, y_pred_2019_rf,average='macro')
print("Precision on latest dataset of 2019=%0.3f" %(precision_2019))
recall_2019=recall_score(Y_2019, y_pred_2019_rf,average='macro')
print("Recall on latest dataset of 2019=%0.3f" %(recall_2019))
fscore_2019=f1_score(Y_2019, y_pred_2019_rf,average='macro')
print("F1score on latest dataset of 2019=%0.3f" %(fscore_2019))

precision_2019=precision_score(Y_2019, y_pred_2019_rf,average='micro')
print("Precision on latest dataset of 2019=%0.3f" %(precision_2019))
recall_2019=recall_score(Y_2019, y_pred_2019_rf,average='micro')
print("Recall on latest dataset of 2019=%0.3f" %(recall_2019))
fscore_2019=f1_score(Y_2019, y_pred_2019_rf,average='micro')
print("F1score on latest dataset of 2019=%0.3f" %(fscore_2019))

#support vector machines
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'linear', random_state = 0)
classifier_svm.fit(X_train, y_train)

# Predicting the Test set results
y_pred_svm = classifier_svm.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
accuracy_svm=accuracy_score(y_test, y_pred_svm)
print("Accuracy =%0.3f" %(accuracy_svm*100))
cv_svm = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
acc_val_svm=cross_val_score(classifier_svm, X, Y, cv=cv_svm)
print("Accuracy =%0.3f" %(np.mean(acc_val_svm*100)))
y_pred_2019_svm = classifier_svm.predict(X_2019)
accuracy_2019_svm=accuracy_score(Y_2019, y_pred_2019_svm)
print("Accuracy on latest dataset of 2019=%0.3f" %(accuracy_2019_svm*100))
cv_2019svm = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
acc_val2019_svm=cross_val_score(classifier_svm, X_2019, Y_2019, cv=cv_2019svm)
acc_val2019_svm=acc_val2019_svm*100
x_line=[1,2,3,4,5,6,7,8,9,10]
plt.plot(x_line,acc_val2019_svm,'-o')
plt.xlabel("Number of splits")
plt.ylabel("Accuracy")
plt.title("Accuracy using Cross-validation in SVM")
plt.show()
print("-------------------------------------------------")
print("Accuracy on 2019=%0.3f" %(np.mean(acc_val2019_svm)))
precision_2019_svm=precision_score(Y_2019, y_pred_2019_svm,average='weighted')
print("Precision on latest dataset of 2019=%0.3f" %(precision_2019_svm))

recall_2019_svm=recall_score(Y_2019, y_pred_2019_svm,average='weighted')
print("Recall on latest dataset of 2019=%0.3f" %(recall_2019_svm))

fscore_2019_svm=f1_score(Y_2019, y_pred_2019_svm,average='weighted')
print("F1score on latest dataset of 2019=%0.3f" %(fscore_2019_svm))

precision_2019_svm=precision_score(Y_2019, y_pred_2019_svm,average='macro')
print("Precision on latest dataset of 2019=%0.3f" %(precision_2019_svm))

recall_2019_svm=recall_score(Y_2019, y_pred_2019_svm,average='macro')
print("Recall on latest dataset of 2019=%0.3f" %(recall_2019_svm))

fscore_2019_svm=f1_score(Y_2019, y_pred_2019_svm,average='macro')
print("F1score on latest dataset of 2019=%0.3f" %(fscore_2019_svm))

precision_2019_svm=precision_score(Y_2019, y_pred_2019_svm,average='micro')
print("Precision on latest dataset of 2019=%0.3f" %(precision_2019_svm))

recall_2019_svm=recall_score(Y_2019, y_pred_2019_svm,average='micro')
print("Recall on latest dataset of 2019=%0.3f" %(recall_2019_svm))

fscore_2019_svm=f1_score(Y_2019, y_pred_2019_svm,average='micro')
print("F1score on latest dataset of 2019=%0.3f" %(fscore_2019_svm))
