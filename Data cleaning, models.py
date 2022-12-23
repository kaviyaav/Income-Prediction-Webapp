#!/usr/bin/env python
# coding: utf-8

# # Data cleaning

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.mlab as mlab
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure
import os
import io
import requests
import pickle
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot
sns.set(style='white', context='notebook', palette='deep')
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# import the metrics class
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score,precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix
# To ignore warning messages
import warnings
warnings.filterwarnings('ignore')


# In[184]:


# import Dataset and convert into dataframe
data = pd.read_csv("income.csv")
data_income = pd.DataFrame(data)


# In[185]:


# preview of the dataset
data_income.head()


# # Missing Data

# In[187]:


#Missing data headtmap for entire dataset

string_columns = data_income.filter(['workclass', 'education','marital.status','occupation', 'relationship','race','sex','native.country']) # first 30 columns
colours = ['#ffff00', '#000099'] # specify the colours - yellow is missing. blue is not missing
isAlpha_check= string_columns.columns.str.isalpha()
sns.heatmap(isAlpha_check[: , np.newaxis], cmap=sns.color_palette(colours))


# In[188]:


# impute missing data
# first create missing indicator for features with missing data
for col in data_income.columns:
    missing = data_income[col].isnull()
    num_missing = np.sum(missing)
    
    if num_missing > 0:  
        print('created missing indicator for: {}'.format(col))
        data_income['{}_ismissing'.format(col)] = missing


# then based on the indicator, plot the histogram of missing values
ismissing_cols = [col for col in data.columns if 'ismissing' in col]
data_income['num_missing'] = data_income[ismissing_cols].sum(axis=1)

data_income['num_missing'].value_counts().reset_index().sort_values(by='index').plot.bar(x='index', y='num_missing')


# In[189]:


# removing null values to avoid errors
data_income.dropna(inplace = True)


# In[190]:


# cleaning null values from data
features = ['age','workclass', 'fnlwgt','education','education.num','marital.status','occupation', 'relationship','race','sex','capital.gain','capital.loss','hours.per.week','native.country','income']
for feature in features:
    data_income[feature].fillna(data_income[feature].mode()[0],inplace=True) #cleaning NaNs
data_income
    


# In[191]:


# drop row with special character '?'
features = ['age','workclass', 'fnlwgt','education','education.num','marital.status','occupation', 'relationship','race','sex','capital.gain','capital.loss','hours.per.week','native.country','income']
for feature in features:
    data_income.drop(data_income.index[data_income[feature] == '?'], inplace=True)
data_income


# In[192]:


# cleaning special character from data which is drop row
features = ['age','workclass', 'fnlwgt','education','education.num','marital.status','occupation', 'relationship','race','sex','capital.gain','capital.loss','hours.per.week','native.country','income']
for feature in features:
    data_income.drop(data_income.index[data_income['sex'] == '.'], inplace=True)
data_income


# In[193]:


# remove unnessary feature num_missing 
data_income = data_income.drop(['num_missing'], axis=1)
data_income


# In[194]:


# without-pay feature is non-informative for workclass so removing it
# Here workclass column has 14 rows of without-pay members those will not contribute to the problem statement so dropping it
data_income.drop(data_income.index[data_income['workclass'] == 'Without-pay'], inplace=True)
data_income


# In[195]:


# uninformative/repititive for entire dataset
# Here capital.loss will ofcourse have repititive values so it is not an issue
num_rows = len(data_income.index)
low_information_cols = []

for col in data_income.columns:
    cnts = data_income[col].value_counts(dropna=False)
    top_pct = (cnts/num_rows).iloc[0]
    
    if top_pct > 0.95:
        low_information_cols.append(col)
        print('{0}: {1:.5f}%'.format(col, top_pct*100))
        print(cnts)
        print()
data_income


# In[196]:


#  Irrelevant (When the features are not serving the project’s goal, we can remove them.)
# both education and education.num serves the same purpose so dropping education.num feature
data_income = data_income.drop(['education.num'], axis=1)
data_income


# In[198]:


# Viewed the categorical features in the dataset
categorical = [var for var in data_income.columns if data_income[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)


# In[199]:


# drop duplicates
# we dont have unique column to find duplicates and remove
data_income.duplicated()


# In[200]:


# find capitalize values in categorical values
for col_name in data_income.columns:
    if data_income[col_name].dtypes == 'O':
        duplicate_count = len(data_income[col_name].value_counts(dropna=False))
        print("Feature '{col_name}' has {duplicate_count} duplicates".format(col_name=col_name, duplicate_count=duplicate_count))


# In[201]:


# convert categorical values to lowercase since python is case sensitive
for col_name in data_income.columns:
    if data_income[col_name].dtypes == 'O':
        lower_case = data_income[col_name].str.lower()
        duplicate_count = len(data_income[col_name].value_counts(dropna=False))
        print("Feature '{col_name}' has {duplicate_count} duplicates".format(col_name=col_name, duplicate_count=duplicate_count))


# In[202]:


# Renaming the column names and printing the converted one
data_income.rename(columns = {'marital.status':'marital_status'}, inplace = True)
data_income.rename(columns = {'capital.gain':'capital_gain'}, inplace = True)
data_income.rename(columns = {'hours.per.week':'hours_per_week'}, inplace = True)
data_income.rename(columns = {'capital.loss':'capital_loss'}, inplace = True)
data_income.rename(columns = {'native.country':'native_country'}, inplace = True)
data_income


# In[203]:


# View statistical property of dataset which includes both numerical nad categorical
data_income.describe(include='all')


# # Outliers

# In[204]:


# When the feature is categorical, we can use a histogram to detect outliers
data_income['workclass'].hist(bins=100)


# In[205]:


# To study the feature closer for integer features, let’s make a box plot to find outliers
data_income.boxplot(column=['capital_gain'])


# In[206]:


# replace missing values with the median.

category_capital = np.array(['capital_gain'])
pd.Categorical(
   data_income['capital_gain'], categories = category_capital, ordered = True)
median = data_income.loc[data_income['capital_gain']> 1091, 'capital_gain'].median()
data_income.loc[data_income['capital_gain'] > median, 'capital_gain'] = np.nan

data_income


# In[207]:


data_income.boxplot(column=['capital_gain'])


# In[208]:


# Removing NaN values
data_income.dropna(inplace=True)
data_income['capital_gain'].isna().sum()


# In[209]:


data_income['native_country'] = ['Others' if x != ('Mexico') and x != ('United-States') and x != ('Canada') and x != ('India') and x != ('Philippines')
                                 and x != ('Germany') and x != ('Puerto-Rico') and x != ('El-Salvador') else x for x in data_income['native_country']]

print(data_income['native_country'].value_counts().sort_values(ascending=False))


# In[210]:


# Functional Approach of EDA
def initial_eda(data_income):
    if isinstance(data_income, pd.DataFrame):
        total_na = data_income.isna().sum().sum()
        print("Dimensions : %d rows, %d columns" % (data_income.shape[0], data_income.shape[1]))
        print("Total NA Values : %d " % (total_na))
        print("%38s %10s     %10s %10s" % ("Column Name", "Data Type", "#Distinct", "NA Values"))
        col_name = data_income.columns
        dtyp = data_income.dtypes
        uniq = data_income.nunique()
        na_val = data_income.isna().sum()
        for i in range(len(data_income.columns)):
            print("%38s %10s   %10s %10s" % (col_name[i], dtyp[i], uniq[i], na_val[i]))
        
    else:
        print("Expect a DataFrame but got a %15s" % (type(data_income)))
initial_eda(data_income)


# # Correlation

# In[166]:


# Find out the correlations
# plot correlation heatmap to find out correlations

data_income.corr().style.format("{:.4}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


# # Declare feature vector and target variable 

# In[36]:


# split the taget variable as and dependent variables
X = data_income.drop(['income'], axis=1)

y = data_income['income']


# In[37]:


# Split data into separate training and test set 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# check the shape of X_train and X_test

X_train.shape, X_test.shape


# # Feature Engineering

# In[38]:


# display the categorical values
categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical


# In[39]:


# display the numerical values
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical


# In[40]:


# Engineering missing values in categorical variables
X_train[categorical].isnull().mean()


# In[41]:


import category_encoders as ce


# In[42]:


encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 
                                 'race', 'sex', 'native_country'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[43]:


X_train.head()


# In[44]:


X_train.shape


# In[45]:


X_test.head()


# In[46]:


X_test.shape


# # Feature Scaling

# In[47]:


cols = X_train.columns


# In[48]:


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[49]:


X_train = pd.DataFrame(X_train, columns=[cols])


# In[50]:


X_test = pd.DataFrame(X_test, columns=[cols])


# In[51]:


X_train


# In[52]:


X_test


# # Model Development

# # KNeighborsClassifier

# In[90]:


from sklearn.neighbors import KNeighborsClassifier

mcc_scores = []
K_values = []

for K in range(1, 13):
    model = KNeighborsClassifier(n_neighbors = K)
    model.fit(X_train, y_train)

    test_prediction = model.predict(X_test)
    test_mcc = matthews_corrcoef(y_test, test_prediction)

    K_values.append(K)
    mcc_scores.append(test_mcc)

ax = sns.lineplot(x = K_values, y = mcc_scores)
ax.set(xlabel='K Neigbours considered', ylabel='MCC Score')
print(f'Best MCC score |{max(mcc_scores)}| achieved with K |{K_values[np.argmax(mcc_scores)]}|')

model3 = KNeighborsClassifier(n_neighbors = K_values[np.argmax(mcc_scores)])
model3.fit(X_train, y_train)


# # Naive Bayes

# In[53]:


# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB


# instantiate the model
gnb = GaussianNB()


# fit the model
gnb.fit(X_train, y_train)


# In[54]:


y_pred = gnb.predict(X_test)

y_pred


# In[55]:


from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[56]:


y_pred_train = gnb.predict(X_train)

y_pred_train


# In[57]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# In[58]:


# print the scores on training and test set

print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))


# In[59]:


# check class distribution in test set

y_test.value_counts()


# In[60]:


# check null accuracy score

null_accuracy = (7407/(7407+2362))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


# In[61]:


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[62]:


# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# In[63]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[64]:


TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# In[65]:


# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))


# In[66]:


# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))


# In[67]:


# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))


# In[68]:


recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))


# In[69]:


true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))


# In[70]:


false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))


# In[71]:


specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# In[72]:


# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = gnb.predict_proba(X_test)[0:10]

y_pred_prob


# In[73]:


# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50K', 'Prob of - >50K'])

y_pred_prob_df


# In[74]:


# print the first 10 predicted probabilities for class 1 - Probability of >50K

gnb.predict_proba(X_test)[0:10, 1]


# In[75]:


# store the predicted probabilities for class 1 - Probability of >50K

y_pred1 = gnb.predict_proba(X_test)[:, 1]


# In[76]:


# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of salaries >50K')


# set the x-axis limit
plt.xlim(0,1)
# set the title
plt.xlabel('Predicted probabilities of salaries >50K')
plt.ylabel('Frequency')


# In[77]:


# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = '>50K')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Salaries')

plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()


# In[78]:


# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


# In[79]:


# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(gnb, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))


# In[82]:


# Applying 10-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(gnb, X_train, y_train, cv = 10, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))


# In[83]:


# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))


# # Decision Tree

# In[86]:


data = pd.read_csv("income.csv")
df = pd.DataFrame(data)
df.head(5)


# In[87]:


# dropping the rows having missing values in workclass
df = df[df['workclass'] !='?']
df.head()


# In[88]:


# dropping the "?"s from occupation and native.country
df = df[df['occupation'] !='?']
df = df[df['native.country'] !='?']


# In[89]:


from sklearn import preprocessing

# encode categorical variables using label Encoder

# select all categorical variables
df_categorical = df.select_dtypes(include=['object'])
df_categorical.head()


# In[90]:


# apply label encoder to df_categorical
le = preprocessing.LabelEncoder()
df_categorical = df_categorical.apply(le.fit_transform)
df_categorical.head()


# In[91]:


# Next, Concatenate df_categorical dataframe with original df (dataframe)

# first, Drop earlier duplicate columns which had categorical values
df = df.drop(df_categorical.columns,axis=1)
df = pd.concat([df,df_categorical],axis=1)
df.head()


# In[92]:


# convert target variable income to categorical
df['income'] = df['income'].astype('category')


# In[93]:


# Importing train_test_split
from sklearn.model_selection import train_test_split


# In[94]:


# Putting independent variables/features to X
X = df.drop('income',axis=1)

# Putting response/dependent variable/feature to y
y = df['income']


# In[95]:


# Splitting the data into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=99)

X_train.head()


# In[96]:


# Importing decision tree classifier from sklearn library
from sklearn.tree import DecisionTreeClassifier

# Fitting the decision tree with default hyperparameters, apart from
# max_depth which is 5 so that we can plot and read the tree.
dt_default = DecisionTreeClassifier(max_depth=5)
dt_default.fit(X_train,y_train)


# In[97]:


# Let's check the evaluation metrics of our default model

# Importing classification report and confusion matrix from sklearn metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# making predictions
y_pred_default = dt_default.predict(X_test)

# Printing classifier report after prediction
print(classification_report(y_test,y_pred_default))


# In[98]:


# Printing confusion matrix and accuracy
conf_matrix=(confusion_matrix(y_test,y_pred_default))
print(conf_matrix)
print(accuracy_score(y_test,y_pred_default))


# In[99]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[100]:


# Importing required packages for visualization
import six
import sys
sys.modules['sklearn.externals.six'] = six
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus,graphviz

# Putting features
features = list(df.columns[1:])
features


# In[101]:


# plotting tree with max_depth=3
import graphviz

g = graphviz.Graph(format='png')
dot_data = StringIO()  
export_graphviz(dt_default, out_file=dot_data,
                feature_names=features, filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[102]:


# GridSearchCV to find optimal max_depth
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(1, 40)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy")
tree.fit(X_train, y_train)


# In[103]:


# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[104]:


# GridSearchCV to find optimal max_depth
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_leaf': range(5, 200, 20)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy")
tree.fit(X_train, y_train)


# In[105]:


# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[106]:


# GridSearchCV to find optimal min_samples_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_split': range(5, 200, 20)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy")
tree.fit(X_train, y_train)


# In[107]:


# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[108]:


# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'criterion': ["entropy", "gini"]
}

n_folds = 5

# Instantiate the grid search model
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 
                          cv = n_folds, verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train,y_train)


# In[109]:


# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results


# In[110]:


# printing the optimal accuracy score and hyperparameters
print("best accuracy", grid_search.best_score_)
print(grid_search.best_estimator_)


# In[111]:


# model with optimal hyperparameters
clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=10, 
                                  min_samples_leaf=50,
                                  min_samples_split=50)
clf_gini.fit(X_train, y_train)


# In[112]:


clf_gini.score(X_test,y_test)


# In[113]:


# plotting the tree
dot_data = StringIO()  
export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[114]:


# tree with max_depth = 3
clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=3, 
                                  min_samples_leaf=50,
                                  min_samples_split=50)
clf_gini.fit(X_train, y_train)

# score
print(clf_gini.score(X_test,y_test))


# In[115]:


# plotting tree with max_depth=3
dot_data = StringIO()  
export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[116]:


# classification metrics
from sklearn.metrics import classification_report,confusion_matrix
y_pred = clf_gini.predict(X_test)
print(classification_report(y_test, y_pred))


# In[117]:


# confusion matrix
conf_matrix=(confusion_matrix(y_test,y_pred))
print(conf_matrix)


# In[118]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# # Logistic Regression

# In[120]:


columns = ["age", "workClass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", "sex", 
           "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

train_data = pd.read_csv('adult.data', names = columns, sep=' *, *', na_values='?')
test_data = pd.read_csv('adult.test', names = columns, sep=' *, *', skiprows =1, na_values='?')


# In[121]:


#ColumnSelector Pipeline
class ColumnsSelector(BaseEstimator, TransformerMixin):
  
  def __init__(self, type):
    self.type = type
  
  def fit(self, X, y=None):
    return self
  
  def transform(self,X):
    return X.select_dtypes(include=[self.type])


# In[122]:


#Numerical Data Pipeline
num_pipeline = Pipeline(steps=[
    ("num_attr_selector", ColumnsSelector(type='int')),
    ("scaler", StandardScaler())
])


# In[123]:


class CategoricalImputer(BaseEstimator, TransformerMixin):
  
  def __init__(self, columns = None, strategy='most_frequent'):
    self.columns = columns
    self.strategy = strategy
    
    
  def fit(self,X, y=None):
    if self.columns is None:
      self.columns = X.columns
    
    if self.strategy is 'most_frequent':
      self.fill = {column: X[column].value_counts().index[0] for column in self.columns}
    else:
      self.fill ={column: '0' for column in self.columns}
      
    return self
      
  def transform(self,X):
    X_copy = X.copy()
    for column in self.columns:
      X_copy[column] = X_copy[column].fillna(self.fill[column])
    return X_copy


# In[124]:


class CategoricalEncoder(BaseEstimator, TransformerMixin):
  
  def __init__(self, dropFirst=True):
    self.categories=dict()
    self.dropFirst=dropFirst
    
  def fit(self, X, y=None):
    join_df = pd.concat([train_data, test_data])
    join_df = join_df.select_dtypes(include=['object'])
    for column in join_df.columns:
      self.categories[column] = join_df[column].value_counts().index.tolist()
    return self
    
  def transform(self, X):
    X_copy = X.copy()
    X_copy = X_copy.select_dtypes(include=['object'])
    for column in X_copy.columns:
      X_copy[column] = X_copy[column].astype({column: CategoricalDtype(self.categories[column])})
    return pd.get_dummies(X_copy, drop_first=self.dropFirst)


# In[125]:


cat_pipeline = Pipeline(steps=[
    ("cat_attr_selector", ColumnsSelector(type='object')),
    ("cat_imputer", CategoricalImputer(columns=['workClass','occupation', 'native-country'])),
    ("encoder", CategoricalEncoder(dropFirst=True))
])


# In[126]:


full_pipeline = FeatureUnion([("num_pipe", num_pipeline), ("cat_pipeline", cat_pipeline)])


# In[127]:


#Building the Model
train_data.drop(['fnlwgt', 'education'], axis=1, inplace=True)
test_data.drop(['fnlwgt', 'education'], axis=1, inplace=True)


# In[128]:


#Preparing the data for training
# copy the data before preprocessing
train_copy = train_data.copy()

# convert the income column to 0 or 1 and then drop the column for the feature vectors
train_copy["income"] = train_copy["income"].apply(lambda x:0 if x=='<=50K' else 1)

# creating the feature vector 
X_train = train_copy.drop('income', axis =1)

# target values
Y_train = train_copy['income']

print(X_train.columns)


# In[129]:


#Training the model
# set parameter type_df as train for categorical encoder 
# we can set parameter using the name of the transformer while defining the pipeline
# syntax:  name_of_the_transformer__ = 

# pass the data through the full_pipeline
X_train_processed = full_pipeline.fit_transform(X_train)
print(X_train_processed.shape)


# In[130]:


model = LogisticRegression(random_state=0)
model.fit(X_train_processed, Y_train)


# In[131]:


model.coef_


# In[132]:


#Testing the model
# take a copy of the test data set
test_copy = test_data.copy()

# convert the income column to 0 or 1
test_copy["income"] = test_copy["income"].apply(lambda x:0 if x=='<=50K.' else 1)

# separating the feature vecotrs and the target values
X_test = test_copy.drop('income', axis =1)
Y_test = test_copy['income']

X_test.columns


# In[133]:


# preprocess the test data using the full pipeline
# here we set the type_df param to 'test'
X_test_processed = full_pipeline.fit_transform(X_test)
X_test_processed.shape


# In[134]:


predicted_classes = model.predict(X_test_processed)
print(predicted_classes)


# In[135]:


#Model Evaluation
accuracy_score(predicted_classes, Y_test.values)


# In[136]:


sns.set(rc={'figure.figsize':(8,6)})
cfm = confusion_matrix(predicted_classes, Y_test.values)
sns.heatmap(cfm, annot=True)
print(cfm)
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')


# In[137]:


#Cross Validation
cross_val_model = LogisticRegression(random_state=0)
scores = cross_val_score(cross_val_model, X_train_processed, Y_train, cv=5)
print(scores)
print(np.mean(scores))


# In[138]:


#Fine Tuning the Model
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
random_state=[0]

# creating a dictionary of hyperparameters
hyperparameters = dict(C=C, penalty=penalty, random_state=random_state)


# In[139]:


clf = GridSearchCV(estimator = model, param_grid = hyperparameters, cv=5)
best_model = clf.fit(X_train_processed, Y_train)
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])


# In[140]:


best_predicted_values = best_model.predict(X_test_processed)
print(best_predicted_values)


# In[141]:


accuracy_score(best_predicted_values, Y_test.values)


# # Random Forest

# In[140]:


data = pd.read_csv("income.csv")
df = pd.DataFrame(data)
df.head(5)


# In[141]:


df['income']=df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
df.head(4)


# In[142]:


# code will replace the special character to nan and then drop the columns 
df['native.country'] = df['native.country'].replace('?',np.nan)
df['workclass'] = df['workclass'].replace('?',np.nan)
df['occupation'] = df['occupation'].replace('?',np.nan)


# In[143]:


df['native.country'] = ['Others' if x != ('Mexico') and x != ('United-States') and x != ('Canada') and x != ('India') and x != ('Philippines')
                                 and x != ('Germany') and x != ('Puerto-Rico') and x != ('El-Salvador') else x for x in df['native.country']]


# In[144]:


df['education'].unique()


# In[145]:


def customize_edu_level(x):
    if 'Bachelors' in x:
        return 'Bachelor’s degree'
    if 'Masters' in x:
        return 'Master’s degree'
    if 'Prof-school' in x or 'Doctorate' in x:
        return 'Ph.d or Doctorate degree'
    return 'Less than a Bachelors'

df['education'] = df['education'].apply(customize_edu_level)


# In[146]:


df['age'].unique()


# In[147]:


def customize_age(x):
    if x < 20:
        return 'Below 20'
    if x <= 30 and x >= 20:
        return '20 - 30'
    if x <= 40 and x > 30:
        return '31 - 40'
    if x <= 50 and x > 40:
        return '41 - 50'
    if x <= 60 and x > 50:
        return '51- 60'
    if x > 60:
        return 'Above 60'
    
df['age'] = df['age'].apply(customize_age)


# In[148]:


df_new = df.copy()


# In[149]:


df_new.dropna(how='any',inplace=True)


# In[150]:


#dropping based on uniquness of data from the dataset 
df_new.drop(['fnlwgt', 'capital.gain','capital.loss', 'education.num', 'sex', 'race', 'marital.status', 'workclass', 'relationship', 'occupation', 'hours.per.week'], axis=1, inplace=True)


# In[152]:


df_new


# In[151]:


#gender
# df_new['age'] = df_new['age'].map({'Below 20': 0, '20 - 30': 1, '31 - 40': 2,'41 - 50': 3,'51- 60': 4,'Above 60': 5}).astype(int)
# df_new['native.country'] = df_new['native.country'].map({'Mexico': 0, 'United-States': 1, 'Canada': 2,'India': 3,'Philippines': 4,'Germany': 5,'Puerto-Rico': 6,'El-Salvador': 7,'Others': 8}).astype(int)
# df_new['sex'] = df_new['sex'].map({'Male': 0, 'Female': 1}).astype(int)
# df_new['race'] = df_new['race'].map({'Black': 0, 'Asian-Pac-Islander': 1,'Other': 2, 'White': 3, 'Amer-Indian-Eskimo': 4}).astype(int)
# df_new['marital.status'] = df_new['marital.status'].map({'Married-spouse-absent': 0, 'Widowed': 1, 'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4,'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)
# df_new['workclass']= df_new['workclass'].map({'Self-emp-inc': 0, 'State-gov': 1,'Federal-gov': 2, 'Without-pay': 3, 'Local-gov': 4,'Private': 5, 'Self-emp-not-inc': 6}).astype(int)
# df_new['relationship'] = df_new['relationship'].map({'Not-in-family': 0, 'Wife': 1, 'Other-relative': 2, 'Unmarried': 3,'Husband': 4,'Own-child': 5}).astype(int)
# df_new['education']= df_new['education'].map({'Bachelor’s degree': 0, 'Master’s degree': 1, 'Ph.d or Doctorate degree': 2, 'Less than a Bachelors': 3}).astype(int)
# df_new['occupation'] = df_new['occupation'].map({ 'Farming-fishing': 0, 'Tech-support': 1, 'Adm-clerical': 2, 'Handlers-cleaners': 3, 
#  'Prof-specialty': 4,'Machine-op-inspct': 5, 'Exec-managerial': 6,'Priv-house-serv': 7,'Craft-repair': 8,'Sales': 9, 'Transport-moving': 10, 'Armed-Forces': 11, 'Other-service': 12,'Protective-serv':13}).astype(int)


# In[154]:


from sklearn.preprocessing import LabelEncoder
label_encode_edu = LabelEncoder()
df_new['education'] = label_encode_edu.fit_transform(df_new['education'])
df_new["education"].unique()


# In[155]:


label_encode_country = LabelEncoder()
df_new['native.country'] = label_encode_country.fit_transform(df_new['native.country'])
df_new["native.country"].unique()


# In[157]:


label_encode_age = LabelEncoder()
df_new['age'] = label_encode_age.fit_transform(df_new['age'])
df_new["age"].unique()


# In[158]:


df_new.head()


# In[160]:


X = df_new.drop('income',axis=1)
y = df_new.income


# In[161]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,shuffle=True)


# In[162]:


rf=RandomForestClassifier(min_samples_split=30)
# Train the model using the training sets
rf.fit(X_train,y_train)
predictions_rf =rf.predict(X_test)
predictions_rf


# In[163]:


accuracy_rf = metrics.accuracy_score(y_test, predictions_rf)


# In[164]:


print(f"The accuracy of the model is {round(metrics.accuracy_score(y_test,predictions_rf),3)*100} %")


# In[165]:


auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print(f"The AUC Score  is {round(auc_rf,3)*100} %")


# In[166]:


#AUC Random Forest
train_probs = rf.predict_proba(X_train)[:,1] 
probs = rf.predict_proba(X_test)[:, 1]
train_predictions = rf.predict(X_train)


# In[167]:


def evaluate_model(y_pred, probs,train_predictions, train_probs):
   
     # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.show()


# In[168]:


evaluate_model(predictions_rf,probs,train_predictions,train_probs)


# In[169]:


import itertools
def plot_confusion_matrix(cm, classes, normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens): # can change color 
    plt.figure(figsize = (5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
             plt.text(j, i, format(cm[i, j], fmt), 
             fontsize = 20,
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

# Let's plot it out
cm = confusion_matrix(y_test, predictions_rf)
plot_confusion_matrix(cm, classes = ['0 - <50k', '1 - >50k'],
                      title = 'Confusion Matrix')


# In[170]:


#Feature importance of our model
feature_importances = list(zip(X_train, rf.feature_importances_))
# Then sort the feature importances by most important first
feature_importances_ranked = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Feature: {:35} Importance: {}'.format(*pair)) for pair in feature_importances_ranked];


# In[43]:


feature_names_8 = [i[0] for i in feature_importances_ranked[:8]]
y_ticks = np.arange(0, len(feature_names_8))
x_axis = [i[1] for i in feature_importances_ranked[:8]]
plt.figure(figsize = (10, 14))
plt.barh(feature_names_8, x_axis)   #horizontal barplot
plt.title('Random Forest Feature Importance (Top 25)',
          fontdict= {'fontname':'Comic Sans MS','fontsize' : 20})
plt.xlabel('Features',fontdict= {'fontsize' : 16})
plt.show()


# In[171]:


X


# In[174]:


# country, edlevel, yearscode
X = np.array([["United-States", 'Master’s degree', 'Below 20' ]])
X


# In[175]:


X[:, 0] = label_encode_country.transform(X[:,0])
X[:, 1] = label_encode_edu.transform(X[:,1])
X[:, 2] = label_encode_age.transform(X[:,2])
X = X.astype(float)
X


# In[176]:


y_pred = rf.predict(X)
y_pred


# In[177]:


import pickle


# In[178]:


data = {"model": rf, "label_encode_country": label_encode_country, "label_encode_edu": label_encode_edu, "label_encode_age":label_encode_age}
with open('random_forest_income_pred.pkl', 'wb') as file:
    pickle.dump(data, file)


# In[179]:


with open('random_forest_income_pred.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data["model"]
label_encode_country = data["label_encode_country"]
label_encode_edu = data["label_encode_edu"]
label_encode_age = data["label_encode_age"]


# In[181]:


y_pred = regressor_loaded.predict(X)
y_pred


# # KNN

# In[241]:


data = pd.read_csv("income.csv")
df = pd.DataFrame(data)
df.head(5)


# In[242]:


df['income']=df['income'].map({'<=50K': 0, '>50K': 1})
df.head()


# In[243]:


from numpy import nan
df=df.replace("?",nan)


# In[244]:


null_values=df.isnull().sum()
null_values=pd.DataFrame(null_values,columns=['null'])
j=1
sum_tot=len(df)
null_values['percent']=null_values['null']/sum_tot
round(null_values*100,3).sort_values('percent',ascending=False)


# In[245]:


#educational-num seems to be not so important
df=df.drop(["education.num"],axis=1)
df=df.drop(["fnlwgt"],axis=1)


# In[246]:


le = preprocessing.LabelEncoder()


# In[247]:


df[['age', 'workclass', 'education', 'marital.status', 'occupation',
       'relationship', 'race', 'sex', 'capital.gain', 'capital.loss',
       'hours.per.week', 'native.country']]=df[['age', 'workclass', 'education', 'marital.status', 'occupation',
       'relationship', 'race', 'sex', 'capital.gain', 'capital.loss',
       'hours.per.week', 'native.country']].apply(le.fit_transform)


# In[248]:


X=df.drop(["income"],axis=1)
y=df["income"]


# In[249]:


#Split Train and Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[250]:


#Using KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)


# In[251]:


pred = knn.predict(X_test)


# In[252]:


#Predictions and Evaluations
print(confusion_matrix(y_test,pred))


# In[253]:


plot_confusion_matrix(knn,X_test,y_test)


# In[254]:


print(classification_report(y_test,pred))


# In[255]:


#Choosing a K Value
accuracy_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,X,df['income'],cv=10)
    accuracy_rate.append(score.mean())


# In[256]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('accuracy_rate vs. K Value')
plt.xlabel('K')
plt.ylabel('accuracy_rate')


# In[259]:


# NOW WITH K=18
knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=23')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[258]:


plot_confusion_matrix(knn,X_test,y_test)


# In[ ]:




