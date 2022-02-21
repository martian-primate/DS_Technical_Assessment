#ignore warnings
import warnings
warnings.filterwarnings("ignore")

#Dependencies
import pandas as pd
import numpy as np

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Modelling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Cross validation
from sklearn.model_selection import KFlold, StratifiedKFold, cross_val_score, GridSearchCV

#ensemble methods
from sklearn import linear_model, tree, ensemble, svm

#gradient coost classifier
from sklearn.ensemble import GradientBoostingClass
from sklearn.metrics import accuracy_score

#Metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

#Uploading data
file_path = 'C:/Users\Kaspersky/Downloads/drive-download-20220218T092015Z-001/combined_data.csv'
df = pd.read_csv(file_path)
df.head(5)
df.info()

#A function to check if there are missing values and if so print the count
def missing_value(dataframe):
    if dataframe.isnull().values.any():
        print(dataframe.isnull().sum())
    else:
        print('No missing valuse')
missing_value(df)

#Continous variables
df.describe()

#Conversion of data types
df['TotalCharges'] = df['TotalCharges'].replace([' '],[0])
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='raise')

#Heat map to show correlation of the continous variables
heatdf = df.loc[:,['tenure','MonthlyCharges','TotalCharges']]
sns.heatmap(heatdf.corr())
print(heatdf.corr())

#Distribution Plots of the continous variables
df_num = ['tenure','MonthlyCharges','TotalCharges']

# plot Numerical Data

a = 3  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(10,15))

for i in df_num:
    plt.subplot(a, b, c)
    plt.title('{} (dist), subplot: {}{}{}'.format(i, a, b, c))
    plt.xlabel(i)
    sns.distplot(heatdf[i])
    c = c + 1

    plt.subplot(a, b, c)
    plt.title('{} (box), subplot: {}{}{}'.format(i, a, b, c))
    plt.xlabel(i)
    plt.boxplot(x = heatdf[i])
    c = c + 1

plt.show()

#Distribution of the continous variable with respect to the target variable
for i in df_num:
    
    fig, ax = plt.subplots()

    ax.hist(df[df['Churn']=='No'][i], bins=15, alpha=0.5, color="red", label="No")
    ax.hist(df[df['Churn']=='Yes'][i], bins=15, alpha=0.5, color="blue", label="Yes")

    ax.set_xlabel(i)
    ax.set_ylabel("Frequency")

    ax.legend();

df['SeniorCitizen'] = df['SeniorCitizen'].replace([1,0],['Yes','No'])
df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

#Categorical Variables
df_cat = df.select_dtypes(include=['object'])
df_cat = df_cat.drop(columns=['customerID','Churn'])

cat_col = list(df_cat.columns.values)

cat_col

count=0
for i in cat_col:
    count+=1
print(count)

#Target Distribution
df['Churn'].value_counts().plot.bar(title = 'Churn Distribution')
print(df['Churn'].value_counts())

for column in cat_col:
    
    
    df[column].value_counts().plot.bar(title = column)
    plt.tight_layout()
    #plt.xlabel('{}'.format(column))
    plt.show()

#Cross Tabulation to show distribution of the categorical variable respect to the target variable

for i in cat_col:

    pd.crosstab(df[i], df.Churn).plot(kind = 'bar')
    plt.xlabel('{}'.format(i))
    plt.ylabel('Churn Status Frequency')
    
plt.show()

#Determining number of categories for the categorical values

object_cols = [col for col in df.columns if df[col].dtype == "object"]

# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: df[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])

#We apply one hot encoding for feature engineering 
for var in cat_col:
    cat_list = 'var' + '_' + 'var'
    cat_list = pd.get_dummies(df[var], prefix=var)
    df_New = pd.concat([df,cat_list],axis = 1)
    df = df_New

data_vars = df.columns.values.tolist()

to_keep = [i for i in data_vars if i not in cat_col]

df_final = df[to_keep]

df_final.columns.values

df_final['Churn'] = df_final['Churn'].replace(['Yes','No'],[1,0])

df_model =df_final.drop(columns = ['customerID'])

label = np.array(df_model['Churn'])

feature_df = df_model.drop('Churn', axis=1)

#A list of features
feature_list = list(feature_df.columns)

# change the feature dataframe to an array
feature= np.array(feature_df)

#Baseline Logistic Model
feature_train, feature_test, label_train, label_test = train_test_split(feature, label,
                                                                        test_size = 0.2, random_state= 42)

#Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)

# Train the model using the training data
rf.fit(feature_train, label_train)

y_pred = rf.predict(feature_test)
acc = accuracy_score(label_test,y_pred)
print('Baseline Model Accuracy: ', acc)

#Apply k fold cross Validation and try other candidate models
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

count = 1

for train_index, test_index in kf.split(feature,label):
    print(f'Fold:{count}, Train set: {len(train_index)},Validation set:{len(test_index)}')
    count+=1

#Random forest after cross validation
score = cross_val_score(ensemble.RandomForestClassifier(random_state= 42), 
                        feature, label, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

#Using Gradient Boosting Classifier
score = cross_val_score(ensemble.GradientBoostingClassifier(random_state= 42), 
                        feature, label, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Gradient Boost Average score: {"{:.2f}".format(score.mean())}')

gbc = ensemble.GradientBoostingClassifier(random_state=42)
gbc.fit(feature, label)

#Top most important features
col_sorted_by_importance=gbc.feature_importances_.argsort()
feat_imp=pd.DataFrame({
    'cols':feature_df.columns[col_sorted_by_importance],
    'imps':gbc.feature_importances_[col_sorted_by_importance]
})
print(feat_imp.sort_values(by=['imps'], ascending=False).head(10))

#Getting the best model using Grid Search Cross validation
model = GradientBoostingClassifier(random_state=1)
space = dict()
space['n_estimators'] = [10, 100, 500]
space['max_features'] = [2, 4, 6]
search = GridSearchCV(model,space, scoring='accuracy',cv = kf, refit=True)
result = search.fit(feature_train, label_train)
best_model = result.best_estimator_

#Evaluate the model on the hold out dataset

yhat = best_model.predict(feature_test)

#Accuracy of the final model

acc = accuracy_score(label_test,yhat)
print(f'Best Model Accuracy: {"{:.3f}".format(acc)}')

#metrics
print("Confusion Matrix:")
print(confusion_matrix(label_test,yhat))
print()
print("Classification Report")
print(classification_report(label_test,yhat))

y_scores_gb = gbc.decision_function(feature_test)
fpr_gb, tpr_gb, _ = roc_curve(label_test, y_scores_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))
