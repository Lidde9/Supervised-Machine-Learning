#!/usr/bin/env python
# coding: utf-8

# # Project #5 Supervised Machine Learning

# Project Description:
# 
# https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
# 
# https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5#Sec10

# Table of Contents:
# 1) Import Data and Libraries
# 2) EDA/Inspect and Clean Data
# 3) Test Classification Models
# 4) Tuning the Best Model
# 5) Final Evaluation

# ![image.png](attachment:image.png)

# # 1) Import Data and Libraries

# In[121]:


# Libraries:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score,f1_score,roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import plot_tree


# In[122]:


# Import Data:
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")


# # 2) EDA Inspect and Prepare Data

# In[123]:


df.info()


# In[124]:


df.describe()


# In[125]:


df.head(5)


# In[126]:


df["DEATH_EVENT"].value_counts()


# In[127]:


correlation_matrix = df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[128]:


correlation_matrix = df.corr()

# Create a mask to hide the upper triangle of the correlation matrix
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Create a heatmap of the correlation matrix with the upper triangle masked
plt.figure(figsize=(10, 8))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True, cmap='mako', fmt=".2f", mask=mask, annot_kws={"fontsize": 12})
plt.title("Correlation Matrix", fontsize=16)
plt.tick_params(axis='both', labelsize=12)
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.tight_layout()
plt.savefig(fname="corrmat.png", format="png", transparent=True)
plt.show()


# In[129]:


df["creatinine_phosphokinase"].plot()


# In[130]:


df["creatinine_phosphokinase"].plot(kind="hist")


# In[131]:


# Assuming 'df' is your DataFrame
variables_to_plot = ["time", "serum_creatinine", "ejection_fraction"]

plt.figure(figsize=(15, 5))  # Set the figure size for horizontal plots

for i, col in enumerate(variables_to_plot):
    plt.subplot(1, len(variables_to_plot), i + 1)  # Create subplots in a horizontal row
    sns.violinplot(data=df, y=col, x="DEATH_EVENT", cut=0, palette="rocket")
    plt.title(f"{col} vs. DEATH_EVENT", fontsize=16)  # Increase title font size
    plt.xlabel("DEATH_EVENT", fontsize=14)  # Increase x-axis label font size
    plt.ylabel(col, fontsize=14)  # Increase y-axis label font size

plt.savefig(fname="violin.png", format="png", transparent=True)
plt.tight_layout()
plt.show()


# In[132]:


# Assuming 'df' is your DataFrame
num_features = len(df.columns) - 1  # Excluding 'DEATH_EVENT'
num_rows = (num_features + 2) // 3  # Calculate the number of rows needed for subplots

plt.figure(figsize=(15, 5 * num_rows))  # Set the figure size dynamically

for i, col in enumerate(df.columns):
    if col != 'DEATH_EVENT':
        plt.subplot(num_rows, 3, i + 1)  # Create subplots in a dynamic grid
        sns.violinplot(data=df, x="DEATH_EVENT", y=col, cut=0)
        plt.title(f"{col} vs. DEATH_EVENT")

plt.tight_layout()
plt.show()


# In[133]:


df["ejection_fraction"].plot()


# In[134]:


df["platelets"].plot()


# In[135]:


df["serum_creatinine"].plot()


# In[136]:


df["serum_creatinine"].plot(kind="hist")


# In[137]:


# Isolate target variable and features
X = df.drop(["DEATH_EVENT"], axis=1)
y = df["DEATH_EVENT"]

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


# In[138]:


columns_to_keep = ["DEATH_EVENT", "ejection_fraction", "serum_creatinine"]

df2 = df[columns_to_keep].copy()

X2 = df2.drop(["DEATH_EVENT"], axis=1)
y2 = df2["DEATH_EVENT"]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=.2) #, random_state=42)


# In[139]:


binary_categorical_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

def to_boolean(df):
    df[binary_categorical_cols] = df[binary_categorical_cols].astype(bool)
    return df

to_boolean(df)


# In[140]:


df


# In[141]:


# Preprocessing steps for numerical data
num_transformer = make_pipeline(StandardScaler())

# Define the selector for numerical columns
num_features = make_column_selector(dtype_include="number")

# Put the pipeline together
preprocessor = make_column_transformer(
    (num_transformer, num_features)
)


# # 3) Test Models

# ## 3.1 Including all features

# In[142]:


classifiers = [
    LogisticRegression(max_iter=1000),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    SVC(),
    KNeighborsClassifier(),
]


# In[143]:


for classifier in classifiers:
    pipe = make_pipeline(preprocessor, classifier)
    grid = GridSearchCV(estimator=pipe, param_grid={}, cv=5, scoring="recall")
    
    grid.fit(X_train, y_train)
    
    print(f"Train score for {classifier} is {grid.best_score_}")
    print("")


# In[144]:


for classifier in classifiers:
    pipe = make_pipeline(preprocessor, classifier)
    grid = GridSearchCV(estimator=pipe, param_grid={}, cv=5, scoring="recall")
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    pre = precision_score(y_train, y_pred)
    rec = recall_score(y_train, y_pred)

    print(f"Model: {classifier}")
    print(f"Best Hyperparameters: {grid.best_params_}")
    print(f"Train score for {classifier} for recall {grid.best_score_}")
    print(f"For the full training data the scores are Accuracy: {acc}, Precision: {pre}, Recall: {rec}")
    print("")
    print("")


# ## 3.2 Only including 2 features

# In[145]:


for classifier in classifiers:
    pipe2 = make_pipeline(preprocessor, classifier)
    grid2 = GridSearchCV(estimator=pipe2, param_grid={}, cv=5, scoring="recall")
    grid2.fit(X2_train, y2_train)
    y2_pred = grid2.predict(X2_train)
    acc2 = accuracy_score(y2_train, y2_pred)
    pre2 = precision_score(y2_train, y2_pred)
    rec2 = recall_score(y2_train, y2_pred)

    print(f"Model: {classifier}")
    print(f"Best Hyperparameters: {grid2.best_params_}")
    print(f"Train score for {classifier} for recall {grid2.best_score_}")
    print(f"For the full training data the scores are Accuracy: {acc2}, Precision: {pre2}, Recall: {rec2}")
    print("")
    print("")


# # 4) Tuning Best Model

# In[146]:


rfb = RandomForestClassifier()

pipe_rfb = make_pipeline(preprocessor, rfb)

grid_rfb = GridSearchCV(pipe_rfb, param_grid={}, cv=10, scoring="recall")

grid_rfb.fit(X_train, y_train)


# In[147]:


y_predb = grid_rfb.predict(X_train)
accb = accuracy_score(y_train, y_predb)
preb = precision_score(y_train, y_predb)
recb = recall_score(y_train, y_predb)

# Calculate F1 score
f1b = f1_score(y_train, y_predb)

# Calculate AUC-ROC score
y_pred_prob_b = grid_rfb.predict_proba(X_train)[:, 1]  # Get the predicted probabilities for class 1
auc_rocb = roc_auc_score(y_train, y_pred_prob_b)

print("Best score: ", grid_rfb.best_score_)
print("Best params: ", grid_rfb.best_params_)

print(f"For the full training data, the scores are:\n Accuracy: {accb}, Precision: {preb}, Recall: {recb}, F1 Score: {f1b}, AUC-ROC Score: {auc_rocb}")


# In[148]:


best_model_nt = grid_rfb.best_estimator_
best_model_nt.fit(X_train, y_train)


# In[149]:


# Parameter grid for RandomForestClassifier
param_grid_rf = {
    'randomforestclassifier__n_estimators': [10, 50, 100, 200],     # Number of trees in the forest
    'randomforestclassifier__max_depth': [None, 2, 5, 10, 20],  # Maximum depth of the trees
    'randomforestclassifier__min_samples_split': [2, 3, 5, 10], # Minimum samples required to split a node
    'randomforestclassifier__min_samples_leaf': [1, 2, 4, 10],  # Minimum samples required at leaf nodes
}


# In[150]:


rf = RandomForestClassifier()

pipe_rf = make_pipeline(preprocessor, rf)

grid_rf = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=10, scoring="recall")

grid_rf.fit(X_train, y_train)


# In[151]:


print("Best score: ", grid_rf.best_score_)
print("Best params: ", grid_rf.best_params_)

y_pred = grid_rf.predict(X_train)
acc = accuracy_score(y_train, y_pred)
pre = precision_score(y_train, y_pred)
rec = recall_score(y_train, y_pred)

# Calculate F1 score
f1 = f1_score(y_train, y_pred)

# Calculate AUC-ROC score
y_pred_prob = grid_rf.predict_proba(X_train)[:, 1]  # Get the predicted probabilities for class 1
auc_roc = roc_auc_score(y_train, y_pred_prob)

print(f"For the full training data, the scores are:\n Accuracy: {acc}, Precision: {pre}, Recall: {rec}, F1 Score: {f1}, AUC-ROC Score: {auc_roc}")


# In[152]:


# Access the best estimator (Pipeline)
best_pipeline = grid_rf.best_estimator_

# Access the Random Forest model inside the pipeline
best_rf_model = best_pipeline.named_steps['randomforestclassifier']

# Access feature importances
feature_importances = best_rf_model.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)


# In[153]:


best_model = grid_rf.best_estimator_
best_model.fit(X_train, y_train)


# ## 4.2) Second Model

# In[154]:


rf = RandomForestClassifier()

pipe_rf2 = make_pipeline(preprocessor, rf)

grid_rf2 = GridSearchCV(pipe_rf2, param_grid=param_grid_rf, cv=10, scoring="recall")

grid_rf2.fit(X2_train, y2_train)


# In[155]:


print("Best score: ", grid_rf2.best_score_)
print("Best params: ", grid_rf2.best_params_)

y2_pred = grid_rf2.predict(X_train)
acc2 = accuracy_score(y2_train, y2_pred)
pre2 = precision_score(y2_train, y2_pred)
rec2 = recall_score(y2_train, y2_pred)

# Calculate F1 score
f12 = f1_score(y2_train, y2_pred)

# Calculate AUC-ROC score
y2_pred_prob = grid_rf2.predict_proba(X2_train)[:, 1]  # Get the predicted probabilities for class 1
auc_roc2 = roc_auc_score(y2_train, y2_pred_prob)

print(f"For the full training data, the scores are:\n Accuracy: {acc2}, Precision: {pre2}, Recall: {rec2}, F1 Score: {f12}, AUC-ROC Score: {auc_roc2}")


# In[156]:


# Access the best estimator (Pipeline)
best_pipeline2 = grid_rf2.best_estimator_

# Access the Random Forest model inside the pipeline
best_rf_model2 = best_pipeline2.named_steps['randomforestclassifier']

# Access feature importances
feature_importances2 = best_rf_model2.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df2 = pd.DataFrame({'Feature': X2_train.columns, 'Importance': feature_importances2})
feature_importance_df2 = feature_importance_df2.sort_values(by='Importance', ascending=False)
print(feature_importance_df2)


# In[157]:


dummy = DummyClassifier()
dummy.fit(X_train, y_train)
y_true = y_test.copy()
y_pred = dummy.predict(X_test)
baseline = accuracy_score(y_true, y_pred)
print(f"The baseline to beat is {baseline}")


# ## Confusion Matrixes

# In[159]:


y_predb = grid_rfb.predict(X_train)
cm = confusion_matrix(y_train, y_predb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
plt.show()


# In[160]:


y_pred = grid_rf.predict(X_train)
cm = confusion_matrix(y_train, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
plt.show()


# In[161]:


y2_pred = grid_rf2.predict(X2_train)
cm = confusion_matrix(y2_train, y2_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
plt.show()


# # 5) Final Evaluation

# In[162]:


y_pred_f_nt = best_model_nt.predict(X_test)

accfn = accuracy_score(y_test, y_pred_f_nt)
prefn = precision_score(y_test, y_pred_f_nt)
recfn = recall_score(y_test, y_pred_f_nt)

# Calculate F1 score
f1fn = f1_score(y_test, y_pred_f_nt)

# Calculate AUC-ROC score
y_pred_prob_f_nt = best_model_nt.predict_proba(X_test)[:, 1]  # Get the predicted probabilities for class 1
auc_rocfn = roc_auc_score(y_test, y_pred_prob_f_nt)

print(f"The model score on test data are:\n Accuracy: {accfn}, Precision: {prefn}, Recall: {recfn}, F1 Score: {f1fn}, AUC-ROC Score: {auc_rocfn}")


# In[163]:


cm = confusion_matrix(y_test, y_pred_f_nt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
plt.show()


# In[164]:


y_pred_f = best_model.predict(X_test)

accf = accuracy_score(y_test, y_pred_f)
pref = precision_score(y_test, y_pred_f)
recf = recall_score(y_test, y_pred_f)

# Calculate F1 score
f1f = f1_score(y_test, y_pred_f)

# Calculate AUC-ROC score
y_pred_prob_f = best_model.predict_proba(X_test)[:, 1]  # Get the predicted probabilities for class 1
auc_rocf = roc_auc_score(y_test, y_pred_prob_f)

print(f"The model score on test data are:\n Accuracy: {accf}, Precision: {pref}, Recall: {recf}, F1 Score: {f1f}, AUC-ROC Score: {auc_rocf}")


# In[168]:


cm = confusion_matrix(y_test, y_pred_f)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
plt.savefig(fname="matrix_final.png", format="png", transparent=True)
plt.show()


# In[166]:


y_test.value_counts(normalize=True)


# In[167]:


y_train.value_counts(normalize=True)


# In[ ]:




