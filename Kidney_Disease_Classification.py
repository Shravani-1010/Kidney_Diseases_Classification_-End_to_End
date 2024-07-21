#!/usr/bin/env python
# coding: utf-8

# # KIDNEY DISEASE CLASSIFICATION

# In[87]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings 
warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')
sns.set()
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df= pd.read_csv("Kidney_disease.csv")


# In[3]:


df.head()


# In[4]:


df['classification'].value_counts()


# In[5]:


df.shape


# In[6]:


df.drop('id',axis=1,inplace=True)


# In[7]:


df.head()


# # bp - blood pressure
# # sg - specific gravity
# # al- albumin
# # su - sugar
# # rbc - red blood cells
# # pc - pus cell
# # pcc - pus cell clumps
# # bgr - blood glucose random
# # bu - blood urea
# # sc - serum creatinie
# # sod - sodium
# # pot - potassium
# # hemo - hemoglobin
# # pcv - packed cell volume
# # wbcc - white blood cell count
# # rbcc - red blood cell count
# # htn - hypertension
# # dm - diabetes mellitus
# #  cad - coronary artey disease
# # appet - appetite
# # pe - pedal edema
# # ane - anemia
# # class - ckd or not ckd

# In[8]:


df.columns = ['age', 'blood_pressure','specific_gravity','albumin','sugar',
              'red_blood_cells','pus_cell','pus_cell_clumps',
              'bacteria','blood_glucose_random','blood_urea','serum_creatinie',
             'sodium','potassium','hemoglobin','packed_cell_volume',
             'white_blood_cell_count','red_blood_cell_count','hypertension',
             'diabetes_mellitus','coronary_artey_disease','appetite',
              'pedal_edema','anemia','class']


# In[9]:


df.head()


# In[10]:


df.describe()


# In[11]:


df.info()


# In[12]:


df['packed_cell_volume']= pd.to_numeric(df['packed_cell_volume'], errors ='coerce')
df['white_blood_cell_count']= pd.to_numeric(df['white_blood_cell_count'], errors ='coerce')
df['red_blood_cell_count']= pd.to_numeric(df['red_blood_cell_count'], errors ='coerce')


# In[13]:


df.info()


# In[14]:


cat_col = [col for col in df.columns if df[col].dtype=='object']
num_col =[col for col in df.columns if df[col].dtype != 'object']


# In[15]:


cat_col


# In[16]:


num_col


# In[17]:


for col in cat_col:
    print(f"{col} has {df[col].unique()}")


# In[18]:


df['diabetes_mellitus'].replace(to_replace={'\tno': 'no','\tyes': 'yes'},inplace =True)


# In[19]:


df['coronary_artey_disease'].replace(to_replace={'\tno': 'no'},inplace = True)


# In[20]:


df['class'] = df['class'].replace(to_replace={'ckd\t':'ckd','notckd':'not ckd'})


# In[21]:


col = ['diabetes_mellitus','coronary_artey_disease','class']
for col in cat_col:
    print(f"{col} has {df[col].unique()}")


# In[22]:


df['diabetes_mellitus'].replace(to_replace={'\tno': 'no','\tyes': 'yes',' yes' : 'yes'},inplace =True)


# In[23]:


col = ['diabetes_mellitus','coronary_artey_disease','class']
for col in cat_col:
    print(f"{col} has {df[col].unique()}")


# In[24]:


# convert target column into numerical one 


# In[ ]:





# In[25]:


df['class'] = df['class'].map({'ckd': 0, 'not ckd': 1})
df['class'] = pd.to_numeric(df['class'], errors ='coerce')


# In[26]:


col = ['diabetes_mellitus','coronary_artey_disease','class']
for col in col:
    print(f"{col} has {df[col].unique()}")


# In[27]:


print(df['class'].unique())


# In[28]:


col = ['diabetes_mellitus','coronary_artey_disease','class']
for col in col:
    print(f"{col} has {df[col].unique()}")


# In[29]:


plt.figure(figsize = (20,15))
plotnumber = 1

for column in num_col:
    if plotnumber <= 14:
        ax = plt.subplot(3, 5,plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column)
    plotnumber += 1
plt.tight_layout()
plt.show()


# In[30]:


plt.figure(figsize = (20,30))
plotnumber = 1

for column in cat_col:
    if plotnumber <= 14:
        ax = plt.subplot(3, 5, plotnumber)
        sns.countplot(data=df, x=column, palette='rocket', ax=ax)
        plt.xlabel(column)
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[31]:


df.corr()


# In[32]:


plt.figure(figsize = (15,18))
sns.heatmap(df.corr, annot= True, linewight= 2 , linecolor ='lightgray')
plt.show()


# In[33]:


corr = df.corr()

# Set up the figure size
plt.figure(figsize=(15, 18))

# Create the heatmap
sns.heatmap(corr, annot=True, linewidths=2, linecolor='lightgray', cmap='coolwarm')

# Show the plot
plt.show()


# In[34]:


#EDA


# In[35]:


def violin_plot(col):
    fig = px.violin(df, y=col, x='class', color='class', box=True, template='plotly_dark')
    fig.show()

def kde_plot(col):
    grid = sns.FacetGrid(df, hue='class', height=6, aspect=2)
    grid.map(sns.kdeplot, col)
    grid.add_legend()
    plt.show()  # Use plt.show() to display the plot
    
def scatter_plot(col1, col2):
    fig = px.scatter(df, x=col1, y=col2, color='class', template='plotly_dark')
    fig.show()


# In[36]:


violin_plot('red_blood_cell_count')


# In[37]:


# DATA PREPROCESSING


# In[38]:


# checking for missing value
df.isnull().sum().sort_values(ascending=False)


# In[39]:


df[num_col].isnull().sum()


# In[40]:


df[cat_col].isnull().sum()


# In[41]:


df.head()


# In[42]:


#TWO METHODS
# RANDOM _SAMPLING  -> HIGHER NULL VALUES 
# MEAN / MODE -> LOWER NULL VALUES


# In[43]:


def random_sampling(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample

def impute_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)


# In[44]:


# random sampling for numerical value
for col in num_col:
    random_sampling(col)
    


# In[45]:


df[num_col].isnull().sum()


# In[46]:


random_sampling('red_blood_cells')
random_sampling('pus_cell')

for col in cat_col:
    impute_mode(col)


# In[47]:


# random sampling for numerical value
for col in cat_col:
    random_sampling(col)
    


# In[48]:


df[cat_col].isnull().sum()


# In[49]:


# Feature Encoding


# In[50]:


for col in cat_col:
    print(f"{col}has {df[col].nunique}")


# In[51]:


#label_encoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in cat_col:
    df[col] = le.fit_transform(df[col])


# In[52]:


df.head()


# In[53]:


# MODEL BUILDING


# In[54]:


x = df.drop("class", axis =1)
y = df['class']


# In[55]:


x


# In[56]:


y


# In[57]:


#split


# In[58]:


from sklearn.model_selection import train_test_split

X_train,X_test, Y_train, Y_test = train_test_split(x,y,test_size= 0.2, random_state = 0)


# In[59]:


#KNN


# In[60]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[61]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)

knn_acc = accuracy_score(Y_test, knn.predict(X_test))
print(f"Training Accuracy of KNN is {accuracy_score(Y_train, knn.predict(X_train))}")
print(f"Testing Accuracy of KNN is {accuracy_score(Y_test, knn.predict(X_test))}")

print(f"Confusion Matrix of KNN is \n {confusion_matrix(Y_test, knn.predict(X_test))}\n")
print(f"Classification Report of KNN is \n{classification_report(Y_test, knn.predict(X_test))}")


# In[62]:


#Decision tree


# In[63]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(X_train, Y_train)
dtc_acc = accuracy_score(Y_test, dtc.predict(X_test))
print(f"Training Accuracy of dtc is {accuracy_score(Y_train, dtc.predict(X_train))}")
print(f"Testing Accuracy of dtc is {accuracy_score(Y_test, knn.predict(X_test))}")

print(f"Confusion Matrix of dtc is \n {confusion_matrix(Y_test, dtc.predict(X_test))}\n")
print(f"Classification Report of dtc is \n{classification_report(Y_test, dtc.predict(X_test))}")


# In[64]:


# Hyper paramneter Tuning


# In[71]:


# Hyper Parameter Tuning
from sklearn.model_selection import GridSearchCV

GRID_PARAMETER = {
    'criterion':['gini','entropy'],
    'max_depth':[3,5,7,10],
    'splitter':['best','random'],
    'min_samples_leaf':[1,2,3,5,7],
    'min_samples_split':[1,2,3,5,7],
    'max_features':['auto', 'sqrt', 'log2']
}

grid_search_dtc = GridSearchCV(dtc, GRID_PARAMETER, cv=5, n_jobs=-1, verbose = 1)
grid_search_dtc.fit(X_train, Y_train)


# In[66]:


print(grid_search_dtc.best_params_)
print(grid_search_dtc.best_score_)


# In[67]:


dtc = grid_search_dtc.best_estimator_

dtc_acc = accuracy_score(Y_test, dtc.predict(X_test))
print(f"Training Accuracy of DTC is {accuracy_score(Y_train, dtc.predict(X_train))}")
print(f"Testing Accuracy of DTC is {accuracy_score(Y_test, dtc.predict(X_test))}")

print(f"Confusion Matrix of DTC is \n {confusion_matrix(Y_test, dtc.predict(X_test))}\n")
print(f"Classification Report of DTC is \n{classification_report(Y_test, dtc.predict(X_test))}")


# In[68]:


#Random Forest Classifier 


# In[69]:


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(criterion = "gini", max_depth = 10, max_features="sqrt", min_samples_leaf= 1, min_samples_split= 7, n_estimators = 400)
rand_clf.fit(X_train, Y_train)


# In[72]:


rand_clf_acc = accuracy_score(Y_test, rand_clf.predict(X_test))
print(f"Training Accuracy of Random Forest is {accuracy_score(Y_train, rand_clf.predict(X_train))}")
print(f"Testing Accuracy of Random Forest is {accuracy_score(Y_test, rand_clf.predict(X_test))}")

print(f"Confusion Matrix of Random Forest is \n {confusion_matrix(Y_test, rand_clf.predict(X_test))}\n")
print(f"Classification Report of Random Forest is \n{classification_report(Y_test, rand_clf.predict(X_test))}")


# In[73]:


# LOGISTICS REGRESSION


# In[74]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)


# In[75]:


lr_acc = accuracy_score(Y_test, lr.predict(X_test))
print(f"Training Accuracy of LR is {accuracy_score(Y_train, lr.predict(X_train))}")
print(f"Testing Accuracy ofLR is {accuracy_score(Y_test, lr.predict(X_test))}")

print(f"Confusion Matrix of LR is \n {confusion_matrix(Y_test, lr.predict(X_test))}\n")
print(f"Classification Report of LR is \n{classification_report(Y_test,lr.predict(X_test))}")


# In[76]:


# SVM


# In[77]:


# SVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm  = SVC(probability=True)

parameter = {
    'gamma':[0.0001, 0.001, 0.01, 0.1],
    'C':[0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20]
}

grid_search = GridSearchCV(svm, parameter)
grid_search.fit(X_train, Y_train)


# In[78]:


print(grid_search.best_params_)
print(grid_search.best_score_)


# In[79]:


svm  = SVC(gamma = 0.0001, C  = 15, probability=True)
svm.fit(X_train, Y_train)


# In[80]:


svm_acc = accuracy_score(Y_test, svm.predict(X_test))
print(f"Training Accuracy of SVC is {accuracy_score(Y_train, svm.predict(X_train))}")
print(f"Testing Accuracy of SVC is {accuracy_score(Y_test, svm.predict(X_test))}")

print(f"Confusion Matrix of SVC is \n {confusion_matrix(Y_test, svm.predict(X_test))}\n")
print(f"Classification Report of SVC is \n{classification_report(Y_test, svm.predict(X_test))}")


# In[81]:


#MODEL COMPARISIONS


# In[82]:


# Model Comparison

models = pd.DataFrame({
    'Model':['Logistic Regression', 'KNN', 'SVM', 'DT', 'Random Forest Classifier'],
    'Score':[lr_acc, knn_acc, svm_acc, dtc_acc, rand_clf_acc]
})

models.sort_values(by='Score', ascending = False)


# In[83]:


import pickle
model = rand_clf
pickle.dump(model, open("Kidney.pkl",'wb'))


# In[85]:


from sklearn import metrics
plt.figure(figsize=(8,5))
models = [
{
    'label': 'LR',
    'model': lr,
},
{
    'label': 'DT',
    'model': dtc,
},
{
    'label': 'SVM',
    'model': svm,
},
{
    'label': 'KNN',
    'model': knn,
},
{
    'label': 'RF',
    'model': rand_clf,
}
]
for m in models:
    model = m['model'] 
    model.fit(X_train, Y_train) 
    y_pred=model.predict(X_test) 
    fpr1, tpr1, thresholds = metrics.roc_curve(Y_test, model.predict_proba(X_test)[:,1])
    auc = metrics.roc_auc_score(Y_test,model.predict(X_test))
    plt.plot(fpr1, tpr1, label='%s - ROC (area = %0.2f)' % (m['label'], auc))

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity (False Positive Rate)', fontsize=12)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize=12)
plt.title('ROC - Kidney Disease Prediction', fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.savefig("roc_kidney.jpeg", format='jpeg', dpi=400, bbox_inches='tight')
plt.show()


# In[86]:


from sklearn import metrics
plt.figure(figsize=(8,5))
models = [
{
    'label': 'LR',
    'model': lr,
},
{
    'label': 'DT',
    'model': dtc,
},
{
    'label': 'SVM',
    'model': svm,
},
{
    'label': 'KNN',
    'model': knn,
},
{
    'label': 'RF',
    'model': rand_clf,
}
]
means_roc = []
means_accuracy = [100*round(lr_acc,4), 100*round(dtc_acc,4), 100*round(svm_acc,4), 100*round(knn_acc,4), 
                  100*round(rand_clf_acc,4)]

for m in models:
    model = m['model'] 
    model.fit(X_train, Y_train) 
    y_pred=model.predict(X_test) 
    fpr1, tpr1, thresholds = metrics.roc_curve(Y_test, model.predict_proba(X_test)[:,1])
    auc = metrics.roc_auc_score(Y_test,model.predict(X_test))
    auc = 100*round(auc,4)
    means_roc.append(auc)

print(means_accuracy)
print(means_roc)


n_groups = 7
means_accuracy = tuple(means_accuracy)
means_roc = tuple(means_roc)


fig, ax = plt.subplots(figsize=(8,5))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_accuracy, bar_width,
alpha=opacity,
color='mediumpurple',
label='Accuracy (%)')

rects2 = plt.bar(index + bar_width, means_roc, bar_width,
alpha=opacity,
color='rebeccapurple',
label='ROC (%)')

plt.xlim([-1, 8])
plt.ylim([45, 104])

plt.title('Performance Evaluation - Kidney Disease Prediction', fontsize=12)
plt.xticks(index, ('   LR', '   DT', '   SVM', '   KNN' , '   RF'), rotation=40, ha='center', fontsize=12)
plt.legend(loc="upper right", fontsize=10)
plt.savefig("PE_kidney.jpeg", format='jpeg', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:





# In[ ]:




