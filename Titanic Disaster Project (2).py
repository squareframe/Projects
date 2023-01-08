#!/usr/bin/env python
# coding: utf-8

# In[197]:


#Importing All Required Libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from warnings import filterwarnings
filterwarnings(action='ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[198]:


##Loading Datasets
df=pd.read_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/titanic_train.csv")


# In[199]:


df.head()


# In[200]:


## Statistical Info
df.describe()


# In[201]:


## datatype info
df.info()


# ### Exploratory Data Analysis

# In[202]:


## Catagorical acttributes
sns.countplot(df['Survived'])


# In[203]:


sns.countplot(df['Pclass'])


# In[204]:


sns.countplot(df['Sex'])


# In[205]:


sns.countplot(df['SibSp'])


# In[206]:


sns.countplot(df['Parch'])


# In[207]:


sns.countplot(df['Embarked'])


# In[208]:


## numerical attributes
sns.distplot(df['Age'])


# ## Majority of the passangers are between 20-30

# In[209]:


sns.distplot(df['Fare'])


# In[210]:


class_fare = df.pivot_table(index = 'Pclass', values= 'Fare')
class_fare.plot(kind = 'bar')
plt.xlabel('Pclass')
plt.ylabel('Avg. Fare')
plt.xticks(rotation=0)
plt.show()


# In[211]:


class_fare = df.pivot_table(index = 'Pclass', values= 'Fare' , aggfunc=np.sum)
class_fare.plot(kind = 'bar')
plt.xlabel('Pclass')
plt.ylabel('Avg. Fare')
plt.xticks(rotation=0)
plt.show()


# In[212]:


sns.barplot(data=df, x='Pclass', y='Fare', hue='Survived')


# ###                Might be more people are survived from the first class

# In[213]:


sns.barplot(data=df, x='Survived', y='Fare', hue='Pclass')


# ## Data Preprocessing

# In[214]:


df.tail()


# In[215]:


## find the null values
df.isnull().sum()


# In[216]:


## drop the column
df = df.drop(columns = ['Cabin'], axis = 1)


# In[217]:


df['Age'].mean()


# In[218]:


## fill missing values using mean of that column Numerical Column
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())


# In[219]:


df['Embarked'].mode()[0]


# In[220]:


## fill missing values using mode of that column Categorical Column

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


# ## Log transformation for uniform data distribution

# In[221]:


sns.distplot(df['Fare'])


# In[222]:


df['Fare'] = np.log(df['Fare']+1)


# In[223]:


sns.distplot(df['Fare'])


# ### It is going almost in the uniform distribution

# ## Correlation matrix to see which input attributes affects the Column

# In[224]:


corr = df.corr()
plt.figure(figsize=(15,9))
sns.heatmap(corr, annot=True)


# In[225]:


df.head()


# In[226]:


## Feature Selection
column_train=['Age','Pclass','SibSp','Parch','Fare','Sex','Embarked']
## #training values
X=df[column_train]
## target value
Y=df['Survived']


# ## Label Encoding

# In[227]:


from sklearn.preprocessing import LabelEncoder
cols = ['Sex', 'Embarked']
le = LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])
df.head()


# ## Train-Test Split

# ## Model Training

# In[228]:


##Training Testing and Spliting the model
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)
    


# ## Logistic Regression

# In[229]:


## Using LogisticRegression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,Y_pred))


# ## Confusion matrix

# In[230]:


## Confusion Matrix
from sklearn.metrics import accuracy_score,confusion_matrix
confusion_mat = confusion_matrix(Y_test,Y_pred)
print(confusion_mat)


# ##  Support vector Classifier

# In[231]:


## Using Support Vector
from sklearn.svm import SVC
model1 = SVC()
model1.fit(X_train,Y_train)

pred_y = model1.predict(X_test)

from sklearn.metrics import accuracy_score
print("Acc=",accuracy_score(Y_test,pred_y))


# In[232]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat = confusion_matrix(Y_test,pred_y)
print(confusion_mat)
print(classification_report(Y_test,pred_y))


# ## KNeighborsClassifier

# In[233]:


## Using KNN Neighbors
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train,Y_train)
y_pred2 = model2.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred2))


# In[234]:



from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat = confusion_matrix(Y_test,y_pred2)
print(confusion_mat)
print(classification_report(Y_test,y_pred2))


# ## GaussianNB

# In[235]:


## Using GaussianNB
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(X_train,Y_train)
y_pred3 = model3.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred3))


# In[236]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat = confusion_matrix(Y_test,y_pred3)
print(confusion_mat)
print(classification_report(Y_test,y_pred3))


# ## DecisionTreeClassifier

# In[237]:


## Using Decision Tree
from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier(criterion='entropy',random_state=7)
model4.fit(X_train,Y_train)
y_pred4 = model4.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred4))


# In[238]:



from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat = confusion_matrix(Y_test,y_pred4)
print(confusion_mat)
print(classification_report(Y_test,y_pred4))


# In[239]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines', 'Naive Bayes','KNN' ,'Decision Tree'],
    'Score': [0.75,0.66,0.76,0.66,0.74]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# # Hence I will use Naive Bayes algorithms for training my model.
# 

# In[ ]:




