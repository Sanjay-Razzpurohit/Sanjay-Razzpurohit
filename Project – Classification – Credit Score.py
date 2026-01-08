#!/usr/bin/env python
# coding: utf-8
# You are working as a data scientist in a
global finance company. Over the
years, the company has collected basic
bank details and gathered a lot of
credit-related information. The
management wants to build an
intelligent system to segregate the
people into credit score brackets to
reduce the manual efforts. Given a
person’s credit-related information,
build a machine learning model that
can classify the credit score.
# We start off this project by importing all the necessary
# libraries that will be required for the process.

# # Data Loading

# In[37]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[38]:


df=pd.read_csv('credit_score.csv')


# In[39]:


df.head(5)


# # Data Cleaning

# In[40]:


#Loading the data and removing unnecessary column from the dataframe
df=df.drop(columns=["ID","Customer_ID","Name","SSN","Type_of_Loan","Credit_History_Age"])
df.head(5)


# Checking the shape of a dataframe and datatypes of all columns
# along with calculating the statistical data.

# In[41]:


df.shape


# In[42]:


df.info()


# In[43]:


df.describe()


# In[44]:


#ing out the missing values in a dataframe 
df.isnull().sum()


# Replacing the special
# characters with empty
# string or with null
# values according to
# the data and
# converting it into int
# or float datatype. Also,
# Converting the
# categorical values of
# some columns into
# integer values.

# In[45]:


import numpy as np
import pandas as pd

# ---------- AGE ----------
df["Age"] = df["Age"].str.replace("_", "", regex=False)
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

# ---------- OCCUPATION ----------
df["Occupation"] = df["Occupation"].replace("_______", np.nan)

# ---------- ANNUAL INCOME ----------
df["Annual_Income"] = df["Annual_Income"].str.replace("_", "", regex=False)
df["Annual_Income"] = pd.to_numeric(df["Annual_Income"], errors="coerce")

# ---------- NUMBER OF LOANS ----------
df["Num_of_Loan"] = df["Num_of_Loan"].str.replace("_", "", regex=False)
df["Num_of_Loan"] = pd.to_numeric(df["Num_of_Loan"], errors="coerce")

# ---------- DELAYED PAYMENTS ----------
df["Num_of_Delayed_Payment"] = df["Num_of_Delayed_Payment"].str.replace("_", "", regex=False)
df["Num_of_Delayed_Payment"] = pd.to_numeric(df["Num_of_Delayed_Payment"], errors="coerce")

# ---------- CREDIT SCORE ----------
df["Credit_Score"] = df["Credit_Score"].map({
    "Poor": 0,
    "Standard": 1,
    "Good": 2
})

# ---------- MONTHLY BALANCE ----------
df["Monthly_Balance"] = df["Monthly_Balance"].str.replace("_", "", regex=False)
df["Monthly_Balance"] = pd.to_numeric(df["Monthly_Balance"], errors="coerce")

# ---------- PAYMENT BEHAVIOUR ----------
df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", np.nan)

# ---------- AMOUNT INVESTED MONTHLY ----------
df["Amount_invested_monthly"] = df["Amount_invested_monthly"].str.replace("_", "", regex=False)
df["Amount_invested_monthly"] = pd.to_numeric(df["Amount_invested_monthly"], errors="coerce")

# ---------- PAYMENT OF MIN AMOUNT ----------
df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", "No")
df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].map({
    "Yes": 1,
    "No": 0
})

# ---------- OUTSTANDING DEBT ----------
df["Outstanding_Debt"] = df["Outstanding_Debt"].str.replace("_", "", regex=False)
df["Outstanding_Debt"] = pd.to_numeric(df["Outstanding_Debt"], errors="coerce")

# ---------- CREDIT MIX ----------
df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan)
df["Credit_Mix"] = df["Credit_Mix"].map({
    "Bad": 0,
    "Standard": 1,
    "Good": 2
})

# ---------- CHANGED CREDIT LIMIT ----------
df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].replace("_", np.nan)
df["Changed_Credit_Limit"] = pd.to_numeric(df["Changed_Credit_Limit"], errors="coerce")


# In[46]:


df.info()

After replacing the
special characters with
null value. The new
missing value is shown
in the figure. Here
Forward and backward
filling method is used
to fill the missing
values.
# In[47]:


df.isnull().sum()


# In[51]:


df.head()


# In[48]:


df=df.fillna(method="ffill")


# In[50]:


df.head()


# In[52]:


df=df.fillna(method="bfill")


# In[53]:


df.head()


# In[56]:


df.columns


# In[57]:


df.isnull().sum()

removing outliers
from age since all
other columns
values are
relevant
# In[58]:


sns.boxplot(df['Age'])
plt.xlabel("Age")
plt.ylabel("count")
plt.show()


# In[59]:


col_names=["Age"]
Q1=df.Age.quantile(0.25)
Q3=df.Age.quantile(0.75)
IQR=Q3-Q1
data=df[(df.Age>=Q1-1.5*IQR) & (df.Age<= Q3+1.5*IQR)]
sns.boxplot(data["Age"])
plt.xlabel("Age")
plt.ylabel("count")
plt.show()


# In[60]:


data.head(5)

# Performing One Hot Encoding for
categorical features of a dataframe
# In[61]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Month']=le.fit_transform(df["Month"])
df['Occupation']=le.fit_transform(df['Occupation'])
df['Payment_Behaviour']=le.fit_transform(df["Payment_Behaviour"])
df.info()


# # Feature Selection 

# In[62]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
col_list = []
for col in df.columns:
    if ((df[col].dtype != 'object') & (col != 'Credit_Score')):
        col_list.append(col)

X = df[col_list]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]
print(vif_data)


# Selecting the features using VIF. VIF should be less
# than 5. Here, all features have VIF value less than
# 5, So we will select all the features. 

# # Logistic Regression
# 

# In[65]:


X=df.drop(columns=["Credit_Score"])
y=df["Credit_Score"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)


pd.DataFrame({"actual_value":y_test,"predicted_value":y_pred})


# In[66]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100


# The accuracy of the logistic regression model is
# 61.8 percentage

# # Decision Tree

# In[67]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred= dt.predict(x_test)
pd.DataFrame({"actual_value":y_test,"predicted_value":y_pred})


# In[68]:


accuracy_score(y_test,y_pred)*100


# The accuracy of the decision tree model is
# 69.7 percentage

# # Hyperparameter Tuning on Decision Tree

# In[69]:


from sklearn.model_selection import GridSearchCV
parameters = {'max_features': ['log2', 'sqrt','auto'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10, 50],
              'min_samples_split': [2, 3, 50, 100],
              'min_samples_leaf': [1, 5, 8, 10]
             }

grid_obj = GridSearchCV(dt, parameters)
grid_obj = grid_obj.fit(x_train, y_train)
dt = grid_obj.best_estimator_
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
acc_dt = round(accuracy_score(y_test, y_pred) * 100, 2)
print('Accuracy of Decision Tree model : ', acc_dt )


# The accuracy of the decision tree model 70.3 percentage

# # Random Forest

# In[25]:


from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
accuracy_score(y_test,y_pred)
pd.DataFrame({"Actual_Value":y_test,"Predicted Value":y_pred})


# In[27]:


accuracy_score(y_test,y_pred)*100


# The accuracy of the random forest model is
# 79.7 percentage

# In[ ]:




