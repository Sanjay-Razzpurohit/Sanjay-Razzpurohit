#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[54]:


#reding file
customer_churn = pd.read_csv("customer_churn[1].csv") 


# In[55]:


#finding the first few rows
customer_churn.head()


# In[56]:


customer_churn.info()


# In[57]:


#Extracting 5th column
customer_5=customer_churn.iloc[:,4] 
customer_5.head()


# In[58]:


#Extracting 15th column
customer_15=customer_churn.iloc[:,14] 
customer_15.head()


# In[61]:


#'Extracting male senior citizen with payment method-> electronic check'
senior_male_electronic=customer_churn[(customer_churn['gender']=='Male') & (customer_churn['SeniorCitizen']==1) & (customer_churn['PaymentMethod']=='Electronic check')]
senior_male_electronic.head()


# In[10]:


#tenure>70 or monthly charges>100
customer_total_tenure=customer_churn[(customer_churn['tenure']>70) | (customer_churn['MonthlyCharges']>100)]
customer_total_tenure.head()


# In[11]:


#cotract is 'two year', payment method is 'Mailed Check', Churn is 'Yes'
two_mail_yes=customer_total_tenure=customer_churn[(customer_churn['Contract']=='Two year') & (customer_churn['PaymentMethod']=='Mailed check') & (customer_churn['Churn']=='Yes')]
two_mail_yes


# In[12]:


#Extracting 333 random records
customer_333=customer_churn.sample(n=333)
customer_333.head()


# In[13]:


len(customer_333)


# In[14]:


#count of levels of churn column
customer_churn['Churn'].value_counts()


# In[15]:


#-------------------------------Data Visualization------------------#


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


#bar-plot for 'InternetService' column
plt.bar(customer_churn['InternetService'].value_counts().keys().tolist(),customer_churn['InternetService'].value_counts().tolist(),color='orange')
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count of categories')
plt.title('Distribution of Internet Service')


# In[18]:


#histogram for 'tenure' column
plt.hist(customer_churn['tenure'],color='green',bins=30)
plt.title('Distribution of tenure')


# In[19]:


#scatterplot 
plt.scatter(x=customer_churn['tenure'],y=customer_churn['MonthlyCharges'],color='brown')
plt.xlabel('Tenure of Customer')
plt.ylabel('Monthly Charges of Customer')
plt.title('Tenure vs Monthly Charges')


# In[20]:


#Box-plot
customer_churn.boxplot(column='tenure',by=['Contract'])


# In[21]:


#-----------------------Linear Regresssion----------------------


# In[22]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[23]:


x=pd.DataFrame(customer_churn['tenure'])
y=customer_churn['MonthlyCharges']


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[25]:


#building the model
from sklearn.linear_model import LinearRegression
simpleLinearRegression = LinearRegression()
simpleLinearRegression.fit(x_train,y_train)


# In[26]:


#predicting the values
y_pred = simpleLinearRegression.predict(x_test)


# In[27]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_pred, y_test)
rmse = np.sqrt(mse)
rmse


# In[28]:


#----------------------------------Logistic Regression-------------------------------


# In[29]:


x=pd.DataFrame(customer_churn['MonthlyCharges'])
y=customer_churn['Churn']


# In[30]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.65,random_state=0)


# In[31]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[32]:


y_pred = logmodel.predict(x_test)


# In[33]:


from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_pred,y_test),accuracy_score(y_pred,y_test)


# In[34]:


#--------------Multiple logistic regression-------------------


# In[35]:


x=pd.DataFrame(customer_churn.loc[:,['MonthlyCharges','tenure']])
y=customer_churn['Churn']


# In[36]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.80,random_state=0)


# In[37]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[38]:


y_pred = logmodel.predict(x_test)


# In[39]:


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[40]:


#---------------decision tree---------------


# In[41]:


x=pd.DataFrame(customer_churn['tenure'])
y=customer_churn['Churn']


# In[42]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)  


# In[43]:


from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(x_train, y_train)  


# In[44]:


y_pred = classifier.predict(x_test)  


# In[45]:


from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(y_test, y_pred))   
print(accuracy_score(y_test, y_pred))  


# In[46]:


#--------------random forest---------------------


# In[47]:


x=customer_churn[['tenure','MonthlyCharges']]
y=customer_churn['Churn']


# In[48]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)  


# In[49]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)


# In[50]:


y_pred=clf.predict(x_test)


# In[52]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)


# In[ ]:





# In[ ]:




