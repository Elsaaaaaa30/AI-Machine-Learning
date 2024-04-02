#!/usr/bin/env python
# coding: utf-8

# ## Data Collection and Preprocessing**

# Import library, metrics, and exclude warnings 

# In[7]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[8]:


#Import libraries and metrics
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,roc_curve, classification_report
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay


# In[9]:


pd.options.mode.chained_assignment = None


# ## Tesco Plc,Inc Historical Data**

# Download data from Yahoo Finance 

# In[10]:


tscodata = yf.download('TSCO', start="2014-01-01", end='2023-12-31')
tscodata


# ## Feature Engineering

# Data Reduction 

# In[11]:


tscodata = tscodata.dropna() 
tscoata = tscodata[['Volume','Open', 'High', 'Low', 'Close']]
tscodata


# Future Creation 

# In[12]:


tscodata['H-L'] = tscodata['High'] - tscodata['Low'] 
tscodata['O-C'] =tscodata['Close'] - tscodata['Open'] 

tscodata['2day MA'] = tscodata['Close'].shift(1).rolling(window = 2).mean() 
tscodata['20day MA'] = tscodata['Close'].shift(1).rolling(window = 20).mean() 
tscodata['200day MA'] = tscodata['Close'].shift(1).rolling(window = 200).mean() 

tscodata['Std_dev']= tscodata['Close'].rolling(5).std() 

tscodata['Price_Rise'] = np.where(tscodata['Close'].shift(-1) >tscodata['Close'], 1, 0)

tscodata = tscodata.dropna() 
tscodata


# ## EDA of Created Features

# Figure 1:Plots Close Price and Volume

# In[13]:


fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:green'
ax1.set_ylabel('Closing Price', color=color)
ax1.plot(tscodata['Close'], label='Close', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Volume', color=color)
ax2.plot(tscodata['Volume'], label='Volume', color=color)
ax2.tick_params(axis='y', labelcolor=color)


plt.title('Closing Price and Volume Relationship')
fig.tight_layout()
plt.show()


# Figure 2: Plot for 3day, 10day,30day MA 

# In[14]:


plt.figure(figsize=(10,8 ))

plt.plot(tscodata['2day MA'], label='2day MA')
plt.plot(tscodata['20day MA'], label='20day MA')
plt.plot(tscodata['200day MA'], label='200day MA')

plt.title('Moving Averages Over Time')
plt.ylabel('Moving Averages')
plt.legend()
plt.show()


# Figure 3: Plots for O-C,H-L and Std_dev

# In[15]:


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(tscodata['O-C'])
axes[0].set_title('O-C')
axes[1].plot(tscodata['H-L'])
axes[1].set_title('H-L')
axes[2].plot(tscodata['Std_dev'])
axes[2].set_title('Std_dev')

plt.show()


# Figure 4: 2day, 20day,200day MA histogram and density

# In[16]:


fig, axes = plt.subplots(1, 3, figsize=(15,3))  # 1 row, 3 columns

sns.histplot(data=tscodata, x="2day MA", kde=True, stat="density", ax=axes[0])
axes[0].set_title('2day MA')

sns.histplot(data=tscodata, x="20day MA", kde=True, stat="density", ax=axes[1])
axes[1].set_title('20day MA')

sns.histplot(data=tscodata, x="200day MA", kde=True, stat="density", ax=axes[2])
axes[2].set_title('200day MA')

plt.show()


# Figure 5: 2day, 20day,200day MA histogram and density

# In[17]:


chart = sns.FacetGrid(tscodata, col='Price_Rise')  
chart.map(sns.histplot, 'Close') 

chart = sns.FacetGrid(tscodata, col='Price_Rise')  
chart.map(plt.scatter, 'Close','Std_dev')  


# Fig 6: Descriptive Statistics of the Data set (description and information)

# In[18]:


tscodata.describe()


# In[19]:


tscodata.info()


# Fig 7: Correlation matrix of data variables

# In[20]:


corr_matrix = tscodata.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


# ## Machine Learning Classification Methods**

# Data Preprocessing

# In[21]:


#Set target variable y(Price_Rise 0 or 1 )and features x 
#(from H-L column to Std_dev column)
X = tscodata.iloc[:, 5:-1] 
Y = tscodata.iloc[:, -1]


# In[22]:


X


# In[23]:


Y #Price Rise


# In[24]:


# Split the data into training and testing sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, 
                                   shuffle=True)
X_test


# In[25]:


# Standardize the features (optional but can be beneficial for 
#logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[26]:


X_test


# # Cross Validation
# 
# 
# 
# ## Cross Validation for Extra Trees

# ## Extra Trees

# In[27]:


#Model
modelet = ExtraTreesClassifier(random_state=101)
# Train the model
modelet.fit(X_train, Y_train)
# Predict on the test set
Y_pred_et = modelet.predict(X_test)

print (classification_report(Y_test, Y_pred_et))


# ###  Figure 9. Extra Tree
# 

# ## Cross Validation Logistic Regression

# In[28]:


# Model
modellr = LogisticRegression(random_state=101)
# Train the model
modellr.fit(X_train, Y_train)
# Predict on the test set
Y_pred_lr = modellr.predict(X_test)

print (classification_report(Y_test,Y_pred_lr))


# ### Figure 8. Cross Validation for Logistic Regression

# ## Cross Validation for Logistic Regression

# In[29]:


accuracy_scores = cross_val_score(modellr, X, Y, cv=5, 
                scoring=make_scorer(accuracy_score))

# Print mean and standard deviation of accuracy
print(f"Mean Accuracy: {accuracy_scores.mean():.2f}")
print(f"Standard Deviation: {accuracy_scores.std():.2f}")


# ### Figure 10. Cross validation of Logistic Regression

# ## Prediction of Price Rise Using Logistic Regression on X_test Data

# In[30]:


# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, Y_pred_lr)

# Evaluate the modelet by means of a Confusion Matrix
matrix = ConfusionMatrixDisplay.from_estimator(modellr, X_test, Y_test)  
plt.title('Confusion Matrix')
plt.show(matrix)
plt.show()


# ### Figure 11. Confusion Matrix Logistic Regression

# In[31]:


#ROC Curve
log_disp = RocCurveDisplay.from_estimator(modellr, X_test, Y_test)


# In[32]:


#Importance of classifiers 
feature_names=X.columns
importance = modelet.feature_importances_ 
indices = np.argsort(importance)
range1 = range(len(importance[indices]))
plt.figure()
plt.title("LosgisticRegression Classifier Feature Importance")
plt.barh(range1,importance[indices])
plt.yticks(range1, feature_names[indices])
plt.ylim([-1, len(range1)])
plt.show()


# ### Figure 12. Roc Curve Logistic Regression

# ##  Cross Validation for Extra Tree

# In[33]:


accuracy_scores = cross_val_score(modelet, X, Y, cv=5, 
                scoring=make_scorer(accuracy_score))

# Mean and standard deviation of accuracy
print(f"Mean Accuracy: {accuracy_scores.mean():.2f}")
print(f"Standard Deviation Accuracy: {accuracy_scores.std():.2f}")


# ### Figure 11. Cross Validation of Extra Trees

# ## Prediction of Price Rise Using Extra Trees on X_test Data

# In[34]:


# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, Y_pred_et)

# Evaluate the modelet by means of a Confusion Matrix
matrix = ConfusionMatrixDisplay.from_estimator(modelet, X_test, Y_test)  
plt.title('Confusion Matrix')
plt.show(matrix)
plt.show()


# In[35]:


# ROC Curve
log_disp = RocCurveDisplay.from_estimator(modelet, X_test, Y_test)


# In[36]:


#Importance of classifiers 
feature_names=X.columns
importance = modelet.feature_importances_ 
indices = np.argsort(importance)
range1 = range(len(importance[indices]))
plt.figure()
plt.title("ExtraTreesClassifier Feature Importance")
plt.barh(range1,importance[indices])
plt.yticks(range1, feature_names[indices])
plt.ylim([-1, len(range1)])
plt.show()


# ## Market and Return Strategies

# ## Data Preprocessing

# In[37]:


#Create new column Y_pred
tscodata['Y_pred_lr'] = np.NaN
tscodata.iloc[(len(tscodata) - len(Y_pred_lr)):,-1] = Y_pred_lr
tscodata['Y_pred_et'] = np.NaN
tscodata.iloc[(len(tscodata) - len(Y_pred_et)):,-1] = Y_pred_et
trade_tscodata = tscodata.dropna()
trade_tscodata


# ## Computation of market returns 

# In[38]:


trade_tscodata['Tomorrows Returns'] = 0.
trade_tscodata['Tomorrows Returns'] = np.log(trade_tscodata['Close']/trade_tscodata['Close'].shift(1))
trade_tscodata['Tomorrows Returns'] = trade_tscodata['Tomorrows Returns'].shift(-1)
trade_tscodata


# ## Computation of strategy returns 

# In[39]:


#Strategy Returns based on Y_Pred
trade_tscodata['Strategy Returns lr'] = 0.
trade_tscodata['Strategy Returns lr'] = np.where(trade_tscodata['Y_pred_lr'] == True,
                                 trade_tscodata['Tomorrows Returns'], - trade_tscodata['Tomorrows Returns'])
trade_tscodata['Strategy Returns et'] = 0.
trade_tscodata['Strategy Returns et'] = np.where(trade_tscodata['Y_pred_et'] == True,
                                 trade_tscodata['Tomorrows Returns'], - trade_tscodata['Tomorrows Returns'])
trade_tscodata


# ## Cummulative Market and Strategies Returns

# ## Computation of cummulative market and strategy returns 

# In[40]:


trade_tscodata['Cumulative Market Returns'] = np.cumsum(trade_tscodata['Tomorrows Returns'])
trade_tscodata['Cumulative Strategy Returns lr'] = np.cumsum(trade_tscodata['Strategy Returns lr'])
trade_tscodata['Cumulative Strategy Returns et'] = np.cumsum(trade_tscodata['Strategy Returns et'])


# Fig 11: Plot of cummulative market and strategy returns based on Y_prediction

# In[41]:


plt.figure(figsize=(10,3))
plt.plot(trade_tscodata['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(trade_tscodata['Cumulative Strategy Returns lr'], color='g', label='Strategy Returns lr')
plt.plot(trade_tscodata['Cumulative Strategy Returns et'], color='b', label='Strategy Returns et')
plt.legend()
plt.show()


# ### Figure 13. Plot of cumulative market returns, and strategy returns based on Y_Prediction.

# In[ ]:




