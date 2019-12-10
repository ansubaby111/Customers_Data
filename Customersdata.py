# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:19:26 2019

@author: Dell
"""

###################################################Required Packages#################################################################################
#####################################################################################################################################################
## Import the libraries


import pandas as pd
import numpy as np
from pandas.io.json import json_normalize 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

%matplotlib inline

data = pd.read_json('C:\\Users\\Dell\\Downloads\\customersdata.json', lines=True)
data = data.replace(np.nan, '', regex=True)

################### split dictionary to multiple columns

data1 = data['customer'].apply(pd.Series)

custom = data1[['customerEmail', 'customerPhone', 'customerDevice', 'customerIPAddress','customerBillingAddress']]
custom = custom.dropna(how='all')
custom = custom.join(data['fraudulent'])

data2 = []

for i in range(len(data['orders'])):
    dat2 = pd.DataFrame.from_dict(data['orders'][i])
    data2.append(dat2)
orders = pd.concat(data2,ignore_index=True)


data3 = []

for i in range(len(data['paymentMethods'])):
    dat3 = pd.DataFrame.from_dict(data['paymentMethods'][i])
    data3.append(dat3)
payments = pd.concat(data3,ignore_index=True)


data4 = []

for i in range(len(data['transactions'])):
    dat4 = pd.DataFrame.from_dict(data['transactions'][i])
    data4.append(dat4)
transac = pd.concat(data4,ignore_index=True)



###################################### Merging dataset

pay_transac = pd.merge(payments,transac,on='paymentMethodId',how="right")


order_pay_transac = pd.merge(orders,pay_transac,on="orderId",how="right")

order_pay_transac = order_pay_transac.rename(columns={'orderShippingAddress':'customerBillingAddress'})
Final = pd.merge(order_pay_transac,custom,on="customerBillingAddress",how="right")




################################### Recoding of variables
Final['fraudulent_new']=np.where(Final['fraudulent'].isin([True]),1,0)
Final['paymentMethodRegistrationFailure_new']=np.where(Final['paymentMethodRegistrationFailure'].isin([True]),1,0)
Final['transactionFailed_new']=np.where(Final['transactionFailed'].isin([True]),1,0)
def paymentMethodType_to_numeric(x):
    if x=='card':
        return 0 
    if x=='paypal':
        return 1
    if x=='apple pay':
        return 2
    if x=='bitcoin':
        return  3
Final['paymentMethodType_new'] = Final['paymentMethodType'].apply(paymentMethodType_to_numeric)
def orderState_to_numeric(x):
    if x=='fulfilled':
        return 0
    if x=='failed':
        return 1
    if x=='pending':
        return 2
Final['orderState_new']=Final['orderState'].apply(orderState_to_numeric)
def paymentMethodProvider_to_numeric(x):
    if x=='JCB 16 digit':
        return 0 
    if x=='VISA 16 digit':
        return 1
    if x=='JCB 15 digit':
        return 2
    if x=='VISA 13 digit':
        return  3
    if x=='American Express':
        return 4 
    if x=='Voyager':
        return 5
    if x=='Maestro':
        return 6
    if x=='Diners Club / Carte Blanche':
        return 7
    if x=='Discover':
        return 8
    if x=='Mastercard':
        return 9
Final['paymentMethodProvider_new']=Final['paymentMethodProvider'].apply(paymentMethodProvider_to_numeric)


################################### Summary Measures
Final1=Final[['orderAmount','transactionAmount']]
Final1.describe()


################################### Frequency of variables
Final['transactionFailed'].value_counts()
Final['fraudulent'].value_counts()
Final['paymentMethodProvider'].value_counts()
Final['paymentMethodRegistrationFailure'].value_counts()
Final['paymentMethodType'].value_counts()
Final['orderState'].value_counts()


################################### Bar graph
Final['fraudulent_new'].value_counts().plot(kind='bar')
Final['paymentMethodType_new'].value_counts().plot(kind='bar')
Final.groupby(['fraudulent_new','paymentMethodType_new']).size().plot(kind='bar')

Final['fraudulent_new'].value_counts().plot(kind='bar')
Final['transactionFailed_new'].value_counts().plot(kind='bar')
Final.groupby(['fraudulent_new','transactionFailed_new']).size().plot(kind='bar')

Final['fraudulent_new'].value_counts().plot(kind='bar')
Final['paymentMethodProvider_new'].value_counts().plot(kind='bar')
Final.groupby(['fraudulent_new','paymentMethodProvider_new']).size().plot(kind='bar')

Final['fraudulent_new'].value_counts().plot(kind='bar')
Final['paymentMethodRegistrationFailure_new'].value_counts().plot(kind='bar')
Final.groupby(['fraudulent_new','paymentMethodRegistrationFailure_new']).size().plot(kind='bar')

Final['fraudulent_new'].value_counts().plot(kind='bar')
Final['orderState_new'].value_counts().plot(kind='bar')
Final.groupby(['fraudulent_new','orderState_new']).size().plot(kind='bar')


################################### Box plot
sns.boxplot(x="fraudulent_new", y="transactionAmount",hue="fraudulent_new", data=Final)
plt.show()

sns.boxplot(x="fraudulent_new", y="orderAmount",hue="fraudulent_new", data=Final)
plt.show()


###################################  Chi-Square Test
crosstab = pd.crosstab(Final['fraudulent_new'], Final['paymentMethodType_new'])
stats.chi2_contingency(crosstab)

crosstab = pd.crosstab(Final['fraudulent_new'], Final['paymentMethodProvider_new'])
stats.chi2_contingency(crosstab)

crosstab = pd.crosstab(Final['fraudulent_new'], Final['transactionFailed_new'])
stats.chi2_contingency(crosstab)

crosstab = pd.crosstab(Final['fraudulent_new'], Final['orderState_new'])
stats.chi2_contingency(crosstab)

crosstab = pd.crosstab(Final['fraudulent_new'], Final['paymentMethodRegistrationFailure_new'])
stats.chi2_contingency(crosstab)


################################### Logistic Regression
X =(Final[['paymentMethodProvider_new','transactionFailed_new','paymentMethodRegistrationFailure_new','orderState_new','paymentMethodType_new','orderAmount','transactionAmount']])
glm_binom = sm.GLM(Final['fraudulent_new'],X,family=sm.families.Binomial(),missing='drop')
res=glm_binom.fit()
print(res.summary())