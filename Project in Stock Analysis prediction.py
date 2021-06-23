#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from pandas import Series,DataFrame
import numpy as np


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


from datetime import datetime


# In[5]:


from pandas_datareader import DataReader


# In[6]:


from __future__ import division


# In[7]:


spy_list = ['TSLA','RYCEY','WMT','XIACF','BABA','NFLX','GOOG']


# In[8]:


end_t = datetime.now()
end_t


# In[9]:


start_t = datetime(end_t.year-1,end_t.month,end_t.day)
start_t


# In[10]:


for stocks in spy_list:
    globals()[stocks] = DataReader(stocks,'yahoo',start_t,end_t)


# In[11]:


RYCEY.head()


# In[12]:


RYCEY.describe()


# In[13]:


RYCEY.info()


# In[14]:


RYCEY['Adj Close'].plot(legend="True",figsize=(10,6))


# In[15]:


NFLX['Adj Close'].plot(legend="True",figsize=(10,6))


# In[16]:


RYCEY['Volume'].plot(figsize=(10,6),legend=True)


# In[17]:


ma_period = [10,50,200]
for ma in ma_period:
    c_name = "MA for {0} days".format(str(ma))
    RYCEY[c_name] = RYCEY['Adj Close'].rolling(window=ma,center=False).mean()


# In[18]:


RYCEY.tail()


# In[19]:


RYCEY[['Adj Close','MA for 10 days','MA for 50 days','MA for 200 days']].plot(subplots=True,figsize=(10,6))


# In[20]:


RYCEY[['Adj Close','MA for 10 days','MA for 50 days','MA for 200 days']].plot(subplots=False,figsize=(10,6))


# In[21]:


RYCEY['Daily Returns']=RYCEY['Adj Close'].pct_change()


# In[22]:


RYCEY.tail()


# In[55]:


RYCEY['Daily Returns'].plot(legend=True,figsize=(10,6),marker='o')


# In[24]:


sns.histplot(RYCEY['Daily Returns'].dropna(),bins=50,color='red')


# In[25]:


c_df = DataReader(spy_list,'yahoo',start_t,end_t)['Adj Close']


# In[26]:


c_df.tail()


# In[27]:


returns_df=c_df.pct_change()


# In[28]:


returns_df.tail()


# In[29]:


sns.jointplot(x='TSLA',y='GOOG',data=returns_df,kind='scatter',color='green')


# In[30]:


sns.pairplot(returns_df.dropna())


# In[31]:


sns.heatmap(returns_df.corr(),annot=True)


# In[32]:


returns=returns_df.dropna()


# In[33]:


plt.figure(figsize=(8,5))
plt.scatter(returns.mean(),returns.std(),s=25)
plt.xlabel('Expected Return')
plt.ylabel('Risk')
for label,x,y in zip(returns.columns,returns.mean(),returns.std()):
    plt.annotate(
    label,
    xy=(x,y),xytext=(-120,20),
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    arrowprops = dict(arrowstyle='->',connectionstyle = 'arc3,rad=-0.5'))


# In[34]:


rt,k=returns.dropna().describe(),[]
for cm in spy_list:
    k.append(round(rt[cm][2],4)*1000)
sl={'Company':spy_list,'s_d':k}
sl=pd.DataFrame(sl)
sns.barplot(x='Company',y='s_d',data=sl)


# In[35]:


sns.displot(RYCEY['Daily Returns'].dropna(),bins=50,color='orange')


# In[36]:


returns.head()


# In[37]:


#Using Pandas built in quantile method
returns['RYCEY'].quantile(0.05)


# In[38]:


returns.info()


# In[48]:


n_days = 365
dt = 1/365
mn = returns.mean()['NFLX']
sma = returns.std()['NFLX']


# In[49]:


def monte_carlo(s_price,days,mn,sma):
    price = np.zeros(days)
    price[0] = s_price
    shock = np.zeros(days)
    drift = np.zeros(days)
    for x in range(1,days):
        shock[x] = np.random.normal(loc=mn*dt,scale=sma*np.sqrt(dt))
        drift[x] = mn*dt
        #New price = Old price + Old price*(shock+drift)
        price[x] = price[x-1] + (price[x-1] * (drift[x]+shock[x]))
    return price


# In[50]:


NFLX.head()


# In[51]:


s_price = 448.559
for run in range(100):
    plt.plot(monte_carlo(s_price,n_days,mn,sma))
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Analysis for NetFlix')


# In[52]:


runs = 10000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = monte_carlo(s_price,n_days,mn,sma)[n_days-1]


# In[53]:


q = np.percentile(simulations,1)
plt.hist(simulations,bins=200)
plt.figtext(0.6,0.8,s="Start price: $%.2f" %s_price)
plt.figtext(0.6,0.7,"Mean final price: $%.2f" % simulations.mean())
plt.figtext(0.6,0.6,"VaR(0.99): $%.2f" % (s_price -q,))
plt.figtext(0.15,0.6, "q(0.99): $%.2f" % q)
plt.axvline(x=q, linewidth=4, color='r')
plt.title(u"Final price distribution for Netflix Stock after %s days" %n_days, weight='bold')


# In[43]:


mn = returns.mean()['GOOG']
sma = returns.std()['GOOG']


# In[44]:


GOOG.head()


# In[45]:


s_price=1408.0000
for run in range(100):
    plt.plot(monte_carlo(s_price,n_days,mn,sma))
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Analysis for Google')


# In[46]:


runs = 10000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = monte_carlo(s_price,n_days,mn,sma)[n_days-1]


# In[47]:


q = np.percentile(simulations,1)
plt.hist(simulations,bins=200)
plt.figtext(0.6,0.8,s="Start price: $%.2f" %s_price)
plt.figtext(0.6,0.7,"Mean final price: $%.2f" % simulations.mean())
plt.figtext(0.6,0.6,"VaR(0.99): $%.2f" % (s_price -q,))
plt.figtext(0.15,0.6, "q(0.99): $%.2f" % q)
plt.axvline(x=q, linewidth=4, color='r')
plt.title(u"Final price distribution for Google Stock after %s days" %n_days, weight='bold')


# <p>
#     <b>
#         We can infer from this that, Google's stock is pretty stable. The starting price that we had was USD622.05, and the <br>average final price over 10,000 runs was USD623.36.
# 
# The red line indicates the value of stock at risk at the desired confidence interval. For every stock, we'd be risking USD18.38, 99% of the time.
#     </b>
# </p>
