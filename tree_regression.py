import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree



df = pd.read_csv('churn_data.csv', sep=';')

#new feature: overage fee to avg monthly bill ratio. we can assume lower bill = lower ability to pay or interest in the product. If the overage fee is a higher ratio compared to their bill, they could be disproportionately affected. 

df['overage_ratio'] = df['OverageFee'] / df['MonthlyCharge']

#df_outofdata = df.loc[(df['DataPlan'] == 0) & df['DataUsage'] > 0]

# we see there are about 600 people who are using data without a data plan, which I assume has higher fees.

df['out_of_plan_data_use'] = (df['DataPlan'] == 0) & (df['DataUsage'] > 0)
df['out_of_plan_data_use'] = df['out_of_plan_data_use'].astype(int)

#Roaming minutes are charged at higher rates than normal daytime minutes. Let's create another variable showing the ratio of roaming to daytime minutes

df['roaming_daytime_ratio'] = df['RoamMins'] / df['DayMins']
df['roaming_daytime_ratio'] = df['roaming_daytime_ratio'].replace(np.inf, 1, inplace=False)
df.describe()

# If the have a high monthly charge, and they have made a lot of calls, then I assume that they would have called to lower the monthly charge. Let's create an interaction term between these two
df['cs_calls_bill'] = df['CustServCalls'] * df['MonthlyCharge']

#daily call length. If they have longer calls per day, they may be business customers who have different needs than regular customers.
df['daily_call_length'] = (df['DayMins'] / 30.417) * df['DayCalls']
#%%
#now let's visualize all of our columns and check the distributions
sns.set_style('whitegrid')

cols = list(df.columns)
for i in range(len(cols)):
    plt.figure(i)
    sns.kdeplot(np.array(df.iloc[i]), bw=0.5).set(title=cols[i])

#%%

#Split features and target into X and y
features = df.drop(labels='Churn', axis=1)
X = np.array(features)
y = np.array(df['Churn'])
y_labels = np.array(['Not churn', 'Churn'])
#%%
#Create tet and train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Train the model using DecisionTree classifier
clf_tree = DecisionTreeClassifier(max_depth=4, random_state=1)
clf_tree.fit(X_train, y_train)

# Predict test set labels
y_pred = clf_tree.predict(X_test)
#%%
# Predict test set labels
y_pred = clf_tree.predict(X_test)
 
# Compute test set accuracy  
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))
#%%
#Plot the tree
plt.figure(figsize=(25,10))
plot_tree = plot_tree(clf_tree,
                      feature_names=features.columns, 
                      class_names=True, 
                      filled=True, 
                      rounded=True, 
                      fontsize=14)



