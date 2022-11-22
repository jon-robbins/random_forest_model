import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



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
#%%
#now let's visualize all of our columns and check the distributions
sns.set_style('whitegrid')

cols = list(df.columns)
for i in range(len(cols)):
    plt.figure(i)
    sns.kdeplot(np.array(df.iloc[i]), bw=0.5).set(title=cols[i])

#%%

# define target

# Labels are the values we want to predict
labels = np.array(df['Churn'])
# Remove the labels from the featuresaxis 1 refers to the columns
features = df.drop('Churn', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
# Split the data into training and testing sets

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
#%%
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)
#%%
# Import DecisionTreeClassifier

# Instantiate rf
rf = RandomForestClassifier(max_depth=9, random_state=0)
             
# Fit rf to the training set    
rf.fit(train_features, train_labels) 
 
# Predict test set labels
y_pred = rf.predict(test_features)
 
# Evaluate acc_test
acc_test = accuracy_score(test_features, test_labels)
print('Test set accuracy of rf: {:.2f}'.format(acc_test)) 




