import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, accuracy_score, roc_auc_score
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
print("Test set accuracy: {:.3f}".format(acc))
#%%
#Plot the tree
plt.figure(figsize=(25,10))
plot_tree = plot_tree(clf_tree,
                      feature_names=features.columns, 
                      class_names=True, 
                      filled=True, 
                      rounded=True, 
                      fontsize=14)

#%%
#Make decision tree regressor
from sklearn.tree import DecisionTreeRegressor
 
# Instantiate dt
dt = DecisionTreeRegressor(max_depth=8,
             min_samples_leaf=0.13,
            random_state=3)
 
# Fit dt to the training set
dt.fit(X_train, y_train)
 
# Compute y_pred
y_pred = dt.predict(X_test)

# Print ROC_AUC score

auc = roc_auc_score(y_test, y_pred)
print("Test set AUC of dt: {:.3f}".format(auc))

#%%
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier

# Set seed for reproducibility
SEED=3
 
# Instantiate knn
knn = KNN(n_neighbors=5)
 
# Instantiate tree
tree = DecisionTreeClassifier(min_samples_leaf = 0.13, random_state=SEED)
 
# Define the list classifiers
classifiers = [('K Nearest Neighbours', knn), ('Classifier Tree', tree)]


#
# Evaluate individual classifiers
#
from sklearn.metrics import accuracy_score
 
# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:    
  
    # Fit clf to the training set
    clf.fit(X_train, y_train)    

    # Predict y_pred
    y_pred = clf.predict(X_test)
     
    # Calculate mse
    accuracy = accuracy_score(y_test, y_pred) 
    
    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))
#%%
#Now let's check if we can get a higher accuracy rating with ensembling

# Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier
 
# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators=classifiers)     
 
# Fit vc to the training set
vc.fit(X_train, y_train)   
 
# Evaluate the test set predictions
y_pred = vc.predict(X_test)
 
# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Voting Classifier: {:.3f}'.format(accuracy))
#Nope, doesn't seem like this will give us a better result. 
#%%
#Bagging
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
 
# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier
 
# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)
 
# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)

# Fit bc to the training set
bc.fit(X_train, y_train)
 
# Predict test set labels
y_pred = bc.predict(X_test)
 
# Evaluate acc_test
acc_test = accuracy_score(y_test, y_pred)
print('Test set accuracy of bc: {:.3f}'.format(acc_test)) 
#Get 1 more percent with bagging. 
#%%
# Instantiate rf
rf = RandomForestClassifier(max_depth=9, random_state=0)
             
# Fit rf to the training set    
rf.fit(X_train, y_train) 
 
# Predict test set labels
y_pred = rf.predict(X_test)
 
# Evaluate acc_test
acc_test = accuracy_score(y_test, y_pred)
print('Test set accuracy of rf: {:.3f}'.format(acc_test)) 
#RF gets 93%

#%%

# Predict the test set labels
y_pred = rf.predict(X_test)
 
# Print ROC_AUC score

bl_auc = roc_auc_score(y_test, y_pred)
print("Test set AUC of rf: {:.3f}".format(bl_auc))
#up from 0.741 to 0.801
#%%
#Let's rank feature importance now.
# Create a pd.Series of features importances
importances = pd.Series(data = rf.feature_importances_,
                        index = features.columns)
 
# Sort importances
importances_sorted = importances.sort_values()
 
# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()

#DayMins and MonthlyCharge are the most important feature, plus the daily_call_length feature I added. 
#%%
#Now let's try using AdaBoost to see if we can improve RMSE
# Import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
 
ada_reg = AdaBoostRegressor(n_estimators=100)

# Fit ada to the training set
ada_reg.fit(X_train, y_train)
 
# Compute the probabilities of obtaining the positive class
y_pred_ada = ada_reg.predict(X_test)

#Print ROC_AUC score

ada_auc = roc_auc_score(y_test, y_pred_ada)
print("Test set AUC of ada: {:.3f}".format(ada_auc))

#Ada brings it from 0.801 to 0.884, nice.
#%%
#Now let's try XGBoost
import xgboost as xgb

xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=1)

# Fit ada to the training set
xgb_model.fit(X_train, y_train)

# Compute the probabilities of obtaining the positive class
y_pred_xg = xgb_model.predict(X_test)

#Print ROC_AUC score

xgb_auc = roc_auc_score(y_test, y_pred)
print("Test set AUC of xgb: {:.3f}".format(xgb_auc))
#XGB worse than ada (.801 to 0.885)
#%%
#Hyperparameter tuning
# Instantiate rf
rf_2 = RandomForestClassifier(max_depth=2, random_state=0)
             
# Fit rf to the training set    
rf_2.fit(X_train, y_train) 
 
# Predict test set labels
y_pred = rf_2.predict(X_test)
 
# Evaluate acc_test
acc_test = accuracy_score(y_test, y_pred)
print('Test set accuracy of rf with max_depth_2: {:.4f}'.format(acc_test)) 

# Instantiate rf
rf_12 = RandomForestClassifier(max_depth=12, random_state=0)
             
# Fit rf to the training set    
rf_12.fit(X_train, y_train) 
 
# Predict test set labels
y_pred = rf_12.predict(X_test)
 
# Evaluate acc_test
acc_test = accuracy_score(y_test, y_pred)
print('Test set accuracy of rf with max_depth_12: {:.4f}'.format(acc_test)) 
#Optimal depth is at 12, gives us 93.5% accuracy
#But, 6 gives 92.5% accuracy. So 6 is more efficient
#%%


