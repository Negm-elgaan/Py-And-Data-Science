# Instantiate a Naive Bayes classifier
clf_nb = GaussianNB()

# Fit the model to the training set
clf_nb.fit(X_train , y_train)

# Calculate the predictions on the test set
pred = clf_nb.predict(X_test)

# Evaluate the performance using the accuracy score
print("Accuracy: {:0.4f}".format(accuracy_score(y_test, pred)))
#############################
# Instantiate a 5-nearest neighbors classifier with 'ball_tree' algorithm
clf_knn = KNeighborsClassifier(algorithm = 'ball_tree' , n_neighbors = 5)

# Fit the model to the training set
clf_knn.fit(X_train,y_train)

# Calculate the predictions on the test set
pred = clf_knn.predict(X_test)

# Evaluate the performance using the accuracy score
print("Accuracy: {:0.4f}".format(accuracy_score(y_test, pred)))
##################################
# Build and fit a Decision Tree classifier
clf_dt = DecisionTreeClassifier(min_samples_leaf = 3 , min_samples_split = 9 , random_state = 500)
clf_dt.fit(X_train , y_train)

# Build and fit a 5-nearest neighbors classifier using the 'Ball-Tree' algorithm
clf_knn = KNeighborsClassifier(n_neighbors = 5 , algorithm = 'ball_tree')
clf_knn.fit(X_train , y_train)

# Evaluate the performance using the accuracy score
print('Decision Tree: {:0.4f}'.format(accuracy_score(y_test, clf_dt.predict(X_test))))
print('5-Nearest Neighbors: {:0.4f}'.format(accuracy_score(y_test, clf_knn.predict(X_test))))
#######################################
from sklearn.ensemble import StackingClassifier as SC
# Prepare the list of tuples with the first-layer classifiers
classifiers = [
	clf_dt,
    clf_knn
]

# Instantiate the second-layer meta estimator
clf_meta = LogisticRegression()

# Build the stacking classifier
clf_stack = SC(
   estimators = classifiers,
   final_estimator = clf_meta,
   stack_method = 'predict_proba',
   passthrough = False
   )
########################################
# Fit the stacking classifier to the training set
clf_stack.fit(X_train , y_train)

# Obtain the final predictions from the stacking classifier
pred_stack = clf_stack.predict(X_test)

# Evaluate the new performance on the test set
print('Accuracy: {:0.4f}'.format(accuracy_score(y_test , pred_stack)))
############################
# Instantiate the first-layer classifiers
clf_dt = DecisionTreeClassifier(min_samples_leaf = 3 , min_samples_split = 9 , random_state = 500)
clf_knn = KNeighborsClassifier(algorithm = 'ball_tree' , n_neighbors = 5)

# Instantiate the second-layer meta classifier
clf_meta = LogisticRegression()

# Build the Stacking classifier
clf_stack = StackingClassifier(classifiers = [clf_dt , clf_knn] , meta_classifier = clf_meta ,use_probas = True , use_features_in_secondary = False)
clf_stack.fit(X_train , y_train)

# Evaluate the performance of the Stacking classifier
pred_stack = clf_stack.predict(X_test)
print("Accuracy: {:0.4f}".format(accuracy_score(y_test, pred_stack)))
##########################################
from mlxtend.regressor import StackingRegressor as SR
# Instantiate the 1st-layer regressors
reg_dt = DecisionTreeRegressor(min_samples_leaf = 11 , min_samples_split = 33 , random_state = 500)
reg_lr = LinearRegression()
reg_ridge = Ridge(random_state = 500)

# Instantiate the 2nd-layer regressor
reg_meta = LinearRegression()

# Build the Stacking regressor
reg_stack = SR(regressors = [reg_dt , reg_lr , reg_ridge] , meta_regressor = reg_meta)
reg_stack.fit(X_train , y_train)

# Evaluate the performance on the test set using the MAE metric
pred = reg_stack.predict(X_test)
print('MAE: {:.3f}'.format(mean_absolute_error(y_test, pred)))
###############################
# Create the first-layer models
clf_knn = KNeighborsClassifier(n_neighbors = 5 , algorithm = 'ball_tree')
clf_dt = DecisionTreeClassifier(min_samples_leaf = 5 , min_samples_split = 15 , random_state=500)
clf_nb = GaussianNB()

# Create the second-layer model (meta-model)
clf_lr = LogisticRegression()

# Create and fit the stacked model
clf_stack = StackingClassifier(classifiers = [clf_knn , clf_dt , clf_nb] , meta_classifier = clf_lr)
clf_stack.fit(X_train, y_train)

# Evaluate the stacked model’s performance
print("Accuracy: {:0.4f}".format(accuracy_score(y_test, clf_stack.predict(X_test))))
################################
ratings.describe()
##########################
# Split into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size =  0.2, random_state=42)

# Instantiate the regressor
reg_dt = DecisionTreeRegressor(min_samples_leaf = 3, min_samples_split = 9, random_state=500)

# Fit to the training set
reg_dt.fit(X_train , y_train)

# Evaluate the performance of the model on the test set
y_pred = reg_dt.predict(X_test)
print('MAE: {:.3f}'.format(mean_absolute_error(y_test, y_pred)))
####################################
# Predict the labels of the test set
pred_lr = clf_lr.predict(X_test)
pred_dt = clf_dt.predict(X_test)
pred_knn = clf_knn.predict(X_test)
###############################
# Make the invidual predictions
pred_lr = clf_lr.predict(X_test)
pred_dt = clf_dt.predict(X_test)
pred_knn = clf_knn.predict(X_test)

# Evaluate the performance of each model
score_lr = f1_score(y_test , pred_lr)
score_dt = f1_score(y_test , pred_dt)
score_knn = f1_score(y_test , pred_knn)

# Print the scores
print(score_lr)
print(score_dt)
print(score_knn)
#################################
from sklearn.ensemble import VotingClassifier as VC
# Instantiate the individual models
clf_knn = KNeighborsClassifier(n_neighbors = 5)
clf_lr = LogisticRegression(class_weight = 'balanced')
clf_dt = DecisionTreeClassifier(min_samples_leaf = 3 , min_samples_split = 9 , random_state = 500)

# Create and fit the voting classifier
clf_vote = VC(estimators =  [('knn' , clf_knn) , ('lr' , clf_lr) , ('dt' , clf_dt)])
clf_vote.fit(X_train, y_train)
###############################
# Calculate the predictions using the voting classifier
pred_vote = clf_vote.predict(X_test)

# Calculate the F1-Score of the voting classifier
score_vote = f1_score(y_test , pred_vote)
print('F1-Score: {:.3f}'.format(score_vote))

# Calculate the classification report
report = classification_report(y_test , pred_vote)
print(report)
##############
got.describe()
got.shape
####################
from sklearn.ensemble import VotingRegressor as VR
# Build the individual models
clf_lr = LogisticRegression(class_weight='balanced')
clf_dt = DecisionTreeClassifier(min_samples_leaf=3, min_samples_split=9, random_state=500)
clf_svm = SVC(probability=True, class_weight='balanced', random_state=500)

# List of (string, estimator) tuples
estimators = [('lr' , clf_lr) , ("dt" , clf_dt) , ('svm' , clf_svm)]

# Build and fit an averaging classifier
clf_avg = VotingClassifier(estimators = estimators , voting = 'soft')
clf_avg.fit(X_train, y_train)

# Evaluate model performance
acc_avg = accuracy_score(y_test,  clf_avg.predict(X_test))
print('Accuracy: {:.2f}'.format(acc_avg))
#################################
# List of (string, estimator) tuples
estimators = [('dt' , clf_dt) , ('lr' , clf_lr) , ('knn' , clf_knn)]

# Build and fit a voting classifier
clf_vote = VotingClassifier(estimators = estimators)
clf_vote.fit(X_train, y_train)

# Build and fit an averaging classifier
clf_avg = VotingClassifier(estimators = estimators , voting = 'soft')
clf_avg.fit(X_train, y_train)

# Evaluate the performance of both models
acc_vote = accuracy_score(y_test, clf_vote.predict(X_test))
acc_avg = accuracy_score(y_test,  clf_avg.predict(X_test))
print('Voting: {:.2f}, Averaging: {:.2f}'.format(acc_vote, acc_avg))
#####################################
# Build unrestricted decision tree
clf = DecisionTreeClassifier(min_samples_leaf = 3 , min_samples_split = 9 , random_state = 500)
clf.fit(X_train, y_train)

# Predict the labels
pred = clf.predict(X_test)

# Print the confusion matrix
cm = confusion_matrix(y_test, pred)
print('Confusion matrix:\n', cm)

# Print the F1 score
score = f1_score(y_test, pred)
print('F1-Score: {:.3f}'.format(score))
###########################
# Build restricted decision tree
clf = DecisionTreeClassifier(max_depth = 4  , max_features = 2 , random_state=500)
clf.fit(X_train, y_train)

# Predict the labels
pred = clf.predict(X_test)

# Print the confusion matrix
cm = confusion_matrix(y_test, pred)
print('Confusion matrix:\n', cm)

# Print the F1 score
score = f1_score(y_test, pred)
print('F1-Score: {:.3f}'.format(score))
#################################
# Take a sample with replacement
X_train_sample = X_train.sample(replace = True , frac = 1.0 , random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]

# Build a "weak" Decision Tree classifier
clf = DecisionTreeClassifier(max_depth = 4 , random_state=500)

# Fit the model to the training sample
clf.fit(X_train_sample , y_train_sample)
#############################################
# Build the list of individual models
clf_list = []
for i in range(21):
	weak_dt = build_decision_tree(X_train , y_train , random_state = i)
	clf_list.append(weak_dt)

# Predict on the test set
pred = predict_voting(clf_list , X_test)

# Print the F1 score
print('F1 score: {:.3f}'.format(f1_score(y_test, pred)))
####################################
# Instantiate the base model
clf_dt = DecisionTreeClassifier(max_depth = 4)

# Build the Bagging classifier
clf_bag = BaggingClassifier(clf_dt , n_estimators = 21 , random_state=500)

# Fit the Bagging model to the training set
clf_bag.fit(X_train, y_train)

# Predict the labels of the test set
pred = clf_bag.predict(X_test)

# Show the F1-score
print('F1-Score: {:.3f}'.format(f1_score(y_test, pred)))
##############################################
# Build and train the bagging classifier
clf_bag = BaggingClassifier(
  clf_dt,
  n_estimators = 21,
  oob_score = True,
  random_state=500)
clf_bag.fit(X_train, y_train)

# Print the out-of-bag score
print('OOB-Score: {:.3f}'.format(clf_bag.oob_score_))

# Evaluate the performance on the test set to compare
pred = clf_bag.predict(X_test)
print('Accuracy: {:.3f}'.format(accuracy_score(y_test, pred)))
##############
uci_secom.describe()
uci_secom['Pass/Fail'].value_counts()
#######################
# Build a balanced logistic regression
clf_lr = LogisticRegression(class_weight = 'balanced' , solver = 'liblinear' , random_state = 42)

# Build and fit a bagging classifier
clf_bag = BaggingClassifier(base_estimator = clf_lr , max_features = 10 , oob_score = True , random_state=500)
clf_bag.fit(X_train, y_train)

# Evaluate the accuracy on the test set and show the out-of-bag score
pred = clf_bag.predict(X_test)
print('Accuracy:  {:.2f}'.format(accuracy_score(y_test, pred)))
print('OOB-Score: {:.2f}'.format(clf_bag.oob_score_))

# Print the confusion matrix
print(confusion_matrix(y_test, pred))
################################
# Build a balanced logistic regression
clf_base = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)

# Build and fit a bagging classifier with custom parameters
clf_bag = BaggingClassifier(base_estimator = clf_base , max_features = 10, n_estimators = 20 , bootstrap = False , max_samples = 0.65 , random_state=500)
clf_bag.fit(X_train, y_train)

# Calculate predictions and evaluate the accuracy on the test set
y_pred = clf_bag.predict(X_test)
print('Accuracy:  {:.2f}'.format(accuracy_score(y_test, y_pred)))

# Print the classification report
print(classification_report(y_test, y_pred))
##############
movies.describe()
################
features.describe()
features['runtime'].value_counts()
features['runtime'].isna().sum()
############################
# Build and fit linear regression model
reg_lm = LinearRegression()
reg_lm.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = reg_lm.predict(X_test)

# Evaluate the performance using the RMSE
rmse = np.sqrt(mean_squared_error(y_test , pred))
print('RMSE: {:.3f}'.format(rmse))
######################
# Fit a linear regression model to the previous errors
reg_error = LinearRegression()
reg_error.fit(X_train_pop , y_train_error)

# Calculate the predicted errors on the test set
pred_error = reg_error.predict(X_test_pop)

# Evaluate the updated performance
rmse_error = np.sqrt(mean_squared_error(y_test_error , pred_error))
print('RMSE: {:.3f}'.format(rmse_error))
##########################################
# Instantiate the default linear regression model
reg_lm = LinearRegression()

# Build and fit an AdaBoost regressor
reg_ada = AdaBoostRegressor(base_estimator = reg_lm , n_estimators = 12 , random_state=500)
reg_ada.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = reg_ada.predict(X_test)

# Evaluate the performance using the RMSE
rmse = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE: {:.3f}'.format(rmse))
#######################
# Build and fit a tree-based AdaBoost regressor
reg_ada = AdaBoostRegressor(n_estimators = 12 , random_state=500)
reg_ada.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = reg_ada.predict(X_test)

# Evaluate the performance using the RMSE
rmse = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE: {:.3f}'.format(rmse))
########################################
# Build and fit an AdaBoost regressor
reg_ada = AdaBoostRegressor(n_estimators = 100 , learning_rate = 0.01 , random_state=500)
reg_ada.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = reg_ada.predict(X_test)

# Evaluate the performance using the RMSE
rmse = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE: {:.3f}'.format(rmse))
################################################
reviews['Sentiment'].isna().sum()
##################
# Build and fit a Gradient Boosting classifier
clf_gbm = GradientBoostingClassifier(n_estimators = 100 , learning_rate = 0.1 , random_state=500)
clf_gbm.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = clf_gbm.predict(X_test)

# Evaluate the performance based on the accuracy
acc = accuracy_score(y_test , pred)
print('Accuracy: {:.3f}'.format(acc))

# Get and show the Confusion Matrix
cm = confusion_matrix(y_test , pred)
print(cm)
####################################
import catboost as CB

# Build and fit a CatBoost regressor
reg_cat = CB.CatBoostRegressor(n_estimators = 100 , learning_rate = 0.1 , max_depth = 3 , random_state = 500)
reg_cat.fit(X_train , y_train)

# Calculate the predictions on the test set
pred = reg_cat.predict(X_test)

# Evaluate the performance using the RMSE
rmse_cat = np.sqrt(mean_squared_error(y_test, pred))
print('RMSE (CatBoost): {:.3f}'.format(rmse_cat))
#######################################
import xgboost as XGB
import lightgbm as LGBM

# Build and fit an XGBoost regressor
reg_xgb = XGB.XGBRegressor(n_estimators = 100 , max_depth = 3 , objective ="reg:squarederror" , learning_rate = 0.1 , n_jobs = 2 , random_state = 500)
reg_xgb.fit(X_train , y_train)

# Build and fit a LightGBM regressor
reg_lgb = LGBM.LGBMRegressor(max_depth = 3 , seed = 500 , objective = 'mean_squared_error' , learning_rate = 0.1 , n_estimators = 100)
reg_lgb.fit(X_train , y_train)

# Calculate the predictions and evaluate both regressors
pred_xgb = reg_xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
pred_lgb = reg_lgb.predict(X_test)
rmse_lgb = np.sqrt(mean_squared_error(y_test, pred_lgb))

print('Extreme: {:.3f}, Light: {:.3f}'.format(rmse_xgb, rmse_lgb))