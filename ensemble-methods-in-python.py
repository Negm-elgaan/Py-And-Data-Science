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