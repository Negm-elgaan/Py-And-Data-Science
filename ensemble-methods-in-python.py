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
