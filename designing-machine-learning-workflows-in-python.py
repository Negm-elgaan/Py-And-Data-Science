# Inspect the first few lines of your data using head()
credit.head()
##################
# Inspect the first few lines of your data using head()
credit.head(3)

# Create a label encoder for each column. Encode the values
for column in non_numeric_columns:
    le = LabelEncoder()
    credit[column] = le.fit_transform(credit[column])
#############################
# Inspect the first few lines of your data using head()
credit.head(3)

# Create a label encoder for each column. Encode the values
for column in non_numeric_columns:
    le = LabelEncoder()
    credit[column] = le.fit_transform(credit[column])

# Inspect the data types of the columns of the data frame
print(credit.dtypes)
#################
# Split the data into train and test, with 20% as test
X_train, X_test , y_train , y_test = train_test_split(
  X, y, test_size=0.2, random_state=1)
#########################################
# Split the data into train and test, with 20% as test
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=1)

# Create a random forest classifier, fixing the seed to 2
rf_model = RandomForestClassifier(random_state=2).fit(
  X_train , y_train)

# Use it to predict the labels of the test data
rf_predictions = rf_model.predict(X_test)
###############################
# Split the data into train and test, with 20% as test
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=1)

# Create a random forest classifier, fixing the seed to 2
rf_model = RandomForestClassifier(random_state=2).fit(
  X_train, y_train)

# Use it to predict the labels of the test data
rf_predictions = rf_model.predict(X_test)

# Assess the accuracy of both classifiers
accuracies['rf'] = accuracy_score(y_test, rf_predictions)