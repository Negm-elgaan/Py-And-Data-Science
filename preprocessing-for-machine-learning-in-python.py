volunteer.isna().sum()
#########################
# Drop the Latitude and Longitude columns from volunteer
volunteer_cols = volunteer.drop(['Latitude' , 'Longitude'] , axis = 1)

# Drop rows with missing category_desc values from volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset = ['category_desc'])

# Print out the shape of the subset
print(volunteer_subset.shape)
#################################
# Print the head of the hits column
print(volunteer["hits"].head())

# Convert the hits column to type int
volunteer["hits"] = volunteer["hits"].astype(int)

# Look at the dtypes of the dataset
print(volunteer.dtypes)
#########################################
volunteer['category_desc'].value_counts()
#######################
# Create a DataFrame with all columns except category_desc
X = volunteer.drop('category_desc', axis = 1)

# Create a category_desc labels dataset
y = volunteer[['category_desc']]

# Use stratified sampling to split up the dataset according to the y dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state=42)

# Print the category_desc counts from y_train
print(y_train['category_desc'].value_counts())
#########################################
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X , y , stratify = y, random_state=42)

knn = KNeighborsClassifier()

# Fit the knn model to the training data
knn.fit(X_train , y_train)

# Score the model on the test data
print(knn.score(X_test,y_test))
########################
wine.var()
#################################
# Print out the variance of the Proline column
print(wine['Proline'].var())

# Apply the log normalization function to the Proline column
wine['Proline_log'] = np.log(wine['Proline'])

# Check the variance of the normalized Proline column
print(wine['Proline_log'].var())
####################################
print(wine.std() , wine.mean())
########################
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create the scaler
scaler = StandardScaler()

# Subset the DataFrame you want to scale 
wine_subset = wine[['Ash' , 'Alcalinity of ash' , 'Magnesium']]

# Apply the scaler to wine_subset
wine_subset_scaled = scaler.fit_transform(wine_subset)
#################################
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Instantiate a StandardScaler
scaler = StandardScaler()

# Scale the training and test features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train_scaled , y_train)

# Score the model on the test data
print(knn.score(X_test_scaled, y_test))
#######################################
# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
hiking['Accessible_enc'] = enc.fit_transform(hiking['Accessible'])

# Compare the two columns
print(hiking[['Accessible', 'Accessible_enc']].head())
#################################
# Transform the category_desc column
category_enc = pd.get_dummies(volunteer["category_desc"])

# Take a look at the encoded columns
print(category_enc.head())
######################################
# Use .loc to create a mean column
running_times_5k["mean"] = running_times_5k.loc[:,'run1':'run5'].mean(axis=1)

# Take a look at the results
print(running_times_5k.head())