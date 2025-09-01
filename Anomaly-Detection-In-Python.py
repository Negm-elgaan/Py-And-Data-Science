# Extract price
prices = airbnb_df['price']

# Print 5-number summary
print(prices.describe())
############################
# Find the square root of the length of prices
n_bins = np.sqrt(len(prices))

# Cast to an integer
n_bins = int(n_bins)

plt.figure(figsize=(8, 4))

# Create a histogram
plt.hist(prices , bins = n_bins, color = 'red')
plt.show()
#################################
# Create a list of consecutive integers
integers = range(len(prices))

plt.figure(figsize=(16, 8))

# Plot a scatterplot
plt.scatter(integers, prices, c='red', alpha=0.5)
plt.show()
########################################
# Create a boxplot with custom whisker lengths
plt.boxplot(prices, whis = 5)
plt.show()
############################
# Calculate the 25th and 75th percentiles
q1 = prices.quantile(0.25)
q3 = prices.quantile(0.75)

# Find the IQR
IQR = q3 - q1
factor = 2.5

# Calculate the lower limit
lower_limit = q1 - factor * IQR

# Calculate the upper limit
upper_limit = q3 + factor * IQR
##################################
# Import the zscores function
from scipy.stats import zscore

# Find the zscores of prices
scores = zscore(prices)

# Check if the absolute values of scores are over 3
is_over_3 = abs(scores) > 3

# Use the mask to subset prices
outliers = prices[is_over_3]

print(len(outliers))
########################################
# Initialize with a threshold of 3.5
mad = MAD(threshold = 3.5)

# Reshape prices to make it 2D
prices_reshaped = prices.values.reshape(-1, 1)

# Fit and predict outlier labels on prices_reshaped
labels = mad.fit_predict(prices_reshaped)

# Filter for outliers
outliers = prices_reshaped[labels == 1]

print(len(outliers))
#########################################
# Import IForest from pyod
from pyod.models.iforest import IForest

# Initialize an instance with default parameters
iforest = IForest()

# Generate outlier labels
labels = iforest.fit_predict(big_mart)

# Filter big_mart for outliers
outliers = big_mart[labels == 1]

print(outliers.shape)