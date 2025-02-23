# Print the first five rows of unemployment
print(unemployment.head(5))
##############
# Print a summary of non-missing values and data types in the unemployment DataFrame
print(unemployment.info())
###########
# Print summary statistics for numerical columns in unemployment
print(unemployment.describe())
#############
# Count the values associated with each continent in unemployment
print(unemployment.value_counts('continent'))
##################
# Import the required visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Create a histogram of 2021 unemployment; show a full percent in each bin
sns.histplot(data = unemployment , x="2021", binwidth=1)
plt.show()
############
unemployment.dtypes
###########
# Update the data type of the 2019 column to a float
unemployment["2019"] = unemployment['2019'].astype('float')
# Print the dtypes to check your work
print(unemployment.dtypes)
###############
# Define a Series describing whether each continent is outside of Oceania
not_oceania = ~unemployment['continent'].isin(['Oceania'])
not_oceania
###################
# Define a Series describing whether each continent is outside of Oceania
not_oceania = ~unemployment["continent"].isin(["Oceania"])

# Print unemployment without records related to countries in Oceania
print(unemployment[not_oceania])
##############################
# Print the minimum and maximum unemployment rates during 2021
print( unemployment['2021'].min() , unemployment['2021'].max())

# Create a boxplot of 2021 unemployment rates, broken down by continent
sns.boxplot(x = '2021' , y = 'continent' , data = unemployment)
plt.show()
####################
# Print the mean and standard deviation of rates by year
print(unemployment.agg(['mean','std']))
##################
# Print yearly mean and standard deviation grouped by continent
print(unemployment.groupby('continent').agg(['mean','std']))
##################
continent_summary = unemployment.groupby("continent").agg(
    # Create the mean_rate_2021 column
    mean_rate_2021=('2021','mean'),
    # Create the std_rate_2021 column
    std_rate_2021=('2021','std')
)
print(continent_summary)
###############
# Create a bar plot of continents and their average unemployment
sns.barplot(x = 'continent' , y = '2021' , data = unemployment)
plt.show()
##################
# Count the number of missing values in each column
print(planes.isna().sum())
####################
# Count the number of missing values in each column
print(planes.isna().sum())

# Find the five percent threshold
threshold = len(planes) * 0.05
#################
# Count the number of missing values in each column
print(planes.isna().sum())

# Find the five percent threshold
threshold = len(planes) * 0.05

# Create a filter
cols_to_drop = planes.columns[planes.isna().sum() <= threshold]

# Drop missing values for columns below the threshold
planes.dropna(subset = cols_to_drop, inplace = True)

print(planes.isna().sum())
###############
# Check the values of the Additional_Info column
print(planes['Additional_Info'].value_counts())
#################
# Check the values of the Additional_Info column
print(planes["Additional_Info"].value_counts())

# Create a box plot of Price by Airline
sns.boxplot(data=planes, x='Airline', y='Price')

plt.show()
####################
# Calculate median plane ticket prices by Airline
airline_prices = planes.groupby("Airline")["Price"].median()

print(airline_prices)
######################
# Calculate median plane ticket prices by Airline
airline_prices = planes.groupby("Airline")["Price"].median()

print(airline_prices)

# Convert to a dictionary
prices_dict = dict(airline_prices)
####################
# Calculate median plane ticket prices by Airline
airline_prices = planes.groupby("Airline")["Price"].median()

print(airline_prices)

# Convert to a dictionary
prices_dict = airline_prices.to_dict()

# Map the dictionary to missing values of Price by Airline
planes["Price"] = planes["Price"].fillna(planes["Airline"].map(prices_dict))

# Check for missing values
print(planes.isna().sum())
################
# Filter the DataFrame for object columns
non_numeric = planes.select_dtypes("object")

# Loop through columns
for col in non_numeric.columns:
  
  # Print the number of unique values
  print(f"Number of unique values in {col} column: ", non_numeric[col].nunique())
#################
# Create a list of categories
flight_categories = ["Short-haul", "Medium" , "Long-haul"]
######################
# Create a list of categories
flight_categories = ["Short-haul", "Medium", "Long-haul"]

# Create short-haul values
short_flights = "^0h|^1h|^2h|^3h|^4h"

# Create medium-haul values
medium_flights = "^5h|^6h|^7h|^8h|^9h"

# Create long-haul values
long_flights = "10h|11h|12h|13h|14h|15h|16h"
#########################
# Create conditions for values in flight_categories to be created
conditions = [
    (planes["Duration"].str.contains(short_flights)),
    (planes["Duration"].str.contains(medium_flights)),
    (planes["Duration"].str.contains(long_flights))
]

# Apply the conditions list to the flight_categories
planes["Duration_Category"] = np.select(conditions, 
                                        flight_categories,
                                        default="Extreme duration")

# Plot the counts of each category
sns.countplot(data=planes, x="Duration_Category")
plt.show()