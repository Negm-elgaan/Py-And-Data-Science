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