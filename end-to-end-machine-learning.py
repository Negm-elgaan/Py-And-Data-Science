# Print the first 5 rows of the DataFrame
print(heart_disease_df.head(5))

# Print information about the DataFrame
print(heart_disease_df.info())
#################################
# Print the first 5 rows of the DataFrame
print(heart_disease_df.head())

# Print information about the DataFrame
print(heart_disease_df.info())

# Visualize the cholesterol column
heart_disease_df['chol'].plot(kind='hist')

# Set the title and axis labels
plt.title('Cholesterol distribution')
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.show()