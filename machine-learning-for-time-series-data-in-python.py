# Print the first 5 rows of data
print(data.head())
#########################################
# Print the first 5 rows of data2
print(data2.head())
#######################
# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(y='data_values', ax=axs[0])
data2.iloc[:1000].plot(y='data_values', ax=axs[1])
plt.show()