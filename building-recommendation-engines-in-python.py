# Inspect the listening_history_df DataFrame
print(listening_history_df.head())

# Calculate the number of unique values
print(listening_history_df[['Rating', 'Skipped Track']].nunique())

# Display a histogram of the values in the Rating column
listening_history_df['Rating'].hist()
plt.show()