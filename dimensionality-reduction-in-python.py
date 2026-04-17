print(pokemon_df.describe())
# Remove the feature without variance from this list
number_cols = ['HP', 'Attack', 'Defense']
########################
# Remove the feature without variance from this list
number_cols = ['HP', 'Attack', 'Defense']

# Leave this list as is for now
non_number_cols = ['Name', 'Type', 'Legendary']

# Sub-select by combining the lists with chosen features
df_selected = pokemon_df[non_number_cols + number_cols]

# Prints the first 5 lines of the new DataFrame
print(df_selected.head())
#############################################
# Leave this list as is
number_cols = ['HP', 'Attack', 'Defense']

print(pokemon_df.describe(exclude = 'number'))
# Remove the feature without variance from this list
non_number_cols = ['Name', 'Type']

# Create a new DataFrame by subselecting the chosen features
df_selected = pokemon_df[number_cols + non_number_cols]

# Prints the first 5 lines of the new DataFrame
print(df_selected.head())