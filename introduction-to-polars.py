import polars as PL
# Load the CSV into a DataFrame
ev_df = PL.read_csv("electric_vehicles.csv")

# Print the first three rows
print(ev_df.head(3))
###################
# Print number of rows and columns of ev_df
print(ev_df.shape)
##########################
# Print the column names of ev_df
print(ev_df.columns)
################################
# Print the column names and dtypes
print(ev_df.schema)
###########################
# Print the first values in a vertical format
print(ev_df.glimpse())