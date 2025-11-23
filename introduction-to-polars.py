import polars as PL
# Load the CSV into a DataFrame
ev_df = PL.read_csv("electric_vehicles.csv")

# Print the first three rows
print(ev_df.head(3))