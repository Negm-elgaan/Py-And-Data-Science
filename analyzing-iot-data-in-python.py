# Imports
import requests as r
import pandas as pd

# Download data from URL
res = r.get(URL)

# Convert the result
data_temp = res.json() 
print(data_temp)

# Convert json data to DataFrame
df_temp = pd.DataFrame(res.json())

print(df_temp.head())