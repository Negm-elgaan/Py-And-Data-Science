fig.show()
#######################
import plotly.express as px
# Define the x and y axis columns
x_column = "month"
y_column = "sales"

# Define the chart title
chart_title = "Sales for Jan-Mar 2025"

# Create a bar plot
fig = px.bar(data_frame = monthly_sales , x = x_column , y = y_column , title = chart_title)

fig.show()