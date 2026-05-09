# Create an array of integers from 0 to 9
quantity = np.arange(0,10)

# Define the cost function
def cost(q): 
  return 50 + 5 * ((q - 2) ** 2)

# Plot cost versus quantity
plt.plot(quantity ,cost(quantity) )
plt.xlabel('Quantity (thousands)')
plt.ylabel('Cost ($ K)')
plt.show()