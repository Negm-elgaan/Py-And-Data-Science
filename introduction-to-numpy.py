# Import NumPy
import numpy as np

# Convert sudoku_list into an array
sudoku_array = np.array(sudoku_list)

# Print the type of sudoku_array 
print(type(sudoku_array))
#########################
# Create an array of zeros which has four columns and two rows
zero_array = np.zeros((2,4))
print(zero_array)
#############################
# Create an array of random floats which has six columns and three rows
random_array = np.random.random((3,6))
print(random_array)