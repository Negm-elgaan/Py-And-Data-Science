# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Scatterplot 1 - father heights vs. son heights with darkred square markers
plt.scatter(father_son.fheight, father_son.sheight, c = 'darkred', marker = 's')

# Show your plot
plt.show()
#########################################
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Scatterplot 2 - yellow markers with darkblue borders
plt.scatter( x=father_son.fheight, y = father_son.sheight , c ='yellow', edgecolor = 'darkblue')

# Show the plot
plt.show()
##############################
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Scatterplot 3
plt.scatter(father_son.fheight, father_son.sheight,  c = 'yellow', edgecolor = 'darkblue')
plt.grid()
plt.xlabel('father height (inches)')
plt.ylabel('son height (inches)')
plt.title('Son Height as a Function of Father Height')

# Show your plot
plt.show()