# Import PyTorch
import torch

temperatures = [[72, 75, 78], [70, 73, 76]]

# Create a tensor from temperatures
temp_tensor = torch.tensor(temperatures)

print(temp_tensor)
#######################
######################
adjustment = torch.tensor([[2, 2, 2], [2, 2, 2]])

# Display the shape of the adjustment tensor
print("Adjustment shape:", adjustment.shape)

# Display the type of the adjustment tensor
print("Adjustment type:", adjustment.dtype)

print("Temperatures shape:", temperatures.shape)
print("Temperatures type:", temperatures.dtype)
##########################
adjustment = torch.tensor([[2, 2, 2], [2, 2, 2]])

# Add the temperatures and adjustment tensors
corrected_temperatures = temperatures + adjustment
print("Corrected temperatures:", corrected_temperatures)
####################################################
import torch
import torch.nn as nn

input_tensor = torch.tensor([[0.3471, 0.4547, -0.2356]])

# Create a Linear layer
linear_layer = nn.Linear(
                         in_features = 3, 
                         out_features = 2
                         )

# Pass input_tensor through the linear layer
output = linear_layer(input_tensor)

print(output)
#########################
import torch
import torch.nn as nn

input_tensor = torch.Tensor([[2, 3, 6, 7, 9, 3, 2, 1]])

# Create a container for stacking linear layers
model = nn.Sequential(nn.Linear(8, 4),
                nn.Linear(4, 1)
                )

output = model(input_tensor)
print(output)
##############################
import torch.nn as nn

model = nn.Sequential(nn.Linear(9, 4),
                      nn.Linear(4, 2),
                      nn.Linear(2, 1))

total = 0

# Calculate the number of parameters in the model
for p in model.parameters():
  total += p.numel()
  
print(f"The number of parameters in the model is {total}")
#####################
input_tensor = torch.tensor([[2.4]])

# Create a sigmoid function and apply it on input_tensor
sigmoid = nn.Sigmoid()
probability = sigmoid(input_tensor)
print(probability)
#############################
input_tensor = torch.tensor([[1.0, -6.0, 2.5, -0.3, 1.2, 0.8]])

# Create a softmax function and apply it on input_tensor
softmax = nn.Softmax(dim = -1)
probabilities = softmax(input_tensor)
print(probabilities)
#################################
import torch
import torch.nn as nn

input_tensor = torch.Tensor([[3, 4, 6, 2, 3, 6, 8, 9]])

# Implement a small neural network for binary classification
model = nn.Sequential(
  nn.Linear(8 , 1),
  nn.Sigmoid()
)

output = model(input_tensor)
print(output)
########################
import torch
import torch.nn as nn

input_tensor = torch.Tensor([[3, 4, 6, 7, 10, 12, 2, 3, 6, 8, 9]])

# Implement a neural network with exactly four linear layers
model = nn.Sequential(
  nn.Linear(11 , 11),
  nn.Linear(11 , 11),
  nn.Linear(11 , 11),
  nn.Linear(11 , 1)
)

output = model(input_tensor)
print(output)
##########################
import torch
import torch.nn as nn

input_tensor = torch.Tensor([[3, 4, 6, 7, 10, 12, 2, 3, 6, 8, 9]])

# Update network below to perform a multi-class classification with four labels
model = nn.Sequential(
  nn.Linear(11, 20),
  nn.Linear(20, 12),
  nn.Linear(12, 6),
  nn.Linear(6, 4),
  nn.Softmax(dim = -1)
)

output = model(input_tensor)
print(output)
###################################
import torch.nn.functional as F 
y = 1
num_classes = 3

# Create the one-hot encoded vector using NumPy
one_hot_numpy = np.array([ 0 , 1 , 0 ])

# Create the one-hot encoded vector using PyTorch
one_hot_pytorch = F.one_hot(torch.tensor(y), num_classes = 3)

print("One-hot vector using NumPy:", one_hot_numpy)
print("One-hot vector using PyTorch:", one_hot_pytorch)
######################################
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

y = [2]
scores = torch.tensor([[0.1, 6.0, -2.0, 3.2]])

# Create a one-hot encoded vector of the label y
one_hot_label = F.one_hot(torch.tensor(y), num_classes = 4)
####################
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

y = [2]
scores = torch.tensor([[0.1, 6.0, -2.0, 3.2]])

# Create a one-hot encoded vector of the label y
one_hot_label = F.one_hot(torch.tensor(y), num_classes=4)

# Create the cross entropy loss function
criterion = CrossEntropyLoss()
####################################
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

y = [2]
scores = torch.tensor([[0.1, 6.0, -2.0, 3.2]])

# Create a one-hot encoded vector of the label y
one_hot_label = F.one_hot(torch.tensor(y), num_classes=4)

# Create the cross entropy loss function
criterion = CrossEntropyLoss()

# Calculate the cross entropy loss
loss = criterion(scores.double() , one_hot_label.double())
print(loss)
##############################
model = nn.Sequential(nn.Linear(16, 8),
                      nn.Linear(8, 2)
                     )

# Access the weight of the first linear layer
weight_0 = model[0].weight
print("Weight of the first layer:", weight_0)

# Access the bias of the second linear layer
bias_1 = model[1].bias
print("Bias of the second layer:", bias_1)
##########################
weight0 = model[0].weight
weight1 = model[1].weight
weight2 = model[2].weight

# Access the gradients of the weight of each linear layer
grads0 = weight0.grad
grads1 = weight1.grad
grads2 = weight2.grad
#################################
weight0 = model[0].weight
weight1 = model[1].weight
weight2 = model[2].weight

# Access the gradients of the weight of each linear layer
grads0 = weight0.grad
grads1 = weight1.grad
grads2 = weight2.grad

# Update the weights using the learning rate and the gradients
weight0 = weight0 - lr * grads0
weight1 = weight1 - lr * grads1
weight2 = weight2 - lr * grads2
##########################
import torch.optim as o
# Create the optimizer
optimizer = o.SGD(model.parameters() , lr = 0.00001)
##########################
# Create the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)

loss = criterion(pred, target)
loss.backward()

# Update the model's parameters using the optimizer
optimizer.step()
#########################
import torch as T
from torch.utils.data import TensorDataset as TD

X = animals.iloc[:, 1:-1].to_numpy()  
y = animals.iloc[:, -1].to_numpy()

# Create a dataset
dataset = TD(T.tensor(X , dtype=T.float) , T.tensor(y , dtype=T.long))

# Print the first sample
input_sample, label_sample = dataset[0]
print('Input sample:', input_sample)
print('Label sample:', label_sample)
##################################
from torch.utils.data import DataLoader as DL

# Create a DataLoader
dataloader = DL(dataset , batch_size = 2 , shuffle = True)

# Iterate over the dataloader
for batch_inputs, batch_labels in dataloader:
    print('batch_inputs:', batch_inputs)
    print('batch_labels:', batch_labels)