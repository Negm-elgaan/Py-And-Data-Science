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
######################################
y_pred = np.array([3, 5.0, 2.5, 7.0])  
y = np.array([3.0, 4.5, 2.0, 8.0])     

# Calculate MSE using NumPy
mse_numpy = np.mean((y_pred - y) ** 2)

# Create the MSELoss function in PyTorch
criterion = nn.MSELoss()

# Calculate MSE using PyTorch
mse_pytorch = criterion(torch.tensor(y_pred), torch.tensor(y))

print("MSE (NumPy):", mse_numpy)
print("MSE (PyTorch):", mse_pytorch)
# Loop over the number of epochs and then the dataloader
for i in range(num_epochs):
  for data in dataloader:
    # Set the gradients to zero
    optimizer.zero_grad()
##########################
# Loop over the number of epochs and the dataloader
for i in range(num_epochs):
  for data in dataloader:
    # Set the gradients to zero
    optimizer.zero_grad()
    # Run a forward pass
    feature, target = data
    prediction = model(feature)
    # Compute the loss
    loss = criterion(prediction , target)
    # Compute the gradients
    loss.backward()
##############################
# Loop over the number of epochs and the dataloader
for i in range(num_epochs):
  for data in dataloader:
    # Set the gradients to zero
    optimizer.zero_grad()
    # Run a forward pass
    feature, target = data
    prediction = model(feature)    
    # Compute the loss
    loss = criterion(prediction, target)    
    # Compute the gradients
    loss.backward()
    # Update the model's parameters
    optimizer.step()
show_results(model, dataloader)
##############################
# Create a ReLU function with PyTorch
relu_pytorch = nn.ReLU()
###########################
# Try a first learning rate value
lr0 = 0.001
optimize_and_plot(lr=lr0)
# Try a second learning rate value
lr1 = 1
optimize_and_plot(lr=lr1)
###################
# Try a third learning rate value
lr2 = 0.088
optimize_and_plot(lr=lr2)
########################
# Try a first value for momentum
mom0 = 0.8999
optimize_and_plot(momentum=mom0)
##################################
# Try a second value for momentum
mom1 = 0.918
optimize_and_plot(momentum=mom1)
######################
for name, param in model.named_parameters():
  
    # Check for first layer's weight
    if name == '0.weight':
   
        # Freeze this weight
        param.requires_grad = False
        
    # Check for second layer's weight
    if name == '1.weight':
      
        # Freeze this weight
        param.requires_grad = False
##############################
layer0 = nn.Linear(16, 32)
layer1 = nn.Linear(32, 64)

# Use uniform initialization for layer0 and layer1 weights
nn.init.uniform_(layer0.weight)
nn.init.uniform_(layer1.weight)

model = nn.Sequential(layer0, layer1)
###################################
# Set the model to evaluation mode
model.eval()
validation_loss = 0.0

with torch.no_grad():
  
  for features, labels in validationloader:
    
      outputs = model(features)
      loss = criterion(outputs, labels)
      
      # Sum the current loss to the validation_loss variable
      validation_loss += loss.item()
########################
# Set the model to evaluation mode
model.eval()
validation_loss = 0.0

with torch.no_grad():
  for features, labels in validationloader:
      outputs = model(features)
      loss = criterion(outputs, labels)
      # Sum the current loss to the validation_loss variable
      validation_loss += loss.item()
      
# Calculate the mean loss value
validation_loss_epoch = validation_loss / len(validationloader)
print(validation_loss_epoch)

# Set the model back to training mode
model.train()
############################
# Create accuracy metric
metric = torchmetrics.Accuracy(task = "multiclass"  , num_classes = 3)
for features, labels in dataloader:
    outputs = model(features)
  
    # Calculate accuracy over the batch
    metric.update(outputs , labels.argmax(dim = -1))
#######################
# Create accuracy metric
metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)
for features, labels in dataloader:
    outputs = model(features)
    
    # Calculate accuracy over the batch
    metric.update(outputs, labels.argmax(dim=-1))
    
# Calculate accuracy over the whole epoch
accuracy = metric.compute()
print(f"Accuracy on all data: {accuracy}")

# Reset metric for the next epoch
metric.reset()
plot_errors(model, dataloader)
##############################
# Model with Dropout
model = nn.Sequential(
    nn.Linear(8, 6),
    nn.Linear(6, 4),
    nn.Dropout(p = 0.5))

# Forward pass in training mode (Dropout active)
model.train()
output_train = model(features)
####################
model = nn.Sequential(
    nn.Linear(8, 6),
    nn.Linear(6, 4),
    nn.Dropout(p=0.5))

model.train()
output_train = model(features)

# Forward pass in evaluation mode (Dropout disabled)
model.eval()
output_eval = model(features)

# Print results
print("Output in train mode:", output_train)
print("Output in eval mode:", output_eval)
##################################
values = []
for idx in range(10):
    # Randomly sample a learning rate factor between 2 and 4
    factor = np.random.uniform(2,4)
    lr = 10 ** -factor
    
    # Randomly select a momentum between 0.85 and 0.99
    momentum = np.random.uniform(0.85 , 0.99)
    
    values.append((lr, momentum))
       
plot_hyperparameter_search(values)