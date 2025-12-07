train_dataset.classes
#########################
class BinaryImageClassifier(nn.Module):
    def __init__(self):
        super(BinaryImageClassifier, self).__init__()
        
        # Create a convolutional layer
        self.conv1 = nn.Conv2d(3 , 16 , 3 , 1 ,1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        # Create a fully connected layer
        self.fc = nn.Linear(16 * 32 * 32 , 1)
        
        # Create an activation function
        self.sigmoid = nn.Sigmoid()
###################################
class MultiClassImageClassifier(nn.Module):
  
    # Define the init method
    def __init__(self, num_classes):
        super(MultiClassImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Create a fully connected layer
        self.fc = nn.Linear(16 * 32 * 32 , num_classes)
        
        # Create an activation function
        self.softmax = nn.Softmax(dim = 1)
#####################################

channels = F.get_image_num_channels(image)
print(channels)
#################################
# Create a model
model = CNNModel()
print("Original model: ", model)

# Create a new convolutional layer
conv2 = nn.Conv2d(32 , 32 , stride = 1 , padding = 1 , kernel_size = 3)

# Append the new layer to the model
model.add_module("conv2",conv2)
print("Extended model: ", model)
#########################################
class BinaryImageClassification(nn.Module):
  def __init__(self):
    super(BinaryImageClassification, self).__init__()
    # Create a convolutional block
    self.conv_block = nn.Sequential(
      nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
    )
    
  def forward(self, x):
    # Pass inputs through the convolutional block
    x = self.conv_block(x)
    return x
##############################
# Save the model
torch.save(model.state_dict(), 'ModelCNN.pth')

# Create a new model
loaded_model = ManufacturingCNN()

# Load the saved model
loaded_model.load_state_dict(torch.load('ModelCNN.pth'))
print(loaded_model)
######################################
# Import resnet18 model
from torchvision.models import resnet18 , ResNet18_Weights

# Initialize model with default weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights = weights)

# Set model to evaluation mode
model.eval()

# Initialize the transforms
transform = weights.transforms()
########################################
# Apply preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Apply model with softmax layer
prediction = model(batch).squeeze(0).softmax(0)

# Apply argmax
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(category_name)
###########################################
# Convert bbox into tensors
bbox_tensor = torch.tensor(bbox)

# Add a new batch dimension
bbox_tensor = bbox_tensor.unsqueeze(0)

# Resize image and transform tensor
transform = transforms.Compose([
  transforms.Resize(224),
  transforms.PILToTensor()
])

# Apply transform to image
image_tensor = transform(image)
print(image_tensor)
######################################
# Import draw_bounding_boxes
from torchvision.utils import draw_bounding_boxes

# Define the bounding box coordinates
bbox = [x_min , y_min , x_max , y_max]
bbox_tensor = torch.tensor(bbox).unsqueeze(0)

# Implement draw_bounding_boxes
img_bbox = draw_bounding_boxes(image_tensor , bbox_tensor , width=3, colors="red")

# Tranform tensors to image
transform = transforms.Compose([
    transforms.ToPILImage()
])
plt.imshow(transform(img_bbox))
plt.show()
####################
iou = box_iou(box_a , bbox)
print(
iou)
ioub = box_iou(box_b , bbox)
print(ioub)
iouc = box_iou(box_c , bbox)
print(iouc)
##################################
# Get model's prediction
with torch.no_grad():
    output = model(test_image)

# Extract boxes from the output
boxes = output[0]['boxes']

# Extract scores from the output
scores = output[0]['scores']

print(boxes, scores)
###################################
# Import nms
from torchvision.ops import nms

# Set the IoU threshold
iou_threshold = 0.5

# Apply non-max suppression
box_indices = nms(boxes , scores , iou_threshold)

# Filter boxes
filtered_boxes = box_indices[0]

print("Filtered Boxes:", filtered_boxes)
#########################################
# Load pretrained weights
vgg_model = vgg16(weights = VGG16_Weights.DEFAULT)

# Extract the input dimension
input_dim = nn.Sequential(*list(vgg_model.classifier.children()))[0].in_features

# Create a backbone with convolutional layers
backbone = nn.Sequential(*list(vgg_model.classifier.children()))

# Print the backbone model
print(backbone)
##################################
# Create a variable with the number of classes
num_classes = 2
    
# Create a sequential block
classifier = nn.Sequential(
	# Create a linear layer with input features
	nn.Linear(in_features = input_dim, out_features = 512),
	nn.ReLU(),
	# Add the output dimension to the classifier
	nn.Linear(512, num_classes),
)
###########################################
# Define the number of coordinates
num_coordinates = 4

bb = nn.Sequential(  
	# Add input and output dimensions
	nn.Linear(input_dim , 32),
	nn.ReLU(),
	# Add the output for the last regression layer
	nn.Linear(32, num_coordinates),
)