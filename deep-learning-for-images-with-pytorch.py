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
##########################
###########################################
# Import AnchorGenerator
from torchvision.models.detection.rpn import AnchorGenerator

# Configure anchor size
anchor_sizes = ((32 , 64 , 128),)

# Configure aspect ratio
aspect_ratios = ((0.5 , 1.0 , 2.0),)

# Instantiate AnchorGenerator
rpn_anchor_generator = AnchorGenerator(anchor_sizes , aspect_ratios)
###################################################
# Import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN as FRCNN
from torchvision.ops import MultiScaleRoIAlign

# Instantiate RoI pooler
roi_pooler = MultiScaleRoIAlign(
	featmap_names = ["0"],
	output_size = 7,
	sampling_ratio = 2,
)

mobilenet = torchvision.models.mobilenet_v2(weights="DEFAULT")
backbone = nn.Sequential(*list(mobilenet.features.children()))
backbone.out_channels = 1280

# Create Faster R-CNN model
model = FRCNN(
	backbone = backbone,
	num_classes = backbone.out_channels,
	anchor_generator = anchor_generator,
	box_roi_pool = roi_pooler,
)
##############################
# Implement the RPN classification loss function
rpn_cls_criterion = nn.BCEWithLogitsLoss()

# Implement the RPN regression loss function
rpn_reg_criterion = nn.MSELoss()

# Implement the R-CNN classification Loss function
rcnn_cls_criterion = nn.CrossEntropyLoss()

# Implement the R-CNN regression loss function
rcnn_reg_criterion = nn.MSELoss()
#################################
# Load mask image
mask = Image.open('annotations/Egyptian_Mau_123.png')

# Transform mask to tensor
transform = transforms.Compose([transforms.ToTensor()])
mask_tensor = transform(mask)

# Create binary mask
binary_mask = torch.where(
    mask_tensor == (1 / 255) , 
    torch.tensor(1.0) ,
    torch.tensor(0.0) ,
)

# Print unique mask values
print(binary_mask.unique())
#####################################################################################
# Load image and transform to tensor
image = Image.open("images/Egyptian_Mau_123.jpg")
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image)

# Segment object out of the image
object_tensor = image_tensor * binary_mask

# Convert segmented object to image and display
to_pil_image = transforms.Compose([transforms.ToPILImage()])
object_image = to_pil_image(object_tensor)
plt.imshow(object_image)
plt.show()
#############################
from torchvision.models.detection import maskrcnn_resnet50_fpn as      mrcnn_rn50_fpn
# Load a pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained = True)
model.eval()

# Load an image and convert to a tensor
image = Image.open("two_cats.jpg")
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0)

# Perform inference
with torch.no_grad():
    prediction = model(image_tensor)
    print(prediction)
########################################
# Extract masks and labels from prediction
masks = prediction[0]['masks']
labels = prediction[0]['labels']

# Plot image with two overlaid masks
for i in range(2):
    plt.imshow(image)
    # Overlay the i-th mask on top of the image
    plt.imshow(masks[i , 0], 'jet', alpha = 0.5)
    plt.title(f"Object: {class_names[labels[i]]}")
    plt.show()
###############################
class Generator(nn.Module):
  def __init__(self , in_dim , out_dim):
    super(Generator,self).__init__()
    self.generator = nn.Sequential(
      gen_block(input_dim , 256) ,
      gen_block(256 , 512) ,
      gen_block(512 , 1024) , 
      nn.Linear(1024 , out_dim) ,
      nn.sigmoid(out_dim)
    )
  
  def forward(x):
    self.generator(x)
 #########################
class Discriminator(nn.Module):
    def __init__(self , in_dim):
        super(Discriminator , self).__init__()
        self.disc = nn.Sequentia(
            disc_block(in_dim , 1024) ,
            disc_block(1024 , 512) , 
            disc_block(512 , 256) ,
            nn.Linear(256 , 1)
        )
    
    def forward(x):
        self.disc(x)
#######################
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Define the decoder blocks
        self.dec1 = self.conv_block(512 , 256)
        self.dec2 = self.conv_block(256  , 128)
        self.dec3 = self.conv_block(128 , 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
#########################
def forward(self, x):
    x1 = self.enc1(x)
    x2 = self.enc2(self.pool(x1))
    x3 = self.enc3(self.pool(x2))
    x4 = self.enc4(self.pool(x3))

    x = self.upconv3(x4)
    x = torch.cat([x, x3], dim=1)
    x = self.dec1(x)

    x = self.upconv2(x)
    x = torch.cat([x, x2], dim=1)
    x = self.dec2(x)

    # Define the last decoder block with skip connections
    x = self.upconv1(x)
    x = torch.cat([x , x1] , dim = 1)
    x = self.dec3(x)

    return self.out(x)