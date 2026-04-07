from transformers import AutoModelForSequenceClassification

# Load a pre-trained bert-base-uncased model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# Print the model's configuration
print(model.config)
#####################
from accelerate import Accelerator

# Declare an accelerator object
accelerator = Accelerator()

# Prepare the model for distributed training
model = accelerator.prepare(model)

print(accelerator.device)