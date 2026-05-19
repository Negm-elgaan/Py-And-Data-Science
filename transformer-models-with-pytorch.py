# Define the transformer model
model = nn.Transformer(nhead = 8 , d_model = 1536 , num_encoder_layers = 6 , num_decoder_layers = 6)

# Print the model object
print(model)