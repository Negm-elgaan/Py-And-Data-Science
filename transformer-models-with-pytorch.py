# Define the transformer model
model = nn.Transformer(nhead = 8 , d_model = 1536 , num_encoder_layers = 6 , num_decoder_layers = 6)

# Print the model object
print(model)
#######################################
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        # Set the model dimensionality and vocabulary size
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Instantiate the embedding layer
        self.embedding = nn.Embedding(vocab_size , d_model)

    def forward(self, x):
        # Return the embeddings multiplied by the square root of d_model
        return self.embedding(x) * self.d_model ** 0.5

# Instantiate InputEmbeddings and apply it to token_ids
embedding_layer = InputEmbeddings(10000 , 512)
output = embedding_layer(token_ids)
print(output.shape)