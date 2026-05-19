# Initiate the Lunar Lander environment
env = gym.make("LunarLander-v2")

class Network(nn.Module):
    def __init__(self, dim_inputs, dim_outputs):
        super(Network, self).__init__()
        # Define a linear transformation layer 
        self.linear = nn.Linear(dim_inputs , dim_outputs)
    def forward(self, x):
        return self.linear(x)

# Instantiate the network
network = Network(8 , 4)

# Initialize the optimizer
optimizer = optim.Adam(network.parameters(), lr=0.0001)

print("Network initialized as:\n", network)