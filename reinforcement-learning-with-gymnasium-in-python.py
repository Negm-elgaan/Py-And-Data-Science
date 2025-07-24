exp_rewards_strategy_1 = np.array([3, 2, -1, 5])

discount_factor = 0.9

# Compute discounts
discounts_strategy_1 = np.array([discount_factor ** i for i in range(len(exp_rewards_strategy_1))])

# Compute the discounted return
discounted_return_strategy_1 = np.sum(discounts_strategy_1)

print(f"The discounted return of the first strategy is {discounted_return_strategy_1}")
#####################
exp_rewards_strategy_1 = np.array([3, 2, -1, 5])

discount_factor = 0.9

# Compute discounts
discounts_strategy_2 = np.array([discount_factor ** i for i in range(len(exp_rewards_strategy_1))])

# Compute the discounted return
discounted_return_strategy_2 = np.sum(discounts_strategy_2)

print(f"The discounted return of the first strategy is {discounted_return_strategy_2}")
##################################
# Import the gymnasium library
import gymnasium as gym

# Create the environment
env = gym.make('MountainCar' , render_mode = 'rgb_array')

# Get the initial state
initial_state, info = env.reset(seed = 42)

position = initial_state[0]
velocity = initial_state[1]

print(f"The position of the car along the x-axis is {position} (m)")
print(f"The velocity of the car is {velocity} (m/s)")
#####################
env = gym.make('MountainCar', render_mode='rgb_array')
initial_state, _ = env.reset()

# Complete the render function
def render():
    state_image = env.render()
    plt.imshow(state_image)
    plt.show()

# Call the render function    
render()
###############
# Define the sequence of actions
actions = [1 , 1 , 2 , 2 , 1 , 2]

for action in actions:
  # Execute each action
  state, reward, terminated, _, _ = env.step(action)
  # Render the environment
  render()
  if terminated:
  	print("You reached the goal!")