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
####################
print(env.action_space)
###################
print(env.observation_space)
#####################
print(env.action_space.n,env.observation_space.n)
#######################
print(env.unwrapped.P)
##########################
# Create the Cliff Walking environment
env = gym.make('CliffWalking')

# Compute the size of the action space
num_actions = env.action_space.n

# Compute the size of the state space
num_states = env.observation_space.n

print("Number of actions:", num_actions)
print("Number of states:", num_states)
##############################
# Choose the state
state = 35

# Extract transitions for each state-action pair
for action in range(num_actions):
    transitions = env.unwrapped.P[state][action]
    # Print details of each transition
    for transition in transitions:
        probability , next_state , reward , done = transition
        print(f"Probability: {probability}, Next State: {next_state}, Reward: {reward}, Done: {done}")
#################################
# Create the environment
env = gym.make('MyGridWorld' , render_mode = 'rgb_array')
state, info = env.reset()

# Define the policy
policy = {0 : 2 , 1 : 2 , 2 : 1 , 5 : 0 , 4 : 0 , 3 : 1 , 6 : 2 , 7 : 2}
##########################
# Create the environment
env = gym.make('MyGridWorld', render_mode='rgb_array')
state, info = env.reset()

# Define the policy
policy = {0:2, 1:2, 2:1, 3:1, 4:0, 5:0, 6:2, 7:2}

terminated = False
while not terminated:
  # Select action based on policy 
  action = policy[state]
  state, reward, terminated, truncated, info = env.step(action)
  # Render the environment
  render()
#############################
# Complete the function
def compute_state_value(state):

    if state == terminal_state:
        return 0

    action = policy[state]
    _ , next_state , reward , _ = env.unwrapped.P[state][action][0] 
    #print(env.unwrapped.P[state][action][0])
    return reward + gamma * compute_state_value(next_state)

# Compute all state values 
state_values = {state : compute_state_value(state) for state in range(num_states)}
print(state_values)
####################
value_function_1 = {0: 1, 1: 2, 2: 3, 3: 7, 4: 6, 5: 4, 6: 8, 7: 10, 8: 0}
value_function_2 = {0: 7, 1: 8, 2: 9, 3: 7, 4: 9, 5: 10, 6: 8, 7: 10, 8: 0}

# Check for each value in policy 1 if it is better than policy 2
one_is_better = [value_function_1[state] >= value_function_2[state] for state in range(num_states)]

# Check for each value in policy 2 if it is better than policy 1
two_is_better = [value_function_2[state] >= value_function_1[state] for state in range(num_states)]

if all(one_is_better):
  print("Policy 1 is better.")
elif all(two_is_better):
  print("Policy 2 is better.")
else:
  print("Neither policy is uniformly better across all states.")
############################
# Complete the function to compute the action-value for a state-action pair
def compute_q_value(state, action):
    if state == terminal_state:
        return None   
    probability, next_state, reward, done = env.unwrapped.P[state][action][0]
    return reward + gamma * compute_state_value(next_state)

# Compute Q-values for each state-action pair
Q = {(state , action): compute_q_value(state , action) for state in range(num_states) for action in range(num_actions)}

print(Q)
############################
improved_policy = {}

for state in range(num_states-1):
    # Find the best action for each state based on Q-values
    max_action = max(range(num_actions) , key = lambda action : Q[(state,action)])
    improved_policy[state] = max_action

terminated = False
while not terminated:
  # Select action based on policy 
  action = improved_policy[max_action]
  # Execute the action
  state, reward, terminated, truncated, info = env.step(action)
  render()
###############################
# Complete the policy evaluation function
def policy_evaluation(policy):
    V = {state :  compute_state_value(state , policy) for state in range(num_states)}
    return V
########################
def policy_improvement(policy):
    improved_policy = {s: 0 for s in range(num_states-1)}
    
 # Compute the Q-value for each state-action pair
    Q = {(state, action): compute_q_value(state , action , policy) for state in range(num_states) for action in range(num_actions)}
            
    # Compute the new policy based on the Q-values
    for state in range(num_states - 1):
        max_action = max(range(num_actions) , key = lambda action : Q[(state,action)])
        improved_policy[state] = max_action
        
    return improved_policy
#############################
# Complete the policy iteration function
def policy_iteration():
    policy = {0:2, 1:2, 2:1, 3:1, 4:0, 5:0, 6:2, 7:2}
    while True:
        V = policy_evaluation(policy)
        improved_policy = policy_improvement(policy)
        if improved_policy == policy:
            break
        policy = improved_policy
    
    return policy, V

policy, V = policy_iteration()
render_policy(policy)
###################
threshold = 0.001
while True:
  new_V = {state : 0 for state in range(num_states)}
  for state in range(num_states-1):
    # Get action with maximum Q-value and its value 
    max_action, max_q_value = get_max_action_and_value(state, V)
    # Update the value function and policy
    new_V[state] = max_q_value
    policy[state] = max_action
  # Test if change in state values is negligeable
  if all(abs(new_V[state] - V[state]) < threshold for state in V):
    break
  V = new_V
render_policy(policy)
###########################
def generate_episode():
    episode = []
    # Reset the environment
    state, info = env.reset(seed = 42)
    terminated = False
    while not terminated:
      # Select a random action
      action = env.action_space.sample()
      next_state, reward, terminated, truncated, info = env.step(action)
      render()
      # Update episode data
      episode.append((state , action , reward))
      state = next_state
    return episode
print(generate_episode())
#########################
for i in range(100):
  episode = generate_episode()
  visited_states = set()
  for j, (state, action, reward) in enumerate(episode):
    # Define the first-visit condition
    if (state , action) not in visited_states:
      # Update the returns, their counts and the visited states
      returns_sum[state, action] += sum([y[2] for y in episode[j:]])
      returns_count[state, action] += 1
      visited_states.add((state,action))

nonzero_counts = returns_count != 0

Q[nonzero_counts] = returns_sum[nonzero_counts] / returns_count[nonzero_counts]
render_policy(get_policy())
##############################
Q = np.zeros((num_states, num_actions))
for i in range(100):
  # Generate an episode
  episode = generate_episode()
  # Update the returns and their counts
  for j, (state, action, reward) in enumerate(episode):
    returns_sum[(state,  action)] += sum([y[2] for y in episode[j:]])
    returns_count[(state,  action)] += 1

# Update the Q-values for visited state-action pairs 
nonzero_counts = returns_count != 0
Q[nonzero_counts] = returns_sum[nonzero_counts] / returns_count[nonzero_counts]
    
render_policy(get_policy())
######################
def update_q_table(state, action, reward, next_state, next_action):
  	# Get the old value of the current state-action pair
    old_value = Q[(state,action)]
    # Get the value of the next state-action pair
    next_value = Q[(next_state,next_action)]
    # Compute the new value of the current state-action pair
    Q[(state, action)] = (1 - alpha) * old_value + alpha * (reward + gamma * next_value)

alpha = 0.1
gamma  = 0.8
Q = np.array([[10,0],[0,20]], dtype='float32')
# Update the Q-table for the ('state1', 'action1') pair
update_q_table(0 , 0 , 5 , 1 , 1)
print(Q)
#########################
for episode in range(num_episodes):
    state, info = env.reset()
    action = env.action_space.sample()
    terminated = False
    while not terminated:
      	# Execute the action
        next_state, reward, terminated, truncated, info = env.step(action)
        # Choose the next action randomly
        next_action = env.action_space.sample()
        # Update the Q-table
        update_q_table(state , action , reward , next_state , next_action)
        state, action = next_state, next_action   
render_policy(get_policy())
###################################
actions = ['action1', 'action2'] 
def update_q_table(state, action, reward, next_state):
  	# Get the old value of the current state-action pair
    old_value = Q[(state , action)]
    # Determine the maximum Q-value for the next state
    next_max =  max(Q[next_state])
    # Compute the new value of the current state-action pair
    Q[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

alpha = 0.1
gamma = 0.95
Q = np.array([[10, 8], [20, 15]], dtype='float32')
# Update the Q-table
update_q_table(0 , 0 , 5 ,1)
print(Q)
##########################
for episode in range(10000):
    state, info = env.reset()
    total_reward = 0
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        # Execute the action
        next_state, reward, terminated, truncated, info = env.step(action)
        # Update the Q-table
        update_q_table(state , action , reward , next_state)
        state = next_state
        total_reward += reward
    # Append the total reward to the rewards list    
    rewards_per_episode.append(reward)
print("Average reward per random episode: ", np.mean(rewards_per_episode))
##############################
for episode in range(10000):
    state, info = env.reset()
    terminated = False
    episode_reward = 0
    while not terminated:
        # Select the best action based on learned Q-table
        action = policy[state]
        new_state, reward, terminated, truncated, info = env.step(action)
        state = new_state
        episode_reward += reward
    reward_per_learned_episode.append(episode_reward)
# Compute and print the average reward per learned episode
avg_reward_per_learned_episode = np.mean(reward_per_learned_episode)
print("Average reward per learned episode: ", reward_per_learned_episode)
print("Average reward per random episode: ", avg_reward_per_learned_episode)
############################
def update_q_table(state, action, next_state, reward):
  	# Calculate the expected Q-value for the next state
    expected_q = np.mean(Q[next_state])
    # Update the Q-value for the current state and action
    Q[state, action] = (1 - alpha) * Q[state , action] + alpha * (reward + gamma * expected_q) 
    
Q = np.random.rand(5, 2)
print("Old Q:\n", Q)
alpha = 0.1
gamma = 0.99

# Update the Q-table
update_q_table(state = 2 , action = 1 , next_state = 3 , reward = 5)
print("Updated Q:\n", Q)
#################################
# Initialize the Q-table with random values
Q = np.zeros((env.observation_space.n , env.action_space.n))
for i_episode in range(num_episodes):
    state, info = env.reset()    
    done = False    
    while not done: 
        action = env.action_space.sample()               
        next_state, reward, done, truncated, info = env.step(action)
        # Update the Q-table
        update_q_table(state , action , next_state , reward)
        state = next_state
# Derive policy from Q-table        
policy = {state: np.argmax(Q[state]) for state in range(env.observation_space.n)}
render_policy(policy)
#########################
Q = [np.random.rand(8,4), np.random.rand(8,4)] 
def update_q_tables(state, action, reward, next_state):
  	# Get the index of the table to update
    i = np.random.randint(2)
    # Update Q[i]
    best_next_action = np.argmax(Q[i][state])
    Q[i][state, action] = (1 - alpha) * Q[i][state , action] + alpha * (reward + gamma * q[1 - i][next_state , best_next_action])
#############################
Q = [np.zeros((num_states, num_actions))] * 2
for episode in range(num_episodes):
    state, info = env.reset()
    terminated = False   
    while not terminated:
        action = np.random.choice(num_actions)
        next_state, reward, terminated, truncated, info = env.step(action)
        # Update the Q-tables
        update_q_tables(state , action , reward , next_state)
        state = next_state
# Combine the learned Q-tables        
Q = Q[0] + Q[1]
policy = {state: np.argmax(Q[state]) for state in range(num_states)}
render_policy(policy)
##################
epsilon = 0.2
env = gym.make('FrozenLake')
q_table = np.random.rand(env.observation_space.n, env.action_space.n)

def epsilon_greedy(state):
    # Implement the condition to explore
    if np.random.rand() < epsilon:
      	# Choose a random action
        action = env.action_space.sample()
    else:
      	# Choose the best action according to q_table
        action = np.argmax(q_table[state , :])
    return action
##########################3
rewards_eps_greedy = []
for episode in range(total_episodes):
    state, info = env.reset()
    episode_reward = 0
    for i in range(max_steps):
      	# Select action with epsilon-greedy strategy
        action = epsilon_greedy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        # Accumulate reward
        episode_reward += reward      
        update_q_table(state, action, reward, next_state)      
        state = next_state
    # Append the toal reward to the rewards list 
    rewards_eps_greedy.append(episode_reward)
print("Average reward per episode: ", np.mean(rewards_eps_greedy))
#########################
rewards_decay_eps_greedy = []
for episode in range(total_episodes):
    state, info = env.reset()
    episode_reward = 0
    for i in range(max_steps):
      	# Implement the training loop
        action = epsilon_greedy(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        update_q_table(state, action, reward, new_state)            
        state = new_state
        

    rewards_decay_eps_greedy.append(episode_reward)
    # Update epsilon
    epsilon =  max(min_epsilon , epsilon * epsilon_decay)
print("Average reward per episode: ", np.mean(rewards_decay_eps_greedy))
################################
def create_multi_armed_bandit(n_bandits):
  	# Generate the true bandits probabilities
    true_bandit_probs = np.random.rand(n_bandits) 
    # Create arrays that store the count and value for each bandit
    counts = np.zeros(n_bandits) 
    values = np.zeros(n_bandits)  
    # Create arrays that store the rewards and selected arms each episode
    rewards = np.zeros(n_iterations)
    selected_arms = np.zeros(n_iterations , dtype = int)
    return true_bandit_probs, counts, values, rewards, selected_arms
####################################
# Create a 10-armed bandit
true_bandit_probs, counts, values, rewards, selected_arms = create_multi_armed_bandit(10)

for i in range(n_iterations): 
  	# Select an arm
    arm = epsilon_greedy()
    # Compute the received reward
    reward = np.random.rand() < true_bandit_probs[arm]
    rewards[i] = reward
    selected_arms[i] = arm
    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]
    # Update epsilon
    epsilon = max(min_epsilon , epsilon * epsilon_decay)
########################
# Initialize the selection percentages with zeros
selections_percentage = np.zeros((n_iterations , n_bandits))
for i in range(n_iterations):
    selections_percentage[i, selected_arms[i]] = 1
# Compute the cumulative selection percentages 
selections_percentage = np.cumsum(selections_percentage, axis= 0) / np.arange(1, n_iterations + 1).reshape(-1, 1)
for arm in range(n_bandits):
  	# Plot the cumulative selection percentage for each arm
    plt.plot(selections_percentage[:,arm] , label=f'Bandit #{arm+1}')
plt.xlabel('Iteration Number')
plt.ylabel('Percentage of Bandit Selections (%)')
plt.legend()
plt.show()
for i, prob in enumerate(true_bandit_probs, 1):
    print(f"Bandit #{i} -> {prob:.2f}")