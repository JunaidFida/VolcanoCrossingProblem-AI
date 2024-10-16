import numpy as np
import ipywidgets as widgets
from IPython.display import display

grid_size = (4, 3)
end_states = [(1, 2), (3, 1)]
dangerous_states = [(1, 1), (2, 2)]
slip_probability = 0.1
alpha = 0.2
gamma = 0.9

# Initialize Q-values
Q_values = np.zeros((grid_size[0], grid_size[1], 4))

def take_action(state, action, grid_size):
    # All possible actions
    MOVE_N = 0
    MOVE_S = 1
    MOVE_W = 2
    MOVE_E = 3

    # current position
    current_row, current_col = state

    # Simulate action
    if action == MOVE_N and current_row > 0:
        next_state = (current_row - 1, current_col)
    elif action == MOVE_S and current_row < grid_size[0] - 1:
        next_state = (current_row + 1, current_col)
    elif action == MOVE_W and current_col > 0:
        next_state = (current_row, current_col - 1)
    elif action == MOVE_E and current_col < grid_size[1] - 1:
        next_state = (current_row, current_col + 1)
    else:
        # If the action is not possible, stay in the current state
        next_state = state 

    return next_state

def calculate_reward(state, end_states, dangerous_states):
    if state in end_states:
        reward = 20.0  # Positive reward
    elif state in dangerous_states:
        reward = -50.0  # Negative reward
    else:
        reward = 0.0  # No additional reward for being in a safe state

    return reward

def run_sarsa(epsilon_value, num_episodes_value, slip_probability_value):
    # Update parameters
    epsilon = epsilon_value
    num_episodes = num_episodes_value
    slip_probability = slip_probability_value

    # SARSA algorithm implementation
    for episode in range(num_episodes):
        # Initialize state
        state = (0, 0)

        # Choose action based on epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q_values[state[0], state[1]])

        while state not in end_states:
            # Simulate slip
            if np.random.rand() < slip_probability:
                next_action = np.random.randint(4)
            else:
                next_action = np.argmax(Q_values[state[0], state[1]])

            # Take action and observe next state and reward
            next_state = take_action(state, action, grid_size)
            reward = calculate_reward(next_state, end_states, dangerous_states)

            # Update Q-value based on SARSA
            Q_values[state[0], state[1], action] = (1 - alpha) * Q_values[state[0], state[1], action] + \
                                                   alpha * (reward + gamma * Q_values[next_state[0], next_state[1], next_action])

            # Move to the next state and action
            state = next_state
            action = next_action

    # Calculate average utility for different episodes of uniformly random policy
    avg_utility = calculate_average_utility(1000)  # You can change the number of episodes here
    print(f"Average Utility for {num_episodes} episodes: {avg_utility}")

    # Display the Q-values
    print("Q-values:")
    print(Q_values)

    # Display the utility values below the grid
    display_utility()

def calculate_average_utility(num_episodes):
    total_utility = 0

    for _ in range(num_episodes):
        state = (0, 0)
        episode_utility = 0

        while state not in end_states:
            action = np.random.randint(4)
            next_state = take_action(state, action, grid_size)
            reward = calculate_reward(next_state, end_states, dangerous_states)
            episode_utility += reward
            state = next_state

        total_utility += episode_utility

    return total_utility / num_episodes

def display_utility():
    # Calculate utility values as the maximum Q-value for each state
    utility_values = np.max(Q_values, axis=2)

    # Display utility values below the grid
    print("Utility values:")
    print(utility_values)

# Create dropdown menus for user input
epsilon_dropdown = widgets.Dropdown(
    options=np.arange(0.01, 1.01, 0.01),
    value=0.01,
    description='Epsilon:'
)

num_episodes_dropdown = widgets.Dropdown(
    options=np.arange(100, 5100, 100),
    value=1000,
    description='Number of Episodes:'
)

slip_probability_dropdown = widgets.Dropdown(
    options=np.arange(0.0, 0.31, 0.01),
    value=0.1,
    description='Slip Probability:'
)

# Run SARSA with widgets
widgets.interact(run_sarsa, epsilon_value=epsilon_dropdown, num_episodes_value=num_episodes_dropdown, slip_probability_value=slip_probability_dropdown);
