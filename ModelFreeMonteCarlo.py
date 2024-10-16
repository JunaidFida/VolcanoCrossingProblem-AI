import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np

grid_size = (4, 3)
end_states = [(1, 2), (3, 1)]
dangerous_states = [(1, 1), (2, 2)]
slip_probability = 0.1
epsilon = 0.1
gamma = 0.9

# Initialize Q-values
Q_values = np.zeros((grid_size[0], grid_size[1], 4))
returns = np.zeros((grid_size[0], grid_size[1], 4, 2))  # (sum of returns, count of visits)

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

def run_monte_carlo(btn):
    slip_probability = float(slip_entry.value)
    epsilon = float(epsilon_entry.value)
    num_episodes = int(episodes_entry.value)

    average_utilities = []

    for num_ep in range(1, num_episodes + 1):
        # Monte Carlo algorithm implementation
        for episode in range(num_ep):
            episode_states = []
            episode_actions = []
            episode_rewards = []

            # Initialize state
            state = (0, 0)

            while state not in end_states:
                # Choose action based on epsilon-greedy policy
                if np.random.rand() < epsilon:
                    action = np.random.randint(4)
                else:
                    action = np.argmax(Q_values[state[0], state[1]])

                # Simulate slip
                if np.random.rand() < slip_probability:
                    action = np.random.randint(4)

                # Take action and observe next state and reward
                next_state = take_action(state, action, grid_size)
                reward = calculate_reward(next_state, end_states, dangerous_states)

                # Store the state, action, and reward
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)

                # Move to the next state
                state = next_state

            # Update Q-values based on the episode
            G = 0
            for t in range(len(episode_states) - 1, -1, -1):
                G = gamma * G + episode_rewards[t]
                s, a = episode_states[t], episode_actions[t]
                returns[s[0], s[1], a, 0] += G
                returns[s[0], s[1], a, 1] += 1
                Q_values[s[0], s[1], a] = returns[s[0], s[1], a, 0] / returns[s[0], s[1], a, 1]

        # Calculate average utility for the current number of episodes
        utility_values = np.max(Q_values, axis=2)
        average_utilities.append(np.mean(utility_values))

    # Print completion message
    clear_output(wait=True)
    print("Monte Carlo completed!")

    # Display the average utility values
    print("Average Utilities:")
    for i, avg_utility in enumerate(average_utilities):
        print(f"Episodes: {i + 1}, Average Utility: {avg_utility:.2f}")

    # Update the grid display
    update_grid()

    # Display the utility values below the grid
    display_utility()

def update_grid():
    # This is a simple example; you might want to customize it based on your needs
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            q_values_str = [f"{q:.2f}" for q in Q_values[row, col, :]]
            label_text = f"({row}, {col})\nQ-values: {', '.join(q_values_str)}"
            print(label_text)

def display_utility():
    utility_values = np.max(Q_values, axis=2)

    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            label_text = f"({row}, {col})\nUtility: {utility_values[row, col]:.2f}"
            print(label_text)

# Create widgets
slip_label = widgets.Label(value="Slip Probability:")
slip_entry = widgets.FloatText(value=0.1)
epsilon_label = widgets.Label(value="Epsilon:")
epsilon_entry = widgets.FloatText(value=0.1)
episodes_label = widgets.Label(value="Number of Episodes:")
episodes_entry = widgets.IntText(value=100)
run_button = widgets.Button(description="Run Monte Carlo")

# Set the event handler for the button
run_button.on_click(run_monte_carlo)

# Display widgets
display(slip_label, slip_entry, epsilon_label, epsilon_entry, episodes_label, episodes_entry, run_button)
