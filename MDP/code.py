# import library

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# create Environment

max_iter_number = 1000
discount_factor = 0.9
map_size = 3

env = gym.make("FrozenLake-v1", desc=generate_random_map(size=map_size),
               render_mode="human", is_slippery=False)
observation, info = env.reset(seed=42)

# create empty dictionary to store the policy for each state
policy = {}
# create a set to store the terminal states
terminal_states = set()
# set the goal state, which is the last cell of the grid
goal_state = (map_size - 1) * map_size + (map_size - 1)


# Value Iteration Algorithm
def value_iteration():
    # Initialize
    v_values, q_values = {}, {}
    convergenceTrack = [0]

    # Initialize the value of each state to 0 and store the terminal states
    for state in env.P:
        if state == goal_state:
            continue
        v_values[state] = 0
        q_values[state] = {}
        for act in env.P[state]:
            q_values[state][act] = 0
            for probability, nextState, reward, isTerminalState in env.P[state][act]:
                if (reward == 0) and isTerminalState:
                    terminal_states.add(nextState)

    # Set the value of the goal state to 1 and the terminal states to -1
    v_values[goal_state] = 1
    for ts in terminal_states:
        v_values[ts] = -1

    for i in range(10000):
        # Check states in the environment
        for state in env.P:
            if (state not in terminal_states) and (state != goal_state):
                # Check actions in that state
                for act in env.P[state]:
                    s = 0
                    # Calculate every result of the action
                    for probability, nextState, reward, isTerminalState in env.P[state][act]:
                        # Calculate the reward of each actions
                        if (reward == 0) and isTerminalState:
                            reward = -1
                        elif (reward == 1) and isTerminalState:
                            reward = 1
                        else:
                            # Calculate the distance to the goal
                            x, y = nextState // map_size, nextState % map_size
                            dist_to_goal = np.sqrt(
                                np.power(x - (map_size - 1), 2) + np.power(y - (map_size - 1), 2))
                            reward = -0.1 / (1 + np.exp(-dist_to_goal))
                        s += probability * \
                            (reward + (discount_factor * v_values[nextState]))
                    # Update the q_value of state and action
                    q_values[state][act] = s
                # Update the v_value of state and policy
                v_values[state] = max(q_values[state].values())
                # Check convergence
                convergenceTrack.append(
                    np.linalg.norm(list(v_values.values())))
                if (i > 1000) and np.isclose(convergenceTrack[-1], convergenceTrack[-2]):
                    print('Values Converged')
                    return v_values, q_values
    return v_values, q_values


if __name__ == "__main__":

    v_values, q_values = value_iteration()
    for state in env.P:
        if (state not in terminal_states) and (state != goal_state):
            policy[state] = max(q_values[state], key=q_values[state].get)

    n_win = 0
    print(f"{q_values=}")
    print(f"{v_values=}")
    print(f"{policy=}")

    # Save the policy
    with open('policy.txt', 'w', encoding="utf-8") as inp:
        for i in range(map_size):
            for j in range(map_size):
                if ((i * map_size) + j) in terminal_states:
                    inp.write(u'☠\t')
                elif ((i * map_size) + j) == goal_state:
                    inp.write(u'🪙\t')
                elif policy[(i * map_size) + j] == 0:
                    inp.write(u'←\t')
                elif policy[(i * map_size) + j] == 1:
                    inp.write(u'↓\t')
                elif policy[(i * map_size) + j] == 2:
                    inp.write(u'→\t')
                elif policy[(i * map_size) + j] == 3:
                    inp.write(u'↑\t')
            inp.write('\n')

    # Test the policy
    for r in range(max_iter_number):
        action = policy[observation]

        observation, reward, terminated, truncated, info = env.step(action)

        if reward:
            n_win += 1

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

    print(f"Number of wins {n_win}")
    print(f"Win Ratio: {n_win / max_iter_number * 100:.3f}%")
