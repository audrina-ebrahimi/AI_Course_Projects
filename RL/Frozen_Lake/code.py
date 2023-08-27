# import library

import gymnasium as gym
import numpy as np
from time import sleep
from tqdm import tqdm
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt


# generate random map
map_size = 64
map_shape = generate_random_map(size=map_size)

# create Enviroment
env = gym.make("FrozenLake-v1", desc=map_shape, is_slippery=False)

# reset enviroment
observation, info = env.reset(seed=42)

# initialize q_table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# set hyperparameters
max_iter_number = 100000
EPSILON = 0.1
DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.8
MAX_STEP = 99

convergenceTrack = [0]


def epsilon_greedy_policy(q_table, observation):
    if np.random.uniform(0, 1) < EPSILON:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[observation])
    return action


def Q_Learning_Algorithm(observation, q_table):
    # train the agent for max_iter_number number of episodes
    for i in tqdm(range(0, max_iter_number)):
        # reset the environment
        observation, info = env.reset()
        terminated, truncated = False, False

        # for each episode, run the algorithm for MAX_STEP number of steps
        for __ in range(MAX_STEP):
            # choose the action using epsilon greedy policy
            action = epsilon_greedy_policy(q_table, observation)

            next_observation, reward, terminated, truncated, info = env.step(action)

            # if the agent reaches the goal, then reward = 1, else if it falls to a hole, reward = -1
            if (reward == 0) and terminated:
                reward = -1
            elif (reward == 1) and terminated:
                reward = 1
            else:
                # Calculate the distance to the goal for agent and reward it based on the distance
                x, y = next_observation // map_size, next_observation % map_size
                dist_to_goal = np.sqrt(
                    np.power(x - (map_size - 1), 2) + np.power(y - (map_size - 1), 2)
                )
                reward = -0.1 / (1 + np.exp(-dist_to_goal))

            # choose the next action
            next_action = np.argmax(q_table[next_observation])

            # update the q_table
            q_table[observation][action] = q_table[observation][
                action
            ] + LEARNING_RATE * (
                reward
                + DISCOUNT_FACTOR * q_table[next_observation][next_action]
                - q_table[observation][action]
            )

            # update the observation
            observation = next_observation

            # if the agent reaches the goal or falls into a hole, then reset the environment
            if terminated or truncated:
                observation, info = env.reset()
                break

        # convergenceTrack.append(np.linalg.norm(q_table.flatten().tolist()))
        # if (i >= 1000) and np.isclose(convergenceTrack[-1], convergenceTrack[-2]):
        #     print('\nValues Converged')
        #     return
    print("Training Completed")
    sleep(2)


def plot_convergence(convergence):
    plt.plot(convergence)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Q-Table Convergence")
    plt.title("Convergence of Q-Learning Algorithm for Frozen Lake")
    plt.savefig("convergence_lakes.png")


def save_q_table(q_table):
    terminal_states = set()
    goal_state = (map_size - 1) * map_size + (map_size - 1)

    for state in env.P:
        if state == goal_state:
            continue
        for act in env.P[state]:
            for probability, nextState, reward, isTerminalState in env.P[state][act]:
                if (reward == 0) and isTerminalState:
                    terminal_states.add(nextState)

    with open("q_table_frozen.txt", "w", encoding="utf-8") as inp:
        for state in range(map_size**2):
            if state in terminal_states:
                inp.write("☠\t")
            elif state == goal_state:
                inp.write("🪙\t")

            else:
                if np.all(q_table[state] == 0):
                    inp.write("⬜\t")
                else:
                    argm = np.argmax(q_table[state])
                    if argm == 0:
                        inp.write("←\t")
                    elif argm == 1:
                        inp.write("↓\t")
                    elif argm == 2:
                        inp.write("→\t")
                    elif argm == 3:
                        inp.write("↑\t")
            if (state + 1) % map_size == 0:
                inp.write("\n")


if __name__ == "__main__":
    # train the agent with no graphical output
    Q_Learning_Algorithm(observation, q_table)

    save_q_table(q_table)

    env.close()

    # create Enviroment
    env = gym.make(
        "FrozenLake-v1", desc=map_shape, render_mode="human", is_slippery=False
    )

    # reset enviroment
    observation, info = env.reset()

    # test the agent for max_iter_number number of episodes
    for i in range(1):
        terminated, truncated = False, False

        for _ in range(MAX_STEP):
            # select the action
            action = np.argmax(q_table[observation])

            next_observation, reward, terminated, truncated, info = env.step(action)

            env.render()

            # check if the agent reaches the goal or falls into a hole
            if terminated or truncated:
                observation, info = env.reset()
                break

            observation = next_observation

    env.close()
    plot_convergence(convergenceTrack)
