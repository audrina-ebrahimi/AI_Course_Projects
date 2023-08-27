# import library

import gym
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

#create Enviroment
env = gym.make("Taxi-v3")
# reset enviroment
observation = env.reset()

# set hyperparameters
MAX_ITER_NUMBER = 10000
EPSILON = 0.9
LEARNING_RATE = 0.8
DISCOUNT_FACTOR = 0.9

# colorful cordinates
MAP_COLORFUL_PLACES = {
    0: np.array([0, 4]), #red
    1: np.array([4, 4]), #green
    2: np.array([0, 0]), #yellow
    3: np.array([3, 0])  #blue
}

convergenceTrack = [0]

# initialize q_table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

def epsilon_greedy_policy(q_table, observation):
    if np.random.uniform(0, 1) < EPSILON:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[observation])
    return action

def Q_Learning_Algorithm(observation, q_table):
    # train the agent for max_iter_number number of episodes
    for i in range(0, MAX_ITER_NUMBER):
        # reset the environment
        done = False

        # explore the environment until the agent reaches the goal
        while not done:
            # choose the action using epsilon greedy policy  
            action = epsilon_greedy_policy(q_table, observation)

            next_observation, reward, done, info = env.step(action)

            # get the current state of the agent, and the position of the passenger and the goal
            taxi_row, taxi_col, pas_pos, goal_pos = list(env.decode(next_observation))

            # if the agent reaches the goal, then reward = 20, else if drop and pickup the passenger, then reward = -10
            if (reward == 20) and done:
                reward = 20
            elif (reward == -10) and not done:
                reward = -10
            # if the agent picks up the passenger
            elif (pas_pos == 4) and reward == -1:
                # Calculate the distance to the goal
                dist_to_goal = np.linalg.norm(np.array([taxi_row, taxi_col]) - MAP_COLORFUL_PLACES[goal_pos])
                reward = -0.1 / (1 + np.exp(-dist_to_goal))
            # if the agent have not pick up the passenger
            elif (pas_pos != 4) and reward == -1:
                # Calculate the distance to the passenger
                dist_to_goal = np.linalg.norm(np.array([taxi_row, taxi_col]) - MAP_COLORFUL_PLACES[pas_pos])
                reward = -0.1 / (1 + np.exp(-dist_to_goal))

            # choose the next action
            next_action = np.argmax(q_table[next_observation])
            # update the q_table
            q_table[observation][action] = q_table[observation][action] + LEARNING_RATE * \
            (reward + DISCOUNT_FACTOR * q_table[next_observation][next_action] - q_table[observation][action])
            # update the observation
            observation = next_observation

        # if the agent reaches the goal, then reset the environment
        if done:
            observation = env.reset()

        # check convergence
        convergenceTrack.append(np.linalg.norm(q_table.flatten().tolist()))
        if (i > 1000) and np.isclose(convergenceTrack[-1], convergenceTrack[-2]):
            print('Values Converged')
            return

        if (i+1) % 100 == 0:
                print(i+1)

    print("Training Completed")
    sleep(2)


def save_q_table(q_table):
    map_size = 6
    with open('q_table_taxi.txt', 'w', encoding="utf-8") as inp:
        for state in range(map_size**2):
            x, y = state // map_size, state % map_size
            if np.array_equal(MAP_COLORFUL_PLACES[0], np.array([x, y])):
                inp.write(u'🔴\t')
            elif np.array_equal(MAP_COLORFUL_PLACES[1], np.array([x, y])):
                inp.write(u'🟢\t')
            elif np.array_equal(MAP_COLORFUL_PLACES[2], np.array([x, y])):
                inp.write(u'🟡\t')
            elif np.array_equal(MAP_COLORFUL_PLACES[3], np.array([x, y])):
                inp.write(u'🔵\t')
            else:
                if np.all(q_table[state] == 0):
                    inp.write(u'⬜\t')
                else:
                    argm = np.argmax(q_table[state])
                    if argm == 0:
                        inp.write(u'↓\t')
                    elif argm == 1:
                        inp.write(u'↑\t')
                    elif argm == 2:
                        inp.write(u'→\t')
                    elif argm == 3:
                        inp.write(u'←\t')
                    elif argm == 4:
                        inp.write(u'🧍\t')
                    elif argm == 5:
                        inp.write(u'🏠\t')
            if (state + 1) % map_size == 0:
                inp.write('\n')

def plot_convergence(convergence):
    plt.plot(convergence)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Q-Table Convergence")
    plt.title("Convergence of Q-Learning Algorithm for Taxi-v3")
    plt.savefig("convergence_taxi.png")
  

if __name__ == "__main__":

    # train the agent
    Q_Learning_Algorithm(observation, q_table)

    save_q_table(q_table)
    plot_convergence(convergenceTrack)

    # reset the environment
    observation = env.reset()

    # test the agent for one time
    done = False

    while not done:
        action = np.argmax(q_table[observation])
        observation, reward, done, info = env.step(action)

        print(env.render())
        sleep(1)
    

    env.close()
    