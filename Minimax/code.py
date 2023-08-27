# import Library
from pettingzoo.classic import connect_four_v3
import numpy as np
import pygame
from os import mkdir
from shutil import rmtree
from time import sleep

# Create Environment
env = connect_four_v3.env(render_mode="human")

env.reset()
# Define Constants
MAX_DEPTH = 6
ROW_COUNT = 6
COLUMN_COUNT = 7
AI_PLAYER = 0
HUMAN_PLAYER = 1
EPSILON = 1


# Heuristics for getting scores from game map
def heuristic(board, agent):
    # Define agent and opponent
    piece = agent + 1
    piece_opp = int(not (piece - 1)) + 1

    # Increase score if the center column contains the agent's piece
    h = 0
    for r in range(ROW_COUNT):
        if board[r][COLUMN_COUNT // 2] == piece:
            h += 3 * EPSILON

    # Check consecutiveness of piece in rows
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            # Increase score for four consecutive pieces of the agent
            if (
                (board[r][c] == piece)
                and (board[r][c + 1] == piece)
                and (board[r][c + 2] == piece)
                and (board[r][c + 3] == piece)
            ):
                h += 100
            # Increase score for three consecutive pieces with an empty space at the end
            elif (
                (board[r][c] == piece)
                and (board[r][c + 1] == piece)
                and (board[r][c + 2] == piece)
                and (board[r][c + 3] == 0)
            ):
                h += 5
            # Increase score for two consecutive pieces with two empty spaces at the end
            elif (
                (board[r][c] == piece)
                and (board[r][c + 1] == piece)
                and (board[r][c + 2] == 0)
                and (board[r][c + 3] == 0)
            ):
                h += 2
            # Decrease score for three consecutive pieces of the opponent with an empty space at the end
            if (
                (board[r][c] == piece_opp)
                and (board[r][c + 1] == piece_opp)
                and (board[r][c + 2] == piece_opp)
                and (board[r][c + 3] == 0)
            ):
                h -= 4

    # Check consecutiveness of piece in columns
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            # Increase score for four consecutive pieces of the agent in a vertical line
            if (
                (board[r][c] == piece)
                and (board[r + 1][c] == piece)
                and (board[r + 2][c] == piece)
                and (board[r + 3][c] == piece)
            ):
                h += 100
            # Increase score for three consecutive pieces in a vertical line with an empty space at the end
            elif (
                (board[r][c] == piece)
                and (board[r + 1][c] == piece)
                and (board[r + 2][c] == piece)
                and (board[r + 3][c] == 0)
            ):
                h += 5
            # Increase score for two consecutive pieces in a vertical line with two empty spaces at the end
            elif (
                (board[r][c] == piece)
                and (board[r + 1][c] == piece)
                and (board[r + 2][c] == 0)
                and (board[r + 3][c] == 0)
            ):
                h += 2
            # Decrease score for three consecutive pieces of the opponent in a vertical line with an empty space at the end
            if (
                (board[r][c] == piece_opp)
                and (board[r + 1][c] == piece_opp)
                and (board[r + 2][c] == piece_opp)
                and (board[r + 3][c] == 0)
            ):
                h -= 4

    # Check consecutiveness of piece in diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            # Increase score for four consecutive pieces of the agent in a diagonal line
            if (
                (board[r][c] == piece)
                and (board[r + 1][c + 1] == piece)
                and (board[r + 2][c + 2] == piece)
                and (board[r + 3][c + 3] == piece)
            ):
                h += 100
            # Increase score for three consecutive pieces in a diagonal line with an empty space at the end
            elif (
                (board[r][c] == piece)
                and (board[r + 1][c + 1] == piece)
                and (board[r + 2][c + 2] == piece)
                and (board[r + 3][c + 3] == 0)
            ):
                h += 5
            # Increase score for three consecutive pieces in a diagonal line with
            elif (
                (board[r][c] == piece)
                and (board[r + 1][c + 1] == piece)
                and (board[r + 2][c + 2] == 0)
                and (board[r + 3][c + 3] == 0)
            ):
                h += 2
            # Decrease score for three consecutive pieces of the opponent in a diagonal line with an empty space at the end
            if (
                (board[r][c] == piece_opp)
                and (board[r + 1][c + 1] == piece_opp)
                and (board[r + 2][c + 2] == piece_opp)
                and (board[r + 3][c + 3] == 0)
            ):
                h -= 4

    # Check consecutiveness of piece in diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            # Increase score for four consecutive pieces of the agent in a diagonal line
            if (
                (board[r][c] == piece)
                and (board[r - 1][c + 1] == piece)
                and (board[r - 2][c + 2] == piece)
                and (board[r - 3][c + 3] == piece)
            ):
                h += 100
            # Increase score for three consecutive pieces of the agent in a diagonal line
            elif (
                (board[r][c] == piece)
                and (board[r - 1][c + 1] == piece)
                and (board[r - 2][c + 2] == piece)
                and (board[r - 3][c + 3] == 0)
            ):
                h += 5
            # Increase score for two consecutive pieces of the agent in a diagonal line
            elif (
                (board[r][c] == piece)
                and (board[r - 1][c + 1] == piece)
                and (board[r - 2][c + 2] == 0)
                and (board[r - 3][c + 3] == 0)
            ):
                h += 2
            # Decrease score for three consecutive pieces of the opponent in a diagonal line
            if (
                (board[r][c] == piece_opp)
                and (board[r - 1][c + 1] == piece_opp)
                and (board[r - 2][c + 2] == piece_opp)
                and (board[r - 3][c + 3] == 0)
            ):
                h -= 4

    return h


def winning_condition(board, agent):
    piece = agent + 1

    # Return True if there are four consecutive pieces of the agent in a row
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if (
                (board[r][c] == piece)
                and (board[r][c + 1] == piece)
                and (board[r][c + 2] == piece)
                and (board[r][c + 3] == piece)
            ):
                return True

    # Return True if there are four consecutive pieces of the agent in a column
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if (
                (board[r][c] == piece)
                and (board[r + 1][c] == piece)
                and (board[r + 2][c] == piece)
                and (board[r + 3][c] == piece)
            ):
                return True

    # Return True if there are four consecutive pieces of the agent in a diagonal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if (
                (board[r][c] == piece)
                and (board[r + 1][c + 1] == piece)
                and (board[r + 2][c + 2] == piece)
                and (board[r + 3][c + 3] == piece)
            ):
                return True

    # Return True if there are four consecutive pieces of the agent in a diagonal
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if (
                (board[r][c] == piece)
                and (board[r - 1][c + 1] == piece)
                and (board[r - 2][c + 2] == piece)
                and (board[r - 3][c + 3] == piece)
            ):
                return True

    # If no winning condition is found, return False
    return False


def generate_map(obs):
    observation = obs.copy()

    # Create a map where AI player's positions are marked as 1
    map_agent = np.where(observation[:, :, AI_PLAYER], 1, 0)

    # Create a map where human player's positions are marked as 2
    map_opponent = np.where(observation[:, :, HUMAN_PLAYER], 2, 0)

    # # Create a map where AI player's positions are marked as 1, human player's positions as 2, and empty positions as 0
    map_total = np.where(map_agent, 1, np.where(map_opponent, 2, 0))

    return map_total


def get_valid_columns(board):
    valid_columns = []

    # Check if the top row of the board is empty
    for column in range(COLUMN_COUNT):
        if board[0][column] == 0:
            valid_columns.append(column)
    return valid_columns


def is_terminal_node(board):
    # Return True if the board is full or if there is a winner
    return (
        (not len(get_valid_columns(board)))
        or (winning_condition(board, AI_PLAYER))
        or (winning_condition(board, HUMAN_PLAYER))
    )


def get_next_open_row(board, column):
    # Return the next open row in the column
    for r in range(ROW_COUNT - 1, -1, -1):
        if board[r][column] == 0:
            return r


def MiniMax(game_map, depth, alpha, beta, player):
    # Check if the current node is a terminal node or if the maximum depth has been reached
    is_terminal = is_terminal_node(game_map)
    if depth == 0 or is_terminal:
        if is_terminal:
            # If the game is over, return the corresponding utility value
            if winning_condition(game_map, AI_PLAYER):
                return (None, 1000000000000000)  # AI wins
            elif winning_condition(game_map, HUMAN_PLAYER):
                return (None, -1000000000000000)
            else:
                return (None, 0)  # It's a draw
        else:
            # If the maximum depth is reached, return the heuristic value of the current node
            return (None, heuristic(game_map, AI_PLAYER))

    # If the current node is not a terminal node or if the maximum depth has not been reached
    if player == AI_PLAYER:
        # If the current player is the AI player, maximize the value of the current node
        opponent = HUMAN_PLAYER
        value = -np.inf
        valid_columns = get_valid_columns(game_map)
        best_move = np.random.choice(valid_columns)

        # Iterate through all possible moves
        for column in valid_columns:
            row = get_next_open_row(game_map, column)

            # Create a copy of the current board and simulate the move
            temp_map = game_map.copy()
            temp_map[row][column] = AI_PLAYER + 1

            # Recursively call the MiniMax function with the new board and the opponent as the current player
            new_score = MiniMax(temp_map, depth - 1, alpha, beta, opponent)[1]

            # If the new score is greater than the current value, update the current value and the best move
            if new_score > value:
                value = new_score
                best_move = column
            # Update alpha
            alpha = max(alpha, value)
            # If alpha is greater than or equal to beta, break out of the loop
            if alpha >= beta:
                break
        # Return the best move and the value of the current node
        return (best_move, value)

    elif player == HUMAN_PLAYER:
        # If the current player is the human player, minimize the value of the current node
        opponent = AI_PLAYER
        value = np.inf
        valid_columns = get_valid_columns(game_map)
        best_move = np.random.choice(valid_columns)

        # Iterate through all possible moves
        for column in valid_columns:
            row = get_next_open_row(game_map, column)

            # Create a copy of the current board and simulate the move
            temp_map = game_map.copy()
            temp_map[row][column] = HUMAN_PLAYER + 1

            # Recursively call the MiniMax function with the new board and the opponent as the current player
            new_score = MiniMax(temp_map, depth - 1, alpha, beta, opponent)[1]

            # If the new score is less than the current value, update the current value and the best move
            if new_score < value:
                value = new_score
                best_move = column

            # Update beta
            beta = min(beta, value)

            # If alpha is greater than or equal to beta, break out of the loop
            if alpha >= beta:
                break

        # Return the best move and the value of the current node
        return (best_move, value)


if __name__ == "__main__":
    try:
        mkdir("steps_pic")
    except Exception:
        pass

    i = 0

    # Iterate over each agent in the environment
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        player = int(agent[-1])

        if termination or truncation:
            action = None

        else:
            if player == AI_PLAYER:
                # Generate a map representation of the current observation
                generated_map = generate_map(observation["observation"])
                # Apply the MiniMax algorithm to choose the best action for the AI player
                action, score = MiniMax(
                    generated_map, MAX_DEPTH, -np.inf, np.inf, AI_PLAYER
                )
                # Adjust the exploration rate (EPSILON) after each AI move
                EPSILON *= 0.9
                print(generated_map)
                print(f"MiniMax_Score: {score}")
                print("------------------------------------")

            elif player == HUMAN_PLAYER:
                # Prompt the human player to enter their action
                action = int(input("Enter Your Action: ")) - 1
                while observation["action_mask"][action] == 0:
                    print("Your action is ilegal! Try Again!")
                    action = int(input("Enter Your Action: ")) - 1

        # Take a step in the environment with the chosen action
        env.step(action)
        if action is not None:
            # Save the current screen as an image in the 'steps_pic' directory
            pygame.image.save(env.unwrapped.screen, f"steps_pic/{i + 1}.jpeg")
            i += 1

    sleep(5)
    env.close()
