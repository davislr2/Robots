import numpy as np
import keras
from keras.layers import Input, Dense
from keras.optimizers.legacy import Adam, Adamax, Adadelta, Adagrad
from rgkit import rg

"""
0 <= game.turn < 101
game.turn == 0 --> spawn new robots during this turn
game.turn % 10 == 0 --> span new robots on these turns
game.turn == 99 --> last turn of the game
game.turn == 100 --> determine reward for the last turn based on resulting state
game.game_over --> 

robot contains following fields:
- hp: (health points [0, 50]) <= 0 means it died after last turn
- location: tuple (row, column)
- player_id: 0 or 1, indicating which team it is on, the bot supplied by get_state and get_reward is on your team.
- teammates: number of teammates this turn
- opponents: number of opponents this turn
- previous_teammates: number of teammates last turn
- previous_opponents: number of opponents last turn
- damage_caused: damage caused by this robot last turn
- damage_taken: damage taken by this robot last turn
- kills: number of opponents this robot killed last turn
- birthturn: True/False whether this is the first turn for this bot
- team_deaths: number of team deaths last turn
- opponent_deaths: number of opponent deaths last turn
"""


def spawn_next_turn(game):
    return game.turn % 10 == 0

def last_turn(game):
    return game.turn == 99


def game_over(game):
    return game.turn == 100


def died(robot):
    return robot.hp <= 0


def get_state(game, robot):
    """
    Determine the "state" of the robot in the game. Robots in the same state should act in the same way
    @param game: the game
    @param robot: the robot
    @return: the state of the robot in the game (numpy vector)

    state[0] = True if there is an enemy in the neighborhood
    state[1] = True if suicide is a good option.
    state[2] = True if there's 2 or more enemies in the neighborhood. 
    """
    neighborhood = rg.get_neighborhood(game, robot, within=1, metric='euclidean')
    state = []
    state0 = any([x is not None for x in neighborhood['enemies']])
    state1 = all([x is not None for x in neighborhood['enemies']])
    state2 = sum(1 for x in neighborhood['enemies'] if x is not None) >= 2
    state3 = sum(1 for x in neighborhood['enemies'] if x is not None)
    state.append(state0)
    state.append(state1)
    state.append(state2)
    state.append(state3)
    state = np.array(state, dtype=np.int32)
    return state


def build_model(input_shape, learning_rate=0.001):
    """
    Build and compile a model that takes the state as input and estimates q-values for each action index.
    @param input_shape: the size of the state vector
    @param learning_rate: learning rate
    @return: the compiled model
    """
    model = keras.Sequential([
        Input(shape=input_shape),
        Dense(units=32, activation='relu'),
        Dense(units=16, activation='relu'),
        Dense(units=8, activation='relu'),
        Dense(units=10, activation='linear'),
    ])
    model.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='mean_squared_error')
    return model


def get_reward(game, bot):
    """
    Use the game and the robot to determine the robot's reward for taking its action last turn.
    @param game: the game
    @param bot: the robot
    @return: the robot's reward
    """
    reward = 0
    num_teammates = bot.teammates
    num_opponents = bot.opponents
    num_previous_teammates = bot.previous_teammates
    num_previous_opponents = bot.previous_opponents
    damage_caused = bot.damage_caused
    damage_taken = bot.damage_taken
    kills = bot.kills
    birthturn = bot.birthturn
    team_deaths = bot.team_deaths
    opponent_deaths = bot.opponent_deaths
    state = get_state(game, bot)
    enemiesInNeighborhood = state[0]
    suicideGood = state[1]
    guardGood = state[2]
    noEnemiesInNeighborhood = state[3]

    

    # Check if it's the last turn of the game
    if game.turn == 99:
        # Add rewards or penalties based on the resulting state
        if bot.hp > 0:
            reward += 75  # A positive reward for surviving until the end
        else:
            reward -= 25  # A penalty for dying before the end

    # Check if it's the end of the game
    elif game.turn == 100:
        # Determine reward based on the resulting state
        if num_teammates + 1 > num_opponents:
            reward += 250  # A higher positive reward for winning the game
        elif num_teammates + 1 == num_opponents:
            reward += 75  # A lower positive reward for tying the game
        else:
            reward -= 150  # A higher penalty for losing the game

    # Otherwise, during the game turns
    else:

        # Reward for attacking when area is clear.
        if (enemiesInNeighborhood and damage_caused > 5 and damage_taken == 0):
            reward += 20
        
        # Reward for comitting suicide in a good situation.
        if (suicideGood and damage_caused > 0 and damage_taken > 0):
            reward += 10
        
        # Reward for guarding when there's 2 or more enemies.
        if (guardGood and damage_caused == 0 and (damage_taken < noEnemiesInNeighborhood * 8)):
            reward += 10

        # Reward for dealing damage
        reward +=   damage_caused

        # Penalty for taking damage
        reward -=   damage_taken

        # Reward for killing opponents
        reward += kills * 15

        # Encourage strategic movements and actions
        if bot.hp > 0:
            reward += 10  # Small positive reward for staying alive

        # Encourage teammates staying alive.
        if num_teammates >  num_previous_teammates:
            reward += 5
        
        # Penalty for collisions. 
        
        if   damage_caused >   damage_taken:
            reward += 5

        # Encourage attacking.
        if   damage_caused >= 8 and   damage_caused <= 10:
            reward +=   damage_caused * 2

        # Encourage guarding. 
        if   damage_taken <= 5 and   damage_caused == 0:
            reward += 3
        
        # Penalty for team deaths
        reward -= team_deaths * 20

    return reward
