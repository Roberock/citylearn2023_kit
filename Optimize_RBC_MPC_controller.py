import numpy as np
import time
import os
from citylearn.citylearn import CityLearnEnv
import pandas as pd
import matplotlib.pyplot as plt
from rewards.user_reward import SubmissionReward
from local_evaluation import create_citylearn_env
from scipy.optimize import minimize, differential_evolution


# ANSI escape code for colored text
blue_color_code = "\033[94m"
green_color_code = "\033[92m"
# Reset ANSI escape code to default color
reset_color_code = "\033[0m"

if __name__ == '__main__':
    class Config:
        data_dir = './data/'
        SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')
        num_episodes = 1


    config = Config()

    env, wrapper_env = create_citylearn_env(config, SubmissionReward)



    # Define the objective function
    n_h = 168  # hours in a week
    n_act = 3  # actions per buildings
    n_obs = 1  # observations per buildings
    n_buildings = 3
    def objective_function(x):
        total_negative_episodic_reward = 0
        env.reset()
        actions = x.reshape((n_h, n_act))
        done = False
        while not done:
            hour_of_the_week = 24 * (env.observations[0][0] - 1) + env.observations[0][1]
            # take action according to the hour of the week
            action_hour_week = actions[hour_of_the_week-1, :]
            # Perform the action and collect the observation, reward, and done flag
            actions_all_buildings = [np.hstack([action_hour_week for b in range(n_buildings)],dtype='float32')]
            _, rew, done, _ = env.step(actions_all_buildings)
            total_negative_episodic_reward -= rew[0]

        print(f"{green_color_code}{'sum episode reward '}{reset_color_code}, {-total_negative_episodic_reward}")
        return total_negative_episodic_reward


    # Assuming you have defined lower_bounds_table and upper_bounds_table as 2D arrays
    lower_bounds_table = np.array([env.action_space[0].low[-3:] for _ in range(n_h)])
    upper_bounds_table = np.array([env.action_space[0].high[-3:] for _ in range(n_h)])
    bounds = [(lower_bounds_table[hour, i], upper_bounds_table[hour, i]) for hour in range(n_h) for i in range(n_act)]

    # from pyomo.environ import *


    # Define the optimization problem
    """result = minimize(
        objective_function,  # Minimize the negative of the objective function (to maximize)
        x0=np.zeros((n_h, 9)).flatten(),
        bounds=bounds)"""

    result = differential_evolution(objective_function, bounds,  x0=np.zeros((n_h, n_act)).flatten())

    # Extract the optimal actions
    optimal_actions = result.x




    # Initialize empty lists to store data
    observations, actions, rewards= [], [], []

    for e in range(config.num_episodes):
        env.reset()
        done = False
        print(f"{green_color_code}{'episode'}{reset_color_code}, {e}")
        while not done:
            hour_of_the_week = 24 * (env.observations[0][0] - 1) + env.observations[0][1]
            # Generate a random action (you may replace this with your actual action selection logic)
            action_rnd = env.action_space[0].sample()

            # Perform the action and collect the observation, reward, and done flag
            obs, rew, done, _ = env.step([action_rnd])


            # Append data to the respective lists
            observations.append(obs[0])
            actions.append(action_rnd)
            rewards.append(rew[0])

    # Create a DataFrame from the lists
    data = {'Observations': observations, 'Actions': actions}
    df = pd.DataFrame(data)
    df_rew = pd.DataFrame({'Rewards': rewards})

    # define a function to split the list into columns
    def split_obs(row):
        return pd.Series(row['Observations'])


    def split_act(row):
        return pd.Series(row['Actions'])

    # apply the function to the DataFrame
    obs_df = df.apply(split_obs, axis=1).rename(columns=lambda x: env.observation_names[0][x])
    act_df = df.apply(split_act, axis=1).rename(columns=lambda x: f"action{x + 1}")
    df_rew = pd.DataFrame({'Rewards': rewards})
    # Concatenate the DataFrames horizontally (column-wise)
    concatenated_df = pd.concat([obs_df, act_df, df_rew], axis=1)
    # Optionally, you can save the DataFrame to a CSV file
    # concatenated_df.to_csv('episodies_data.csv', index=False)


    # Define a function to perform Min-Max scaling
    def min_max_scaling(column):
        min_val = column.min()
        max_val = column.max()
        scaled_column = -1 + (2 * (column - min_val) / (max_val - min_val))
        return scaled_column


    # Apply Min-Max scaling to each column in concatenated_df
    normalized_df = concatenated_df.apply(min_max_scaling)

    # Create the plot
    """fig, ax = plt.subplots()
    for i, row in normalized_df.iterrows():
        if row['Rewards'] > 0.8:
            ax.plot(range(len(normalized_df.columns)), row, label=f'Data Point {i + 1}', c='r', alpha=0.5)
        else:
            ax.plot(range(len(normalized_df.columns)), row, label=f'Data Point {i + 1}',c='k', alpha=0.1)

    plt.show()"""

