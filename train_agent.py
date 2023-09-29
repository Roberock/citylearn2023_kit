import os
from citylearn.citylearn import CityLearnEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch as th
from rewards.combined_demand_reward import MixReward
from stable_baselines3.common.callbacks import EvalCallback
from multiprocessing import cpu_count
import numpy as np
from citylearn.reward_function import ComfortReward
from stable_baselines3.common.utils import set_random_seed
from local_evaluation import WrapperEnv, create_citylearn_env
import gymnasium
from gymnasium import spaces

class Config:
    data_dir = './data/'
    SCHEMA = os.path.join(data_dir, 'schemas/warm_up/schema.json')
    num_episodes = 1

"""class reward_function(ComfortReward): 
    def __init__(self, env_metadata):
        super().__init__(env_metadata)
    def calculate(self, observations):
        return super().calculate(observations)
"""

reward_function = MixReward

def make_env(config, seed: int = 0):
    def _init():
        env_rank = CityLearnWrapper(conf=config, reward_fun=reward_function)
        env_rank.reset()
        return env_rank
    set_random_seed(seed)
    return _init



class CityLearnWrapper(gymnasium.Env):
    def __init__(self, conf, reward_fun):
        super(CityLearnWrapper, self).__init__()

        self.env = CityLearnEnv(conf.SCHEMA, reward_function=reward_fun)
        obs_env = self.env.observation_space[0]
        act_env = self.env.action_space[0]
        # Define observation space and action space based on your CityLearnEnv
        self.observation_space = spaces.Box(low=obs_env.low, high=obs_env.high,
                                            shape=obs_env.shape, dtype=np.float32)

        self.action_space = spaces.Box(low=act_env.low, high=act_env.high,
                                       shape=act_env.shape,  dtype=np.float32)



    def step(self, action):
        # Take a step in the CityLearnEnv with the given action
        obs, reward, done, info = self.env.step([action])
        # todo:
        #  1) add hour_of_week_feature (day-1)*24 + hour
        #  2) remove constant observations (screening, feature importance, etc)
        #  3) remove 'identical' private observations for two of the buildings... [Month	Hour	Day Type]
        #  4) Normalize reward rew(action)/rew(action=0)? ....
        #  run episode with an action_vector = [0,0,0,0,0,0,0,0,0]...
        #  collect rewards r_t(a=zeros)....load within the wrapper and normalize

        truncate = False
        return obs[0], reward[0], done, truncate, info

    def render(self, mode='human'):
        # Implement rendering if applicable
        pass

    def reset(self, seed=None, options=None):
        # Reset the CityLearnEnv and return the initial observation
        obs_list = self.env.reset()
        info = self.env.get_info()
        return obs_list[0], info

    def close(self):
        # Implement any necessary cleanup
        pass


if __name__ == '__main__':
    # load data and config
    config = Config()

    # prepare directories
    model_out_dir = './agents/models/'
    tens_br_dir = "./tensorboard_log"
    model_name = 'PPO_citylearn2023_mixed_reward'

    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    if not os.path.exists(tens_br_dir):
        os.makedirs(tens_br_dir)

    # define policy network
    total_time_steps = 20_000
    policy_kwargs = dict(activation_fn=th.nn.LeakyReLU,
                         net_arch=dict(pi=[250, 250],
                                       vf=[250, 250]))

    env = CityLearnWrapper(conf=config, reward_fun=reward_function)
    env.reset()

    try:  # if it exist, load te RL model
        print(' --- Loading saved model, ' + model_name)
        model = PPO.load(model_out_dir + model_name, env)
    except:
        print(' --- Creating model -------' + model_name)
        model = PPO("MlpPolicy", env,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=tens_br_dir,
                    n_steps=2_000, batch_size=200,  verbose=1, learning_rate=0.0005)
    print(' --- START the training')
    model.learn(total_timesteps=100_0000,
                tb_log_name=model_name) #, callback=eval_callback
    print(' --- SAVING MODEL --')
    model.save(model_out_dir + model_name)
    print(' --- END the training')

    # prepare vectorized environment
    # num_cpu = cpu_count() - 3
    num_cpu = 4  # Number of cpu to be    ....
    vec_env_LIC_Gym = SubprocVecEnv([make_env(config, i) for i in range(num_cpu)], start_method='spawn')  # Create the vectorized

    try:  # if it exist, load te RL model
        print(' --- Loading saved model, ' + model_name)
        print('CPU used:', num_cpu)
        model = PPO.load(model_out_dir + model_name, vec_env_LIC_Gym)
    except:
        print(' --- Creating model -------')
        print('CPU used:', num_cpu)
        # if it exist, load te RL model
        model = PPO("MlpPolicy", vec_env_LIC_Gym,
                    n_steps=10_000, batch_size=500,
                    policy_kwargs=policy_kwargs,
                    verbose=1, learning_rate=0.0005,
                    tensorboard_log=tens_br_dir)

    model.learn(total_timesteps=total_time_steps,
                tb_log_name=model_name) #, callback=eval_callback

    model.save(model_out_dir + model_name)

    print(' --- END of training')



