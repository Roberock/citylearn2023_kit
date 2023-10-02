from stable_baselines3 import PPO
from citylearn.agents.rbc import BasicRBC, Agent
import pickle as pkl
import numpy as np
# agent_name = 'PPO_citylearn2023_comfort_5M'
agent_name = 'PPO_citylearn2023_mixed_reward'
# agent_name = 'PPO_citylearn2023_mixed_reward_530k'
agent_namev2 = 'PPO_citylearn2023_mix_week_hour_feature'

class PPO_agent_v0(Agent):
    """ Can be any subclass of citylearn.agents.base.Agent """
    def __init__(self, env):
        super().__init__(env)
        self.agent_load_dir = './agents/models/'
        self.agent_load_name = agent_name
        self.policy = PPO.load(self.agent_load_dir + self.agent_load_name).policy
        self.exploit = True

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)

    def predict(self, observations):
        """ Just a passthrough, can implement any custom logic as needed """
        return self.policy.predict(observations[0], deterministic=self.exploit)



class PPO_agent_v2(Agent):
    """ Can be any subclass of citylearn.agents.base.Agent """
    def __init__(self, env):
        super().__init__(env)
        self.agent_load_dir = './agents/models/'
        self.agent_load_name = agent_namev2
        self.policy = PPO.load(self.agent_load_dir + self.agent_load_name).policy
        self.exploit = True

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)

    def predict(self, observations):
        """ Just a passthrough, can implement any custom logic as needed """

        hour_week = (observations[0][0]-1)*24+observations[0][1]
        obs = [hour_week] + observations[0]
        return self.policy.predict(obs, deterministic=self.exploit)


class PPO_agent_pickle(Agent):
    """ Can be any subclass of citylearn.agents.base.Agent """
    def __init__(self, env):
        super().__init__(env)
        self.agent_load_dir = './agents/models/'
        self.agent_load_name = 'PPO_model.zip'


        # Load the model using the default protocol (protocol 5)
        with open(self.agent_load_dir + self.agent_load_name, "rb") as file:
                self.policy = pkl.load(file)

        self.exploit = True

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)

    def predict(self, observations):
        """ Just a passthrough, can implement any custom logic as needed """
        n_buildings = len(self.env.buildings_metadata)
        hour_week = (observations[0][0]-1)*24+observations[0][1]
        obs = [hour_week] + observations[0]
        if n_buildings == 3:
            return self.policy.predict(obs, deterministic=self.exploit)
        elif n_buildings == 6:
            obs_3_shared = obs[:19]
            obs_3_public = obs[19:19+11*3]
            obs_3_private = obs[19+11*3:]
            a_pub = self.policy.predict(obs_3_shared+obs_3_public, deterministic=self.exploit)
            a_priv = self.policy.predict(obs_3_shared + obs_3_private, deterministic=self.exploit)
            return (np.hstack([a_pub[0],a_priv[0]]), None)