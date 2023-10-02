from stable_baselines3 import PPO
from citylearn.agents.rbc import BasicRBC, Agent
import pickle as pkl
import numpy as np
from typing import Any, Mapping, List, Union

# agent_name = 'PPO_citylearn2023_comfort_5M'
agent_name = 'PPO_citylearn2023_mixed_reward'
# agent_name = 'PPO_citylearn2023_mixed_reward_530k'
agent_namev2 = 'PPO_citylearn2023_mix_week_hour_feature'



shared_observations =    ['day_type', 'hour',
                          'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_6h', 'outdoor_dry_bulb_temperature_predicted_12h',  'outdoor_dry_bulb_temperature_predicted_24h',
                          'diffuse_solar_irradiance', 'diffuse_solar_irradiance_predicted_6h', 'diffuse_solar_irradiance_predicted_12h', 'diffuse_solar_irradiance_predicted_24h',
                          'direct_solar_irradiance', 'direct_solar_irradiance_predicted_6h', 'direct_solar_irradiance_predicted_12h', 'direct_solar_irradiance_predicted_24h',
                          'carbon_intensity',
                          'electricity_pricing', 'electricity_pricing_predicted_6h',  'electricity_pricing_predicted_12h', 'electricity_pricing_predicted_24h']


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
        self.exploit = True
        self.shared_observations = shared_observations
        self.shared_observation_index = [self.observation_names[0].index(s) for s in shared_observations]

        # Load the model using the default protocol (protocol 5)
        with open(self.agent_load_dir + self.agent_load_name, "rb") as file:
            self.policy = pkl.load(file)



    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)

    def predict(self, observations: List[List[float]], deterministic=True) -> List[List[float]]:
        """ Just a passthrough, can implement any custom logic as needed """
        hour_week = [(observations[0][0]-1)*24+observations[0][1]]
        n_buildings = len(self.env.buildings_metadata)
        if n_buildings == 3:
            obs = hour_week + observations[0]
            actions = self.policy.predict(obs, deterministic=self.exploit)[0].tolist()
            return [actions]
        else:
            obs_3_shared = [observations[0][i] for i in self.shared_observation_index]
            obs_3_shared_p1 = obs_3_shared[:14]
            obs_3_shared_p2 = obs_3_shared[14:]
            obs_pub_part_1 = [observations[0][i] for i in [15, 16, 17, 18, 19, 20]]  # fix this cuckoo's
            obs_pub_part_2 = observations[0][25:52]
            obs_pub_priv_1 = [observations[0][i] for i in [53, 54, 55, 56, 57, 58]]  # fix this cuckoo's
            obs_pub_priv_2 = observations[0][58:]
            obs_public = hour_week + obs_3_shared_p1 + obs_pub_part_1 + obs_3_shared_p2 + obs_pub_part_2
            obs_priv = hour_week + obs_3_shared_p1 + obs_pub_priv_1 + obs_3_shared_p2 + obs_pub_priv_2
            a_pub = self.policy.predict(obs_public, deterministic=self.exploit)
            a_priv = self.policy.predict(obs_priv, deterministic=self.exploit)
            actions = [np.hstack([a_pub[0], a_priv[0]]).tolist()]
            return actions