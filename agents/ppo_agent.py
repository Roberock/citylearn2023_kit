from stable_baselines3 import PPO
from citylearn.agents.rbc import BasicRBC, Agent

class PPO_comfort_agent(Agent):
    """ Can be any subclass of citylearn.agents.base.Agent """
    def __init__(self, env):
        super().__init__(env)
        self.agent_load_dir = './agents/models/'
        self.agent_load_name = 'PPO_citylearn2023_comfort_5M'
        self.policy = PPO.load(self.agent_load_dir + self.agent_load_name).policy
        self.exploit = True

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        return self.predict(observations)

    def predict(self, obs):
        """ Just a passthrough, can implement any custom logic as needed """
        return self.policy.predict(obs[0], deterministic=self.exploit)


class PPO_mix_agent(Agent):
    """ Can be any subclass of citylearn.agents.base.Agent """
    def __init__(self, env):
        super().__init__(env)
        self.agent_load_dir = './agents/models/'
        self.agent_load_name = 'PPO_citylearn2023_mixed_reward'
        self.policy = PPO.load(self.agent_load_dir + self.agent_load_name).policy
        self.exploit = True

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        return self.predict(observations)

    def predict(self, observations):
        """ Just a passthrough, can implement any custom logic as needed """
        return self.policy.predict(observations[0], deterministic=self.exploit)

class PPO_mix_agent_530(Agent):
    """ Can be any subclass of citylearn.agents.base.Agent """
    def __init__(self, env):
        super().__init__(env)
        self.agent_load_dir = './agents/models/'
        self.agent_load_name = 'PPO_citylearn2023_mixed_reward_530k'
        self.policy = PPO.load(self.agent_load_dir + self.agent_load_name).policy
        self.exploit = True

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        return self.predict(observations)

    def predict(self, observations):
        """ Just a passthrough, can implement any custom logic as needed """
        return self.policy.predict(observations[0], deterministic=self.exploit)


class PPO_mix_agent_week_hour_feature(Agent):
    """ Can be any subclass of citylearn.agents.base.Agent """
    def __init__(self, env):
        super().__init__(env)
        self.agent_load_dir = './agents/models/'
        self.agent_load_name = 'PPO_citylearn2023_mix_week_hour_feature'
        self.policy = PPO.load(self.agent_load_dir + self.agent_load_name).policy
        self.exploit = True

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        return self.predict(observations)

    def predict(self, observations):
        """ Just a passthrough, can implement any custom logic as needed """

        hour_week = (observations[0][0]-1)*24+observations[0][1]
        obs = [hour_week] + observations[0]
        return self.policy.predict(obs, deterministic=self.exploit)

class PPO_mix_agent_week_hour_feature_v02(Agent):
    """ Can be any subclass of citylearn.agents.base.Agent """
    def __init__(self, env):
        super().__init__(env)
        self.agent_load_dir = './agents/models/'
        self.agent_load_name = 'PPO_citylearn2023_mix_week_hour_feature_v0.2'
        self.policy = PPO.load(self.agent_load_dir + self.agent_load_name).policy
        self.exploit = True

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        return self.predict(observations)

    def predict(self, observations):
        """ Just a passthrough, can implement any custom logic as needed """

        hour_week = (observations[0][0]-1)*24+observations[0][1]
        obs = [hour_week] + observations[0]
        return self.policy.predict(obs, deterministic=self.exploit)