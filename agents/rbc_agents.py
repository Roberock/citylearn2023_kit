from citylearn.agents.rbc import BasicRBC, Agent
from citylearn.agents.rbc import OptimizedRBC
import numpy as np

class BasicRBCAgent(BasicRBC):
    """ Can be any subclass of citylearn.agents.base.Agent """
    def __init__(self, env):
        super().__init__(env)

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)

    def predict(self, observations):
        """ Just a passthrough, can implement any custom logic as needed """
        return super().predict(observations)



class OptimizedRBCAgent(OptimizedRBC):
    """ Can be any subclass of citylearn.agents.base.Agent """
    def __init__(self, env):
        super().__init__(env)

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)

    def predict(self, observations):
        """ Just a passthrough, can implement any custom logic as needed """
        return super().predict(observations)



class BasicRBCAgent_week_hour(Agent):
    """ Can be any subclass of citylearn.agents.base.Agent """
    def __init__(self, env):
        super().__init__(env)
        # Load the model using the default protocol (protocol 5)
        self.actions_tabular = np.genfromtxt('./agents/models/action_week_hour.csv', delimiter=',')

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)

    def predict(self, observations):
        """ Just a passthrough, can implement any custom logic as needed """
        hour_week = (observations[0][0]-1)*24+observations[0][1]
        n_buildings = len(self.env.buildings_metadata)
        act_bi = self.actions_tabular[hour_week-1]
        actions_district = [np.hstack([act_bi for _ in range(n_buildings)], dtype='float32')]
        # fixme
        return actions_district

