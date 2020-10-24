from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
import random
import math

class Human(Agent):
    humans = []

    def __init__(self, config, section):
        super().__init__(config, section)
        Human.humans.append(self)
        self.id = None

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    @classmethod
    def get_random_humans(cls, human_num):
        # return random.sample(cls.humans, int(math.ceil(len(Human.humans)/10)))

        return random.sample(cls.humans, int(math.ceil(human_num/10)))
