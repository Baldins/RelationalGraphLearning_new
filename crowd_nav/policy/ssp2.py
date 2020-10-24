import numpy as np
import math
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class SSP2(Policy):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.kinematics = 'holonomic'
        self.multiagent_training = True


    def configure(self, config):
        assert True

    def predict(self, state):
        self_state = state.self_state

        close_obst = []
        for a in state.human_states:

            distance = math.sqrt((a.px - self_state.px)**2 + (a.py - self_state.py)**2)
            if( distance <= 1):
                close_obst.append([a.px, a.py, a.radius])


        # Go to goal
        if(len(close_obst) == 0):           # No obstacle in sensor skirt
            theta = np.arctan2(self_state.gy-self_state.py, self_state.gx-self_state.px)
            vx = np.cos(theta) * self_state.v_pref
            vy = np.sin(theta) * self_state.v_pref
            action = ActionXY(vx, vy)

        # # Paranoid behavior - stop
        else:

            theta = np.arctan2(self_state.gy-self_state.py, self_state.gx-self_state.px)
            vx = 0.5 * np.cos(theta) * self_state.v_pref
            vy = 0.5 * np.sin(theta) * self_state.v_pref
            action = ActionXY(vx, vy)

        return action
