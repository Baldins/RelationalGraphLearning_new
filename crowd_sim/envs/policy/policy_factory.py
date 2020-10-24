from crowd_sim.envs.policy.linear import Linear
from crowd_sim.envs.policy.orca import ORCA, CentralizedORCA
from crowd_sim.envs.policy.socialforce import SocialForce, CentralizedSocialForce
from crowd_sim.envs.policy.ssp import SSP
from crowd_sim.envs.policy.ssp2 import SSP2


def none_policy():
    return None


policy_factory = dict()
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['socialforce'] = SocialForce
policy_factory['centralized_orca'] = CentralizedORCA
policy_factory['centralized_socialforce'] = CentralizedSocialForce
policy_factory['ssp'] = SSP
policy_factory['ssp2'] = SSP2
policy_factory['none'] = none_policy
