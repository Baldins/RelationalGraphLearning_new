from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.gcn import GCN
from crowd_nav.policy.model_predictive_rl import ModelPredictiveRL
from crowd_nav.policy.ssp import SSP
from crowd_nav.policy.ssp2 import SSP2



policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['gcn'] = GCN
policy_factory['ssp'] = SSP
policy_factory['ssp2'] = SSP2
policy_factory['model_predictive_rl'] = ModelPredictiveRL
