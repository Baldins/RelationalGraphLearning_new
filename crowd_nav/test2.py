import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import random
import math
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.human import Human


def main(args):
    # configure logging and device
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    logging.info('Using device: %s', device)

    if args.model_dir is not None:
        if args.config is not None:
            config_file = args.config
        else:
            config_file = os.path.join(args.model_dir, 'configs/icra_benchmark/config.py')
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
            logging.info('Loaded IL weights')
        elif args.rl:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                print(os.listdir(args.model_dir))
                model_weights = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
            logging.info('Loaded RL weights')
        else:
            model_weights = os.path.join(args.model_dir, 'best_val.pth')
            logging.info('Loaded RL weights with best VAL')

    else:
        config_file = args.config

    spec = importlib.util.spec_from_file_location('config', config_file)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure policy
    policy_config = config.PolicyConfig(args.debug)
    policy = policy_factory[policy_config.name]()
    if args.planning_depth is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_depth = args.planning_depth
    if args.planning_width is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_width = args.planning_width
    if args.sparse_search:
        policy_config.model_predictive_rl.sparse_search = True

    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.load_model(model_weights)

    # configure environment
    env_config = config.EnvConfig(args.debug)

    if args.human_num is not None:
        env_config.sim.human_num = args.human_num
    # env = gym.make('CrowdSim-v0')
    # env.configure(env_config)
    env = gym.make('CrowdSim_mixed-v0')
    env.configure(env_config)

    PPL = env.human_num


    if args.square:
        env.test_scenario = 'square_crossing'
    if args.circle:
        env.test_scenario = 'circle_crossing'
    if args.test_scenario is not None:
        env.test_scenario = args.test_scenario

    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    robot.time_step = env.time_step
    robot.set_policy(policy)
    explorer = Explorer(env, robot, device, None, gamma=0.9)

    train_config = config.TrainConfig(args.debug)
    epsilon_end = train_config.train.epsilon_end
    if not isinstance(robot.policy, ORCA):
        robot.policy.set_epsilon(epsilon_end)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = args.safety_space
        else:
            robot.policy.safety_space = args.safety_space
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()


    ppl_local = []
    robot_states = []
    robot_vel = []
    non_attentive_humans = []
    # non_attentive_humans = Human.get_random_humans(env.human_num)
    #
    # non_attentive_humans = set(non_attentive_humans)
    for case in range(args.test_case):
        rewards = []

        ob = env.reset(non_attentive_humans, test_case=case)

        # non_attentive_humans = Human.get_random_humans(env.human_num)

        # ob = env.reset(non_attentive_humans, test_case=case)

        # print(case)
        # print(non_attentive_humans)

        # ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(robot.get_position())
        non_attentive_humans = random.sample(env.humans, int(math.ceil(env.human_num/2)))

        non_attentive_humans = set(non_attentive_humans)

        while not done:

            action = robot.act(ob, non_attentive_humans)
            ob, _, done, info, ppl_count, robot_pose, robot_velocity, dmin = env.step(action, non_attentive_humans)
            rewards.append(_)

            ppl_local.append(ppl_count)
            robot_states.append(robot_pose)
            robot_vel.append(robot_velocity)


            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos

        gamma = 0.9
        cumulative_reward = sum([pow(gamma, t * robot.time_step * robot.v_pref)
             * reward for t, reward in enumerate(rewards)])

        if args.visualize:

            if args.traj:
                env.render('traj', args.video_file)
            else:
                if args.video_dir is not None:
                    if policy_config.name == 'gcn':
                        args.video_file = os.path.join(args.video_dir, policy_config.name + '_' + policy_config.gcn.similarity_function)
                    else:
                        args.video_file = os.path.join(args.video_dir, policy_config.name)
                        # args.video_file = os.path.join(args.video_dir, policy_config.name + '_' + f'test_case_{case}'+f'human_num_{env.human_num}')

                    args.video_file = args.video_file + '_' + args.phase + '_' + str(args.test_case) + '.mp4'
                env.render(case,  args.policy, args.human_policy, env.human_num, args.trained_env,  'video', args.video_file)

        logging.info('It takes %.2f seconds to finish. Final status is %s, cumulative_reward is %f', env.global_time, info, cumulative_reward)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))

        # if robot.visible and info == 'reach goal':
        #     human_times = env.get_human_times()
        #
        #     logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))

        # logging.info('PPl counter ', ppl_local)
        main = "Results/"
        human_policy = args.human_policy
        if human_policy == 'socialforce':
            maindir = 'SocialForce/'
            if not os.path.exists(main+maindir):
                os.mkdir(main+maindir)
        else:
            maindir = 'ORCA/'
            if not os.path.exists(main+maindir):
                os.mkdir(main+maindir)

        robot_policy = args.policy
        trained_env = args.trained_env
        if robot_policy == 'ssp':
            method_dir = 'ssp/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)
        if (robot_policy == 'model_predictive_rl'and trained_env == 'orca'):
            method_dir = 'model_predictive_rl/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)
        if (robot_policy == 'model_predictive_rl' and trained_env == 'socialforce'):
            method_dir = 'model_predictive_rl_social/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)

        if (robot_policy == 'rgl'and trained_env == 'orca'):
            method_dir = 'rgl/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)
        if (robot_policy == 'rgl' and trained_env == 'socialforce'):
            method_dir = 'rgl_social/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)

        if robot_policy == 'ssp2':
            method_dir = 'ssp2/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)
        elif robot_policy == 'cadrl':
            method_dir = 'cadrl/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)
        elif robot_policy == 'sarl':
            method_dir = 'sarl/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)
        elif robot_policy == 'lstm_rl':
            method_dir = 'lstm_rl/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)
        elif robot_policy == 'orca':
            method_dir = 'orca/'
            if not os.path.exists(main+maindir + method_dir):
                os.mkdir(main+maindir + method_dir)

        robot_data = pd.DataFrame()
        robot_data['robot_x'] = np.array(robot_states)[:, 0]
        robot_data['robot_y'] = np.array(robot_states)[:, 1]
        robot_data['local_ppl_cnt'] = np.array(ppl_local)

        out_name = f'robot_data{case}.csv'

        if not os.path.exists(main+maindir + method_dir + f'{PPL}/'):
            os.mkdir(main+maindir + method_dir + f'{PPL}/')
        # outdir = f'{PPL}/robot_data_{PPL}/'
        if not os.path.exists(main+maindir + method_dir + f'{PPL}/robot_data_{PPL}/'):
            os.mkdir(main+maindir + method_dir + f'{PPL}/robot_data_{PPL}/')

        fullname = os.path.join(main+maindir + method_dir + f'{PPL}/robot_data_{PPL}/', out_name)

        robot_data.to_csv(fullname, index=True)

        if not os.path.exists(main+maindir + method_dir + f'{PPL}/time_{PPL}'):
            os.mkdir(main+maindir + method_dir + f'{PPL}/time_{PPL}')
        Time_data = pd.DataFrame()
        Time_data['time (s)'] = [env.global_time]
        Time_data['mean_local'] = np.mean(ppl_local)
        Time_data['std_local'] = np.std(ppl_local)
        Time_data['collision_flag'] = info
        Time_data['dmin'] = dmin
        Time_data['reward'] = cumulative_reward

        Time_data.to_csv(main+maindir + method_dir + f'{PPL}/time_{PPL}/robot_time_data_seconds_{PPL}_{case}.csv')

        if not os.path.exists(main+maindir + method_dir + f'{PPL}/localdensity_{PPL}'):
            os.mkdir(main+maindir + method_dir + f'{PPL}/localdensity_{PPL}')
        LD = pd.DataFrame()
        LD['local_ppl_cnt'] = np.array(ppl_local)
        LD['vx'] = np.array(robot_vel)[:, 0]
        LD['vy'] = np.array(robot_vel)[:, 1]
        LD.to_csv(main+maindir + method_dir + f'{PPL}/localdensity_{PPL}/localdensity_{PPL}_{case}.csv')



        # else:
        #     explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)
        #     if args.plot_test_scenarios_hist:
        #         test_angle_seeds = np.array(env.test_scene_seeds)
        #         b = [i * 0.01 for i in range(101)]
        #         n, bins, patches = plt.hist(test_angle_seeds, b, facecolor='g')
        #         plt.savefig(os.path.join(args.model_dir, 'test_scene_hist.png'))
        #         plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default='configs/icra_benchmark/rgl.py')
    parser.add_argument('--policy', type=str, default='rgl')
    parser.add_argument('-m', '--model_dir', type=str, default='data/output')
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default="/home/fbaldini/Desktop")
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--human_num', type=int, default=None)
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=True, action='store_true')
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--sparse_search', default=False, action='store_true')
    parser.add_argument('--human_policy',  type=str, default='socialforce')
    parser.add_argument('--trained_env',  type=str, default='orca')


    sys_args = parser.parse_args()

    main(sys_args)
