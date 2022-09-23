import os
import click
import numpy as np
import pickle

from baselines import logger
from baselines.common import set_global_seeds

import baselines.eher.experiment.config as config

from baselines.her.rollout import RolloutWorker
from baselines.common.env_util import build_env, get_game_envs

_game_envs = get_game_envs(print_out=False)

@click.command()
@click.argument('policy_file', type=str, default=None)
@click.option('--logdir', type=str, default='./results/her')
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=100)
@click.option('--render', type=int, default=0)
@click.option('--record_video', type=bool, default=False)
def main(policy_file, logdir, seed, n_test_rollouts, render, record_video):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params = config.prepare_params(params)

    logdir = os.path.join(logdir, f"eher_{env_name}_{seed}")
    if logdir or logger.get_dir() is None:
        logger.configure(dir=logdir, format_strs=['stdout', 'log', 'csv'])
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_Q': True,
        'rollout_batch_size': 1,
        'render': bool(render),
    }

    for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]

    env_params = dict(seed=seed, env_name=env_name, num_env=1)
    env = build_env(env_params, _game_envs, record_video=record_video)
    evaluator = RolloutWorker(env, policy, dims, logger, **eval_params)
    # evaluator.seed(seed)

    # # Run evaluation.
    # evaluator.clear_history()
    # for i in range(n_test_rollouts):
    #     episode =  evaluator.generate_rollouts()

    # # record logs
    # for key, val in evaluator.logs('test'):
    #     logger.record_tabular(key, np.mean(val))
    # logger.dump_tabular()

    total_reward, num_success = 0, 0
    for episode in range(n_test_rollouts):
        episode_rew = np.zeros(1)
        episode_scs = np.zeros(1)
        obs = env.reset()
        for step in range(50):
            actions, _, _, _ = policy.step(obs)
            obs, rew, done, info = env.step(actions)
            episode_rew += rew
            success = np.array([i.get('is_success', 0.0) for i in info])
            episode_scs += success
        if episode_scs > 0:
            num_success += 1
        total_reward += episode_rew
    print(f"success rate: {num_success/n_test_rollouts}")
    print(f"total reward: {total_reward/n_test_rollouts}")


if __name__ == '__main__':
    main()
