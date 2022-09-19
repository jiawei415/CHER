import os
import click
import numpy as np
import pickle

from baselines import logger
from baselines.common import set_global_seeds

import baselines.cher.experiment.config as config

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

    logdir = os.path.join(logdir, f"cher_{env_name}_{seed}")
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

    env_params = dict(seed=seed, env_name=env_name, num_env=1, record_video=record_video)
    env = build_env(env_params, _game_envs)
    evaluator = RolloutWorker(env, policy, dims, logger, **eval_params)
    # evaluator.seed(seed)


    # Run evaluation.
    evaluator.clear_history()
    for _ in range(n_test_rollouts):
        evaluator.generate_rollouts()

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    main()
