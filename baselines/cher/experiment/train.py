import os
import sys
import copy
import time
import json
import click
import numpy as np
from mpi4py import MPI
from baselines import logger
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
from baselines.common.env_util import build_env, get_game_envs
import baselines.cher.experiment.config as config
import baselines.cher.config_curriculum as config_cur

_game_envs = get_game_envs(print_out=False)


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, random_init, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    # random_init for o/g/rnd stat and model training
    if random_init:
        logger.info('Random initializing ...')
        rollout_worker.clear_history()
        # rollout_worker.render = True
        for epi in range(int(random_init) // rollout_worker.rollout_batch_size): 
            episode = rollout_worker.generate_rollouts(random_ac=True)
            policy.store_episode(episode)

    logger.info("Training...")
    best_success_rate = -1
    for epoch in range(n_epochs):
        # train
        time_start = time.time()
        config_cur.learning_step = 0
        rollout_worker.clear_history()
        for _ in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train()
            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        time_end = time.time()
        total_time = time_end - time_start
        logger.record_tabular('epoch/num', epoch)
        logger.record_tabular('epoch/time(min)', total_time/60)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(env, num_env,
    env_name, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return,
    override_params={}, save_policies=False
):
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        whoami = mpi_fork(num_cpu)
        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        logdir = os.path.join(logdir, f"cher_{env_name}_{seed}")
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    if env_name.startswith('Point2D'):
        params.update(config.DEFAULT_ENV_PARAMS['Point2D'])
    if env_name.startswith('PointMass'):
        params.update(config.DEFAULT_ENV_PARAMS['PointMass'])
    elif env_name.startswith('FetchReach'):
        params.update(config.DEFAULT_ENV_PARAMS['FetchReach'])
    elif env_name.startswith('Fetch'):
        params.update(config.DEFAULT_ENV_PARAMS['Fetch'])
    elif env_name.startswith('SawyerReach'):
        params.update(config.DEFAULT_ENV_PARAMS['SawyerReach'])
    elif env_name.startswith('Sawyer'):
        params.update(config.DEFAULT_ENV_PARAMS['Sawyer'])
    elif env_name.startswith('Hand'):
        params.update(config.DEFAULT_ENV_PARAMS['Hand'])
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    params = config.prepare_params(params)
    params['rollout_batch_size'] = num_env

    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        dump_params = copy.deepcopy(params)
        for key, value in params.items():
            dump_params[key] = str(value)
        json.dump(dump_params, f)
    config.log_params(params, logger=logger)

    random_init = params['random_init']
    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    eval_env =  env
    rollout_worker = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)
    evaluator = RolloutWorker(eval_env, policy, dims, logger, **eval_params)

    train(
        logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies, random_init=random_init)


@click.command()
@click.option('--env_name', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default='~/results', help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=50, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--num_env', type=int, default=1, help='Number of environment copies being run')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')


def main(**kwargs):
    env = build_env(kwargs, _game_envs)
    kwargs.update({"env": env})
    launch(**kwargs)


if __name__ == '__main__':
    main()
