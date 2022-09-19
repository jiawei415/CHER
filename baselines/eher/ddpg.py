from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea
from baselines import logger
from baselines.her.util import import_function, store_args, flatten_grads, transitions_in_episode_batch
from baselines.her.normalizer import Normalizer
from baselines.eher.replay_buffer import ReplayBuffer, ReplayBufferEnergy, PrioritizedReplayBuffer
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
from baselines.common.mpi_adam import MpiAdam
from baselines.common import tf_util



def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class DDPG(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, last_hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, gamma, temperature, prioritization, env_name, k_heads, share_network,
                 alpha, beta0, beta_iters, eps, max_timesteps, rank_method, reuse=False, **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        self.prioritization = prioritization
        self.env_name = env_name
        self.temperature = temperature
        self.rank_method = rank_method

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        stage_shapes['w'] = (None,)
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T, self.dimg)
        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size

        if self.prioritization == 'energy':
            self.buffer = ReplayBufferEnergy(buffer_shapes, buffer_size, self.T, self.sample_transitions, 
                                            self.prioritization, self.env_name)
        elif self.prioritization == 'tderror':
            self.buffer = PrioritizedReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions, alpha, self.env_name)
            if beta_iters is None:
                beta_iters = max_timesteps
            self.beta_schedule = LinearSchedule(beta_iters, initial_p=beta0, final_p=1.0)
        else:
            self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def get_actions(self, kth_head, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf_dict[kth_head]]
        if compute_Q:
            vals += [policy.Q_pi_tf_dict[kth_head]]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)

        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def get_td_errors(self, o, g, u):
        o, g = self._preprocess_og(o, g, g)
        vals = [self.td_error_tf]
        r = np.ones((o.reshape(-1, self.dimo).shape[0],1))

        feed = {
            self.target.o_tf: o.reshape(-1, self.dimo),
            self.target.g_tf: g.reshape(-1, self.dimg),
            self.bath_tf_r: r,
            self.main.o_tf: o.reshape(-1, self.dimo),
            self.main.g_tf: g.reshape(-1, self.dimg),
            self.main.u_tf: u.reshape(-1, self.dimu)
        }
        td_errors = self.sess.run(vals, feed_dict=feed)
        td_errors = td_errors.copy()

        return td_errors

    def store_episode(self, episode_batch, dump_buffer, w_potential, w_linear, w_rotational, rank_method, clip_energy, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        if self.prioritization == 'tderror':
            self.buffer.store_episode(episode_batch, dump_buffer)
        elif self.prioritization == 'energy':
            self.buffer.store_episode(episode_batch, w_potential, w_linear, w_rotational, rank_method, clip_energy)
        else:
            self.buffer.store_episode(episode_batch)

        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            
            if self.prioritization == 'energy':
                if not self.buffer.current_size==0 and not len(episode_batch['ag'])==0:
                    transitions = self.sample_transitions(episode_batch, num_normalizing_transitions, 'none', 1.0, True)
            elif self.prioritization == 'tderror':
                transitions, weights, episode_idxs = \
                self.sample_transitions(self.buffer, episode_batch, num_normalizing_transitions, beta=0)
            else:
                transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)


            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def dump_buffer(self, epoch):
        self.buffer.dump_buffer(epoch)

    def _sync_optimizers(self):
        for i in range(self.k_heads):
            self.Q_adams[i].sync()
            self.pi_adams[i].sync()

    def _grads(self, kth_head):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_ops[kth_head],
            self.main.Q_pi_tf_dict[kth_head],
            self.Q_grads[kth_head],
            self.pi_grads[kth_head],
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad

    def _update(self, kth_head, Q_grad, pi_grad):
        self.Q_adams[kth_head].update(Q_grad, self.Q_lr)
        self.pi_adams[kth_head].update(pi_grad, self.pi_lr)

    def sample_batch(self, t):

        if self.prioritization == 'energy':
            transitions = self.buffer.sample(self.batch_size, self.rank_method, temperature=self.temperature)
            weights = np.ones_like(transitions['r']).copy()
        elif self.prioritization == 'tderror':
            transitions, weights, idxs = self.buffer.sample(self.batch_size, beta=self.beta_schedule.value(t))
        else:
            transitions = self.buffer.sample(self.batch_size)
            weights = np.ones_like(transitions['r']).copy()

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        transitions['w'] = weights.flatten().copy() # note: ordered dict
        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]

        if self.prioritization == 'tderror':
            return (transitions_batch, idxs)
        else:
            return transitions_batch

    def stage_batch(self, t, batch=None): #
        if batch is None:
            if self.prioritization == 'tderror':
                batch, idxs = self.sample_batch(t)
            else:
                batch = self.sample_batch(t)
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

        if self.prioritization == 'tderror':
            return idxs

    def train(self, t, dump_buffer, stage=True):
        if not self.buffer.current_size==0:
            for i in range(self.k_heads):
                if stage:
                    self.stage_batch(t)
                critic_loss, actor_loss, Q_grad, pi_grad = self._grads(i)
                self._update(i, Q_grad, pi_grad)
                self.critic_loss_dict[i].append(critic_loss)
                self.actor_loss_dict[i].append(actor_loss)
            return critic_loss, actor_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        for i in range(self.k_heads):
            self.sess.run(self.update_target_net_ops[i])

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf_util.get_session()
        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('u_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.u_stats = Normalizer(self.dimu, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])
        batch_tf['w'] = tf.reshape(batch_tf['w'], [-1, 1])

        # network
        with tf.variable_scope(f'main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope(f'target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = self.create_actor_critic(target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()

        self.log_op_list = [self.o_stats.mean, self.o_stats.std, self.g_stats.mean, self.g_stats.std, self.u_stats.mean, self.u_stats.std]

        # build ops for train
        self.Q_adams, self.pi_adams = {}, {}
        self.Q_grads, self.pi_grads = {}, {}
        self.Q_loss_ops, self.pi_loss_ops = {}, {}
        self.Q_train_ops, self.pi_train_ops = {}, {}
        self.update_target_net_ops = {}
        for i in range(self.k_heads):
            # critic loss
            target_Q_pi_tf = self.target.Q_pi_tf_dict[i]
            clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
            target_Q_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
            td_error_tf = tf.square(tf.stop_gradient(target_Q_tf) - self.main.Q_tf_dict[i])
            Q_loss_tf = tf.reduce_mean(td_error_tf)
            self.Q_loss_ops[i] = Q_loss_tf
            # actor_loss
            main_Q_pi_tf = self.main.Q_pi_tf_dict[i]
            pi_reg_tf = tf.square(self.main.pi_tf_dict[i] / self.max_u)
            pi_loss_tf = -tf.reduce_mean(main_Q_pi_tf) + self.action_l2 * tf.reduce_mean(pi_reg_tf)
            self.pi_loss_ops[i] = pi_loss_tf
            # update main net ops
            main_Q_vars = self._vars('main/shared_Q') + self._vars(f'main/Q_{i}/')
            main_pi_vars = self._vars('main/shared_pi') + self._vars(f'main/pi_{i}/')
            Q_grads_tf = tf.gradients(Q_loss_tf, main_Q_vars)
            pi_grads_tf = tf.gradients(pi_loss_tf, main_pi_vars)
            assert len(main_Q_vars) == len(Q_grads_tf)
            assert len(main_pi_vars) == len(pi_grads_tf)
            self.Q_grads[i] = flatten_grads(grads=Q_grads_tf, var_list=main_Q_vars)
            self.pi_grads[i] = flatten_grads(grads=pi_grads_tf, var_list=main_pi_vars)
            self.Q_adams[i] = MpiAdam(main_Q_vars, scale_grad_by_procs=False)
            self.pi_adams[i] = MpiAdam(main_pi_vars, scale_grad_by_procs=False)
            # self.Q_train_ops[i] = tf.train.AdamOptimizer(self.Q_lr).minimize(Q_loss_tf, var_list=main_Q_vars)
            # self.pi_train_ops[i] = tf.train.AdamOptimizer(self.pi_lr).minimize(pi_loss_tf, var_list=main_pi_vars)
            # update target net ops
            main_vars = main_Q_vars + main_pi_vars
            target_vars = self._vars(f'target/shared_Q') + self._vars(f'target/Q_{i}/') + self._vars('target/shared_pi') + self._vars(f'target/pi_{i}/')
            self.update_target_net_ops[i] = list(
                map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(target_vars, main_vars))) # polyak averaging

        main_vars, target_vars = self._vars('main'), self._vars('target')
        self.init_target_net_op = list(map(lambda v: v[0].assign(v[1]), zip(target_vars, main_vars)))

        self.sess.run(tf.variables_initializer(self._global_vars(""))) # init global vars
        self._sync_optimizers()
        self._init_target_net()
        self.critic_loss_dict, self.actor_loss_dict = {k: [] for k in range(self.k_heads)}, {k: [] for k in range(self.k_heads)}

    def logs(self, prefix=''):
        o_mean, o_std, g_mean, g_std, u_mean, u_std = self.sess.run(self.log_op_list)
        logs = []
        logs += [('stats/o_mean', np.mean(o_mean)), ('stats/o_std', np.mean(o_std))]
        logs += [('stats/g_mean', np.mean(g_mean)), ('stats/g_std', np.mean(g_std))]
        logs += [('stats/u_mean', np.mean(u_mean)), ('stats/u_std', np.mean(u_std))]
        for i, (critic_loss, actor_loss) in enumerate(zip(self.critic_loss_dict.values(), self.actor_loss_dict.values())):
            logs += [(f'head/head_{i}_critic_loss', np.mean(critic_loss))]
            logs += [(f'head/head_{i}_actor_loss', np.mean(actor_loss))]
            self.critic_loss_dict[i], self.actor_loss_dict[i]  = [], []
        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs



    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None
        state['env_name'] = None # No need for playing the policy

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)
