from contextlib import contextmanager
import torch
import time
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import memory as mem
from feedforward import Feedforward

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(device)
torch.set_num_threads(1)

class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100],
                 learning_rate = 0.0002):
        # TODO: Setup network with right input and output size (using super().__init__)
        super().__init__(
            input_size=observation_dim+action_dim,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )
        # END
        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss()

    def fit(self, observations, actions, targets): # all arguments should be torch tensors
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass

        pred = self.Q_value(observations,actions)
        # Compute Loss
        loss = self.loss(pred, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        # TODO: implement the forward pass.
        s_a_tuple = torch.cat((observations, actions), dim=1)
        return self.forward(s_a_tuple) 


# Ornstein Uhlbeck noise, Nothing to be done here
class OUNoise():
    def __init__(self, shape, theta: float = 0.15, dt: float = 1e-2):
        self._shape = shape
        self._theta = theta
        self._dt = dt
        self.noise_prev = np.zeros(self._shape)
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * ( - self.noise_prev) * self._dt
            + np.sqrt(self._dt) * np.random.normal(size=self._shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape)


@contextmanager
def catchtime(time_dict, name) -> float:
    """Context manager to measure time yields the measured time and then modifies it after exiting the with block.
    """
    start_ts = time.time()
    try:
        yield
    finally:
        time_dict[name] = time.time() - start_ts
    

class DDPGAgent(object):
    """
    Agent implementing DDPG.
    """
    def __init__(self, observation_space, action_space, **userconfig):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))

        self._observation_space = observation_space
        self._obs_dim=self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        self._config = {
            "eps": 0.1,            # Epsilon: noise strength to add to policy
            "discount": 0.95,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "learning_rate_actor": 0.00001,
            "learning_rate_critic": 0.0001,
            "hidden_sizes_actor": [128,128],
            "hidden_sizes_critic": [128,128,64],
            "update_target_every": 100,
            "use_target_net": True
        }
        self._config.update(userconfig)
        self._eps = self._config['eps']
        self._discount = self._config['discount']

        self.action_noise = OUNoise((self._action_n))

        self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        # Q Network
        self.Q = QFunction(observation_dim=self._obs_dim,
                           action_dim=self._action_n,
                           hidden_sizes= self._config["hidden_sizes_critic"],
                           learning_rate = self._config["learning_rate_critic"])
        # target Q Network
        self.Q_target = QFunction(observation_dim=self._obs_dim,
                                  action_dim=self._action_n,
                                  hidden_sizes= self._config["hidden_sizes_critic"],
                                  learning_rate = 0)

        high, low = torch.from_numpy(self._action_space.high), torch.from_numpy(self._action_space.low)
        # TODO:
        # The activation function of the policy should limit the output the action space
        # and makes sure the derivative goes to zero at the boundaries
        # Use Tanh, which is between -1 and 1 and scale it to [low, high]
        # Hint: use torch.nn.Tanh()(x)
        def scaled_tanh(x):
            tanh = torch.nn.Tanh()(x)
            # First scale to [0, 1]
            tanh = (tanh + 1) / 2
            # Then scale to [low, high]
            tanh = tanh * (high - low) + low
            assert tanh.shape == x.shape
            return tanh
           
        output_activation = lambda x: scaled_tanh(x)

        self.policy = Feedforward(input_size=self._obs_dim,
                                  hidden_sizes= self._config["hidden_sizes_actor"],
                                  output_size=self._action_n,
                                  activation_fun = torch.nn.ReLU(),
                                  output_activation = output_activation)
        self.policy_target = Feedforward(input_size=self._obs_dim,
                                         hidden_sizes= self._config["hidden_sizes_actor"],
                                         output_size=self._action_n,
                                         activation_fun = torch.nn.ReLU(),
                                         output_activation = output_activation)

        self._copy_nets()

        self.optimizer=torch.optim.Adam(self.policy.parameters(),
                                        lr=self._config["learning_rate_actor"],
                                        eps=0.000001)
        self.train_iter = 0

    def _copy_nets(self):
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def act(self, observation, eps=None):
        # TODO: implement this: use self.action_noise() (which provides normal noise with standard variance)
        if eps is None:
            eps = self._eps
        observation = torch.from_numpy(observation.astype(np.float32))
        action = self.policy(observation).detach().numpy()
        return action + eps * self.action_noise()


    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def state(self):
        return (self.Q.state_dict(), self.policy.state_dict())

    def restore_state(self, state):
        self.Q.load_state_dict(state[0])
        self.policy.load_state_dict(state[1])
        self._copy_nets()

    def reset(self):
        self.action_noise.reset()

    def train(self, iter_fit=32, time_dict={}):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        losses = []
        self.train_iter+=1
        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self._copy_nets()

        for i in range(iter_fit):
            # sample from the replay buffer
            with catchtime(time_dict, 'sample'):
                data=self.buffer.sample(batch=self._config['batch_size'])

            s = to_torch(np.stack(data[:,0])) # s_t
            a = to_torch(np.stack(data[:,1])) # a_t
            rew = to_torch(np.stack(data[:,2])[:,None]) # rew  (batchsize,1)
            s_prime = to_torch(np.stack(data[:,3])) # s_t+1
            done = to_torch(np.stack(data[:,4])[:,None]) # done signal  (batchsize,1)
            # TODO: Implement the rest of the algorithm

            # Optimize critic 
            # Compute the target Q value 
            q_prime = self.Q_target.Q_value(s_prime, self.policy_target(s_prime))
            td_target = rew + self._discount * (1.0 - done) * q_prime

            with catchtime(time_dict, 'fit critic'):
                q_loss_value = self.Q.fit(s, a, td_target)

            # Optimize actor
            with catchtime(time_dict, 'fit actor'):
                self.optimizer.zero_grad()

                # Actor loss is negative Q value for current policy
                actor_loss = -self.Q.Q_value(s, self.policy(s)).mean(dim=0) 

                actor_loss.backward()
                self.optimizer.step()

            # assign q_loss_value  and actor_loss to we stored in the statistics


            losses.append((q_loss_value , actor_loss.item()))

        return losses

