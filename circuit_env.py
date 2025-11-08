import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import os
import yaml
import math
from ngspice_interface import DUT as DUT_NGSpice
from utils.plotting import plotLearning, plot_running_maximum

class CircuitEnv(gym.Env):
    PER_LOW, PER_HIGH = -np.inf, +np.inf
    
    def __init__(self, config=None, circuit_name='TwoStage', run_id='rllib_baseline',
                 success_threshold=0.0, simulator='ngspice'):
        self.run_id = run_id
        self.max_steps_per_episode = 10
        self.env_steps = 0
        self.episode_steps = 0
        self.success_threshold = success_threshold
        self.circuit_name = circuit_name
        self.score = 0.0

        project_path = os.getcwd()
        yaml_directory = os.path.join(project_path, f"{simulator}_interface", 'files', 'yaml_files')
        circuit_yaml_path = os.path.join(yaml_directory, f'{circuit_name}.yaml')
        with open(circuit_yaml_path, 'r') as f:
            yaml_data = yaml.load(f, Loader=yaml.Loader)
        
        self.dict_params = yaml_data['params']
        self.dict_targets = yaml_data['targets']

        self.hard_constraints = yaml_data['hard_constraints'] #gain, noise,phm,slewrate and ugbw
        self.optimization_targets = yaml_data['optimization_targets'] #area and current

        # number of input action components
        self.n_actions = len(self.dict_params) #20 

        # number of output observation components
        self.obs_dim = len(self.dict_targets) #7

        self.param_ranges = {}
        for name, value in self.dict_params.items():
           self.param_ranges[name] = {'min': value[0], 'max': value[1], 'step': value[2]}
        
        self.simulation_engine = DUT_NGSpice(circuit_yaml_path)

        print(f"\n Initialized {circuit_name} with simulator {simulator} \n")

        # Action space & Observation space
        act_high = np.array([1 for _ in range(self.n_actions)])
        act_low = np.array([-1 for _ in range(self.n_actions)])
        self.action_space = Box(low=act_low, high=act_high)
        obs_high = np.array([CircuitEnv.PER_HIGH]*self.obs_dim, dtype=np.float32)
        obs_low = np.array([CircuitEnv.PER_LOW]*self.obs_dim, dtype=np.float32)
        self.observation_space = Box(low=obs_low, high=obs_high)

        # spec reward weights
        self.spec_weights = yaml_data['spec_weights']

        self.reward_history = []
        self.score_history = []

        self.counter = 0
        self.pvt_corner = {'process': 'TT', 'voltage': 1.2, 'temp': 27}
    
    def action_refine(self, action):
        """
        TODO: Implement a function that converts normalized actions to actual parameter values.
        
        This function should:
        1. Take a flattened numpy array of actions (values between -1 and 1)
        2. Convert each action value to the actual parameter value from self.parameter_ranges
        
        You have two options;
            1) Continuous mapping: convert the action directly to the parameter value based on the min-max values
            2) Discretize mapping: create a vector of possible values using the min-max-step from self.parameter_ranges, 
                                    then, map the action to an index of that vector and retreive the corresponding actual value.

        Args:
            action: numpy array of normalized values between -1 and 1
            
        Returns:
            dict: Dictionary mapping parameter names to their actual sizing values
                e.g., {'mp1': 13, 'wp1':2.0e-06, 'lp1':9.0e-08, ...}
        
        """
        action = np.array(action).flatten()
        real_params = {}
        for i, (param_name, range_info) in enumerate(self.param_ranges.items()):
            vmin = range_info['min']
            vmax = range_info['max']

            # Continuous mapping from [-1, 1] → [vmin, vmax]
            real_value = (action[i] + 1) * 0.5 * (vmax - vmin) + vmin

            real_params[param_name] = real_value

        return real_params

        pass
    
    def simulate(self, params):
        """
        TODO: Create/Simulate netlist with the given parameters and return the measured metrics.

        Args:
            params (dict): A dictionary containing the parameters for the simulation.

        Returns:
            dict: A dictionary containing the measured metrics from the simulation.
        """
        new_netlist_path = self.simulation_engine.create_new_netlist(params, process=self.pvt_corner['process'], vdd=self.pvt_corner['voltage'], temp_pvt=self.pvt_corner['temp'])
        self.simulation_engine.simulate(new_netlist_path)
        dict = self.simulation_engine.measure_metrics()
        return dict
        pass
    
    def normalize_specs(self, spec_dict):
        """
        TODO: Normalize the specifications in `spec_dict` based on target specifications.

        This function normalizes the values in `spec_dict` by comparing them to the
        target specifications stored in `self.dict_targets`. The normalization is 
        performed using the formula:
            normalized_value = (spec_value - goal_value) / (spec_value + goal_value) 
        
        Args:
            spec_dict (dict): A dictionary containing the specifications to be normalized.
                              The keys should match those in `self.dict_targets`.
        
        Returns:
            dict: A dictionary containing the normalized specifications, with the same keys
                  as `spec_dict`.
        """
        norm_spec_dict = {}
        for key, spec_value in spec_dict.items():
            goal_value = self.dict_targets[key]
            if (spec_value + goal_value)==0:
                normalized_value = 0
            else:
                normalized_value = (spec_value - goal_value) / (spec_value + goal_value)    
            norm_spec_dict[key] = normalized_value 
        return norm_spec_dict
        pass
    
    def evaluate(self, action):
        self.param_values = self.action_refine(action)
        self.real_specs = self.simulate(self.param_values)
        self.cur_norm_specs = self.normalize_specs(self.real_specs)
    
    def reset(self, *, seed=None, options=None):
        """
        TODO: Reset the environment to an initial state and return the initial observation.

        Parameters:
            No mandatory input parameters (leave the input signature as is).
            seed (int, optional): A seed for the random number generator to ensure reproducibility.
            options (dict, optional): Additional options for the reset process.

        Returns:
            np.ndarray: The initial observation of the environment, which is the normalized current specifications.

        This method performs the following steps:
        1. Initializes the episode steps counter to zero.
        2. Generates a random action within the range [-1, 1] for each action parameter.
        3. Evaluates the environment with the generated random action.
        4. Resets the episode score to zero.
        5. Constructs the initial observation by returning the normalized current specifications.

        Note:
        - The observation is returned as a NumPy array of type float32.
        """
        self.episode_steps = 0
        random_action = np.random.uniform(low=-1.0, high=1.0, size=self.n_actions)
        ob, reward, done, info = self.step(random_action)
        self.episode_steps = 0
        self.score = 0.0
        return ob

        pass
    
    def step(self, action):
        """
        TODO: Perform a single step in the environment using the given action.

        This function should:
        1. Evaluate the given action.
        2. Compute the reward and check if the hard constraints are satisfied.
        3. Update the current observation.
        4. Append the reward to the reward history and update the score.
        5. Create output directories if they do not exist.
        6. If the goal state is reached, plot the running maximum reward.
        7. Increment the environment and episode step counters.
        8. Check if the maximum steps per episode have been reached, setting the done flag if true.
        9. Every 10 steps, update the score history, also plot the learning curve, and reset the score.
        10. Return the current observation, reward, done flag, and additional information.

        Args:
            action: The action to be taken in the environment. An array of values between -1 and 1.

        Returns:
            tuple: A tuple containing:
            - ob (np.ndarray): The current observation of the environment.
            - reward (float): The reward obtained from taking the action.
            - done (bool): A flag indicating whether the episode has ended.
            - info (dict): Additional information, including whether the goal state was reached.
        """
        done = False
        info = {}
        self.evaluate(action)
        reward, hard_satisfied = self.reward_computation(self.cur_norm_specs)
        
        self.reward_history.append(reward)
        self.score += reward
        info["goal"] = hard_satisfied
    
        obs_values = []                     
        for key,value in self.cur_norm_specs.items():                
            obs_values.append(value)         
        ob = np.array(obs_values, dtype=np.float32)

        out_dir = f'./output_figs/{self.run_id}/'
        os.makedirs(out_dir, exist_ok=True)

        if hard_satisfied and len(self.reward_history) > 0:
            plot_running_maximum(self.reward_history,self.run_id)
        
        self.env_steps += 1
        self.episode_steps += 1
        
        if self.episode_steps >=self.max_steps_per_episode:
            self.score_history.append(self.score)
            done = True
            plotLearning(self.score_history,self.run_id)
            self.score = 0.0
        return ob, reward, done, info
        pass
    
    def reward_computation(self, norm_specs):
        """
        TODO: Compute the reward based on normalized specifications and hard constraints.

        Args:
            norm_specs (dict): A dictionary containing normalized specifications.

        Returns:
            tuple: A tuple containing:
            - reward (float): The computed reward value.
            - hard_satisfied (bool): A boolean indicating whether the hard constraints are satisfied.

        The function performs the following steps:
        1. Initialize the reward to 0.0 and hard_satisfied to False.
        2. Iterate over the hard constraints and adjust the reward based on the specifications.
        3. Check if all hard constraints are satisfied or the total reward passes the success threshold of 0.
            - If so, set hard_satisfied to True, add a bonus reward (+0.3), and adjust the reward based on optimization targets.
            - If not, add weighted reward components of the optimization targets.
        4. Return the computed reward and the hard_satisfied flag.
        """
        reward = 0.0
        hard_satisfied =False
        all_satisfied = True
        r_H = 0.0
        r_T = 0.0
        for spec_name in self.hard_constraints:
            if spec_name == "noise":
                r_y = -max(norm_specs[spec_name],0)
            else:
                r_y  = min(norm_specs[spec_name],0)
            r_H += r_y
        for spec_name in self.optimization_targets:
            r_t = -norm_specs[spec_name]
            r_T += r_t
        if r_H >= self.success_threshold:
            all_satisfied = True
            reward = 0.3 + r_T
        else:
            reward = r_H + 0.05 * r_T
        return reward, hard_satisfied


if __name__ == '__main__':
    env = CircuitEnv(
        circuit_name='TwoStage', 
        run_id=0, 
        simulator='ngspice', 
        success_threshold=0.0
        )
    print(env.action_space)
    print(env.observation_space)

    ob = env.reset()
    print("Initial observation: ", ob)
    print("Initial parameters: ", env.param_values)
    print("Initial specs: ", env.real_specs)

    action = np.random.uniform(-1, 1, [env.n_actions])
    ob, reward, done, info = env.step(action)
    print("Next parameters: ", env.param_values)
    print("Next observation: ", ob)
    print("Next specs: ", env.real_specs)
    print("Reward: ", reward)
    print("Done: ", done)

    parameters_1 = {
    'mp1': 6,
    'wp1': 2.25e-06,
    'lp1': 1.35e-07,
    'mn1': 10,
    'wn1': 5e-07,
    'ln1': 1.35e-07,
    'mp3': 10,
    'wp3': 1.75e-06,
    'lp3': 9e-08,
    'mn3': 1,
    'wn3': 1e-06,
    'ln3': 1.35e-07,
    'mn4': 9,
    'wn4': 1.25e-06,
    'ln4': 1.35e-07,
    'mn5': 6,
    'wn5': 7.5e-07,
    'ln5': 9e-08,
    'cap': 8e-13,
    'res': 9500.0
    }
    parameters_2 = {
    'mp1': 15,
    'wp1': 1.25e-06,
    'lp1': 9e-08,
    'mn1': 10,
    'wn1': 1e-06,
    'ln1': 9e-08,
    'mp3': 12,
    'wp3': 1.25e-06,
    'lp3': 9e-08,
    'mn3': 15,
    'wn3': 1.5e-06,
    'ln3': 9e-08,
    'mn4': 11,
    'wn4': 1.5e-06,
    'ln4': 9e-08,
    'mn5': 11,
    'wn5': 1e-06,
    'ln5': 9e-08,
    'cap': 4.5e-12,
    'res': 4800.0
    }
    parameters_3 = {
    'mp1': 7,
    'wp1': 1.75e-06,
    'lp1': 1.35e-07,
    'mn1': 20,
    'wn1': 2.25e-06,
    'ln1': 1.35e-07,
    'mp3': 22,
    'wp3': 2.25e-06,
    'lp3': 1.35e-07,
    'mn3': 21,
    'wn3': 1e-06,
    'ln3': 9e-08,
    'mn4': 8,
    'wn4': 7.5e-07,
    'ln4': 9e-08,
    'mn5': 17,
    'wn5': 2.5e-07,
    'ln5': 4.5e-08,
    'cap': 8e-13,
    'res': 3900.0
    }
    parameters_4 = {
    'mp1': 8,
    'wp1': 1.5e-06,
    'lp1': 1.35e-07,
    'mn1': 15,
    'wn1': 1.75e-06,
    'ln1': 1.35e-07,
    'mp3': 18,
    'wp3': 2e-06,
    'lp3': 1.35e-07,
    'mn3': 15,
    'wn3': 1.25e-06,
    'ln3': 9e-08,
    'mn4': 13,
    'wn4': 1.5e-06,
    'ln4': 9e-08,
    'mn5': 17,
    'wn5': 5e-07,
    'ln5': 4.5e-08,
    'cap': 3.5e-12,
    'res': 6000.0
    }
