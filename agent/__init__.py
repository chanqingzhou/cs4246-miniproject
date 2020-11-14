try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass
import random
import numpy as np
import time
import torch
import os
import torch.autograd as autograd
import torch.nn as nn
'''
An example to import a Python file.

Uncomment the following lines (both try-except statements) to import everything inside models.py
'''
# try: # server-compatible import statement
#     from models import *
# except: pass

script_path = os.path.dirname(os.path.realpath(__file__))
model1 = os.path.join(script_path, 'modelfile2')
model2 = os.path.join(script_path, 'modelfile2_lookahead')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class carRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Linear(100,50)
        self.rnn = nn.RNN(100, 100, 1,nonlinearity='relu')

    def forward(self, input,hidden):
        output, hidden = self.rnn(input,hidden)
        output = self.readout(output)
        output = torch.sigmoid(output)

        return (output, hidden)

    def initHidden(self):
        return torch.zeros(1,10, 100).to(device)

class ExampleAgent(Agent):
    '''
    An example agent that just output a random action.
    '''
    def __init__(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent with the `test_case_id` (string), which might be important
        if your agent is test case dependent.
        
        For example, you might want to load the appropriate neural networks weight 
        in this method.
        '''
        test_case_id = kwargs.get('test_case_id')
        self.model = carRNN().to(device)
        self.model_look_ahead = carRNN().to(device)
        self.model_look_ahead.load_state_dict(torch.load(model2))
        self.model.load_state_dict(torch.load(model1))
        self.hidden_state = self.model.initHidden()
        self.hidden_state_look = self.model_look_ahead.initHidden()
        self.t_max = kwargs.get('t_max')
        '''
        # Uncomment to help debugging
        print('>>> __INIT__ >>>')
        print('test_case_id:', test_case_id)
        '''

    def initialize(self, **kwargs):
        '''
        [OPTIONAL]
        Initialize the agent.

        Input:
        * `fast_downward_path` (string): the path to the fast downward solver
        * `agent_speed_range` (tuple(float, float)): the range of speed of the agent
        * `gamma` (float): discount factor used for the task

        Output:
        * None

        This function will be called once before the evaluation.
        '''
        fast_downward_path  = kwargs.get('fast_downward_path')
        agent_speed_range   = kwargs.get('agent_speed_range')
        gamma               = kwargs.get('gamma')
        self.t_max = kwargs.get('t_max')

        '''
        # Uncomment to help debugging
        print('>>> INITIALIZE >>>')
        print('fast_downward_path:', fast_downward_path)
        print('agent_speed_range:', agent_speed_range)
        print('gamma:', gamma)
        '''

    def reset(self, state, *args, **kwargs):
        ''' 
        [OPTIONAL]
        Reset function of the agent which is used to reset the agent internal state to prepare for a new environement.
        As its name suggests, it will be called after every `env.reset`.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * None
        '''
        '''
        # Uncomment to help debugging
        print('>>> RESET >>>')
        print('state:', state)
        '''
        self.hidden_state = self.model.initHidden()
        self.hidden_state_look = self.model_look_ahead.initHidden()


    def step(self, state, *args, **kwargs):
        ''' 
        [REQUIRED]
        Step function of the agent which computes the mapping from state to action.
        As its name suggests, it will be called at every step.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * `action`: `int` representing the index of an action or instance of class `Action`.
                    In this example, we only return a random action
        '''
        '''
        # Uncomment to help debugging
        print('>>> STEP >>>')
        print('state:', state)
        '''
        height = len(state[0])
        width = len(state[0][0])
        inputs = [torch.tensor(state[0]).to(device).type(torch.float32),
                            torch.tensor(state[-1]).to(device).type(torch.float32)]
        inputs = torch.cat(inputs, dim=1).unsqueeze(0)

        prob_graph, self.hidden_state = self.model.forward(inputs, self.hidden_state)
        look_prob_graph, self.hidden_state_look = self.model_look_ahead.forward(inputs,self.hidden_state_look)

        agent_y, agent_x = np.where(state[1] == 1)

        agent_x = agent_x[0]
        agent_y = agent_y[0]

        if agent_y == 0:
            actions = [2,3,4]
        else:
            actions = [0,4,3,2]

        if agent_y == len(state[0]) - 1:
            if 1 in actions:
                actions.remove(1)

        # if there is a car in front moving forward will cause the agent to bump into the trial
        if agent_x <= 1 or state[0][agent_y][agent_x-1] == 1:
            if 3 in actions:
                actions.remove(3)
            if 2 in actions:
                actions.remove(2)

        if agent_x <= 2 or state[0][agent_y][agent_x-2] == 1:
            if 2 in actions:
                actions.remove(2)

        if agent_y == 0:
            return actions[0]

        #moving forward too fast will result in suboptimal path
        if agent_x == agent_y + 1:
            if 2 in actions:
                actions.remove(2)
            if 3 in actions:
                actions.remove(3)

        if agent_x == agent_y + 2:
            if 2 in actions:
                actions.remove(2)

        if agent_x <= agent_y:
            return 0

        prob = {}

        for action in actions:
            if action == 0:
                next_x = max(agent_x-1,0)
                next_y = agent_y - 1
                prob[action] = 1 - prob_graph[0][next_y][next_x].item()

            elif action == 1:
                print('never called')
                next_x = max(agent_x-1,0)
                next_y = agent_y + 1
                prob[action] = 1 - prob_graph[0][next_y][next_x].item()
            else:
                next_y = agent_y
                if action == 2:
                    next_x = agent_x - 3
                elif action == 3:
                    next_x = agent_x - 2
                elif action ==4:
                    next_x = agent_x - 1
                final_x = next_x - 1
                final_y = next_y - 1
                prob[action] = (1 - look_prob_graph[0][final_y][final_x].item()) * \
                               (1 - prob_graph[0][next_y][next_x].item())
        original_ratio = width / height - 1  # range
        ratio = agent_x / agent_y - 1  # (4,0)
        # ratio/original 1 -> 0
        # 1-ratio/original 0 -> 1)
        # as (1-ratio/original) ratio goes up, we need to take 0 with higher probability
        factor = 0.075
        start_prob = 0.95 if self.t_max == 50 else 0.94
        risk = min(start_prob - factor * (1 - ratio / original_ratio), start_prob)
        scale  = 0.98 * 0.98 if self.t_max >= 50 else 0.94 * 0.97
        delay = 0
        if prob[0] >= risk:
            assert actions[0] == 0
            return actions[0]
        else:
            best_action = 4
            best_prob = 0
            for i in actions:
                if i == 0:
                    continue
                if prob[i] >= scale + delay:
                    return i
                elif prob[i] - delay > best_prob:
                    best_action = i
                    best_prob = prob[i]
                delay += 0.01
            if ratio > original_ratio:
                return best_action
            else:
                return 4
        #0 is scale * risk
        #1 ius 0.97 * scale
        #2 is 1 + return 4
        '''

        '''



    def update(self, *args, **kwargs):
        '''
        [OPTIONAL]
        Update function of the agent. This will be called every step after `env.step` is called.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`
        * `action` (`int` or `Action`): the executed action (given by the agent through `step` function)
        * `reward` (float): the reward for the `state`
        * `next_state` (same type as `state`): the next state after applying `action` to the `state`
        * `done` (`int`): whether the `action` induce terminal state `next_state`
        * `info` (dict): additional information (can mostly be disregarded)

        Output:
        * None

        This function might be useful if you want to have policy that is dependant to its past.
        '''
        state       = kwargs.get('state')
        action      = kwargs.get('action')
        reward      = kwargs.get('reward')
        next_state  = kwargs.get('next_state')
        done        = kwargs.get('done')
        info        = kwargs.get('info')
        self.prev_state = state
        '''
        # Uncomment to help debugging
        print('>>> UPDATE >>>')
        print('state:', state)
        print('action:', action)
        print('reward:', reward)
        print('next_state:', next_state)
        print('done:', done)
        print('info:', info)
        '''


def create_agent(test_case_id, *args, **kwargs):
    '''
    Method that will be called to create your agent during testing.
    You can, for example, initialize different class of agent depending on test case.
    '''
    return ExampleAgent(test_case_id=test_case_id)


if __name__ == '__main__':
    import sys
    import time
    from env import construct_random_lane_env

    FAST_DOWNWARD_PATH = "/fast_downward/"

    def test(agent, env, runs=1000, t_max=100):
        rewards = []
        agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3,-1), 'gamma' : 1,'t_max':t_max}
        agent.initialize(**agent_init)
        for run in range(runs):
            state = env.reset()
            agent.reset(state)
            episode_rewards = 0.0
            for t in range(t_max):
                action = agent.step(state)
                next_state, reward, done, info = env.step(action)
                full_state = {
                    'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 
                    'done': done, 'info': info
                }

                agent.update(**full_state)
                state = next_state
                episode_rewards += reward
                if done:
                    break
            rewards.append(episode_rewards)
        avg_rewards = sum(rewards)/len(rewards)
        print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
        return avg_rewards

    def timed_test(task):
        start_time = time.time()
        rewards = []
        for tc in task['testcases']:
            agent = create_agent(tc['id'])
            print("[{}]".format(tc['id']), end=' ')
            avg_rewards = test(agent, tc['env'], tc['runs'], tc['t_max'])
            rewards.append(avg_rewards)
        point = sum(rewards)/len(rewards)
        elapsed_time = time.time() - start_time

        print('Point:', point)

        for t, remarks in [(0.4, 'fast'), (0.6, 'safe'), (0.8, 'dangerous'), (1.0, 'time limit exceeded')]:
            if elapsed_time < task['time_limit'] * t:
                print("Local runtime: {} seconds --- {}".format(elapsed_time, remarks))
                print("WARNING: do note that this might not reflect the runtime on the server.")
                break

    def get_task():
        tcs = [('t2_tmax50', 50), ('t2_tmax40', 40)]
        #tcs = [('t2_tmax40', 40)]

        return {
            'time_limit': 600,
            'testcases': [{ 'id': tc, 'env': construct_random_lane_env(), 'runs': 300, 't_max': t_max } for tc, t_max in tcs]
        }

    task = get_task()
    timed_test(task)
