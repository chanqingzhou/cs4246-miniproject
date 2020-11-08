try:
    from runner.abstracts import Agent
except:
    class Agent(object):
        pass
import random
import numpy as np
import time
import torch

'''
An example to import a Python file.

Uncomment the following lines (both try-except statements) to import everything inside models.py
'''
# try: # server-compatible import statement
#     from models import *
# except: pass
try:  # local-compatible import statement
    from .models import *
except:
    pass


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
        fast_downward_path = kwargs.get('fast_downward_path')
        agent_speed_range = kwargs.get('agent_speed_range')
        gamma = kwargs.get('gamma')
        self.max_speed = [999, 0]
        self.prev_state = None

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
        self.speed_count = []
        self.prev_state = None
        self.count = 0

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
        negative_weighting_for_action_1 = 0.035
        height = len(state[0])
        width = len(state[0][0])

        if self.speed_count == []:
            for i in range(height):
                self.speed_count.append({})

        if self.prev_state == None:
            return 4, 1

        self.count += 1
        for h in range(height):
            prev_channel = self.prev_state[0][h]
            channel = state[0][h]

            channel_cars = np.where(channel == 1)[0]
            prev_channel_cars = np.where(prev_channel == 1)[0]
            prev_channel_car = prev_channel_cars[-1]
            for i in range(len(channel_cars) - 1, -1, -1):
                car = channel_cars[i]
                if prev_channel_car >= car:
                    speed = prev_channel_car - car
                    if speed > self.max_speed[1]:
                        self.max_speed[1] = speed
                    if speed < self.max_speed[0]:
                        self.max_speed[0] = speed
                    if speed in self.speed_count[h]:
                        self.speed_count[h][speed] += 1
                    else:
                        self.speed_count[h][speed] = 2
                    break

        agent_y, agent_x = np.where(state[1]==1)

        agent_x = agent_x[0]
        agent_y = agent_y[0]
        if agent_y == 9 and agent_x == 49:
            return 4,0

        if agent_y == 0:
            actions = [2,3,4]
        else:
            actions = [0,4,3,2]

        if agent_y == len(state[0]) -1:
            if 1 in actions:
                actions.remove(1)

        if agent_x <= 1 or state[0][agent_y][agent_x-1] ==1:
            if 3 in actions:
                actions.remove(3)
            if 2 in actions:
                actions.remove(2)
        if agent_x <= 2 or state[0][agent_y][agent_x-2] == 1:
            if 2 in actions:
                actions.remove(2)

        if agent_y == 0:
            return actions[0],0

        if agent_x == agent_y + 1:
            if 2 in actions:
                actions.remove(2)
            if 3 in actions:
                actions.remove(3)

        if agent_x == agent_y + 2:
            if 2 in actions:
                actions.remove(2)

        if agent_x <= agent_y:
            return 0,0.95

        # construct curve one tick in front

        prob = []
        min_speed = self.max_speed[0]
        max_speed = self.max_speed[1]

        for action in actions:
            if action == 0:
                next_x = max(agent_x-1,0)
                next_y = agent_y - 1
            elif action == 1:
                next_x = max(agent_x-1,0)
                next_y = agent_y + 1
            else:
                next_y = agent_y
                if action == 2:
                    next_x = agent_x - 3
                elif action == 3:
                    next_x = agent_x -2
                elif action ==4:
                    next_x = agent_x -1

            channel = state[0][next_y]

            max_speed_so_far = max(list(self.speed_count[next_y].keys()))
            min_speed_so_far = min(list(self.speed_count[next_y].keys()))
            max_value = max(list(self.speed_count[next_y].values()))
            hit = 0
            total = 0

            for speed in range(min_speed,max_speed+1):

                multiplier = 1

                if min_speed_so_far <= speed <= max_speed_so_far:
                    multiplier = max_value
                if speed == 0:
                    if channel[next_x] == 1:
                        hit += 1 * multiplier
                    total += 1 * multiplier
                else:
                    for i in range(1,speed+1):

                        if np.roll(channel, -i)[next_x] == 1:
                            hit += 1 * multiplier
                        total += 1 * multiplier
            if action == 1:
                prob.append((total-hit)/total-negative_weighting_for_action_1)
            else:
                prob.append((total - hit) / total)



        # construct look ahead curve:
        look_ahead_prob = []
        empty_channel = np.zeros([10,50])
        for next_y in range(height):
            max_speed_so_far = max(list(self.speed_count[next_y].keys()))
            min_speed_so_far = min(list(self.speed_count[next_y].keys()))
            max_value = max(list(self.speed_count[next_y].values()))
            hit = 0
            total = 0
            channel = state[0][next_y]
            for speed in range(min_speed,max_speed+1):

                multiplier = 1

                if min_speed_so_far <= speed <= max_speed_so_far:
                    multiplier =(self.count + 5) /3
                empty_channel += np.roll(channel, -speed) * multiplier

        for action in actions:
            if action == 0:
                look_ahead_prob.append(0)
                continue
            elif action ==1:
                next_y = agent_y + 1
                next_x = agent_x - 1
            else:
                next_y = agent_y
                if action == 2:
                    next_x = agent_x - 3
                elif action == 3:
                    next_x = agent_x -2
                elif action ==4:
                    next_x = agent_x -1
            final_x = next_x - 1
            final_y = next_y - 1
            max_speed_so_far = max(list(self.speed_count[final_y].keys()))
            min_speed_so_far = min(list(self.speed_count[final_y].keys()))
            channel = empty_channel[final_y]
            final_channel = np.zeros(50)
            for speed in range(min_speed, max_speed + 1):
                multiplier = 1
                if min_speed_so_far <= speed <= max_speed_so_far:
                    multiplier = (self.count + 5) / 3

                for i in range(1, speed + 1):
                    final_channel += np.roll(channel, -i) * multiplier

            final_multiplier = (max_speed_so_far + min_speed_so_far + 2)/2 * sum(state[0][final_y])

            look_ahead_prob.append((sum(final_channel)-final_multiplier * final_channel[final_x])/sum(final_channel))




        if prob[0] >= 0.95:
            assert actions[0] == 0
            return actions[0], 0.95

        else:
            final_prob = []
            for i in range(len(prob)):
                final_prob.append(look_ahead_prob[i]*prob[i])
            max_prob = max(final_prob)
            if max_prob/(sum(final_prob[1:])/len(final_prob)-1) >1.05:
                return actions[final_prob.index(max_prob)], max_prob
            return 4,0

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
        state = kwargs.get('state')
        action = kwargs.get('action')
        reward = kwargs.get('reward')
        next_state = kwargs.get('next_state')
        done = kwargs.get('done')
        info = kwargs.get('info')
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
        agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3, -1), 'gamma': 1}
        agent.initialize(**agent_init)
        total_actions = 0
        total_prob = 0
        total_failed = 0
        total_failed_prob = 0
        for run in range(runs):
            state = env.reset()
            agent.reset(state)
            episode_rewards = 0.0
            action_1 = 0
            for t in range(t_max):
                action, prob = agent.step(state)
                next_state, reward, done, info = env.step(action)
                full_state = {
                    'state': state, 'action': action, 'reward': reward, 'next_state': next_state,
                    'done': done, 'info': info
                }

                total_actions += 1
                total_prob += prob
                agent.update(**full_state)
                state = next_state
                episode_rewards += reward
                if done:
                    if not reward:
                        total_failed_prob += prob
                        total_failed += 1
                    break
            rewards.append(episode_rewards)
        avg_rewards = sum(rewards) / len(rewards)
        print("{} run(s) avg rewards : {:.1f}".format(runs, avg_rewards))
        print("avg prob:{}, failed prob:{}".format(str(total_prob / total_actions),
                                                   str(total_failed_prob / total_failed)))
        return avg_rewards


    def timed_test(task):
        start_time = time.time()
        rewards = []
        for tc in task['testcases']:
            agent = create_agent(tc['id'])
            print("[{}]".format(tc['id']), end=' ')
            avg_rewards = test(agent, tc['env'], tc['runs'], tc['t_max'])
            rewards.append(avg_rewards)
        point = sum(rewards) / len(rewards)
        elapsed_time = time.time() - start_time

        print('Point:', point)

        for t, remarks in [(0.4, 'fast'), (0.6, 'safe'), (0.8, 'dangerous'), (1.0, 'time limit exceeded')]:
            if elapsed_time < task['time_limit'] * t:
                print("Local runtime: {} seconds --- {}".format(elapsed_time, remarks))
                print("WARNING: do note that this might not reflect the runtime on the server.")
                break


    def get_task():
        tcs = [('t2_tmax50', 50), ('t2_tmax40', 40)]
        return {
            'time_limit': 600,
            'testcases': [{'id': tc, 'env': construct_random_lane_env(), 'runs': 300, 't_max': t_max} for tc, t_max in
                          tcs]
        }


    task = get_task()
    timed_test(task)
