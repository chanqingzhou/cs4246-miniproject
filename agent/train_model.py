import torch
import torch.autograd as autograd
import torch.nn as nn
from models import carRNN
import torch.optim as optim
try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass
import random
import numpy as np
import time
from env import construct_random_lane_env



def get_task():
    tcs = [('t2_tmax50', 50), ('t2_tmax40', 40)]
    return {
        'time_limit': 600,
        'testcases': [{'id': tc, 'env': construct_random_lane_env(), 'runs': 300, 't_max': t_max} for tc, t_max in tcs]
    }

if __name__ == '__main__':
    learning_rate = 0.001

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = carRNN().to(device)
    history = []
    optimizer = optim.Adam(model.parameters(),learning_rate)
    env = construct_random_lane_env()
    state = env.reset()
    history = [[torch.tensor(state[0]).to(device).type(torch.float32),
                            torch.tensor(state[-1]).to(device).type(torch.float32)]]
    print_interval = 20
    count = 0
    while True:
        look_ahead= 2
        next_state, reward, done, info = env.step(4)
        if not done:
            history.append([torch.tensor(next_state[0]).to(device).type(torch.float32),
                            torch.tensor(next_state[-1]).to(device).type(torch.float32)])
        else:
            losses = []
            hidden = model.initHidden()
            for i in range(len(history)-look_ahead):
                hist = history[i]

                inputs = torch.cat(hist,dim=1).unsqueeze(0)
                output, hidden = model.forward(inputs,hidden)

                desired_output = (history[i+2][0] + history[i+look_ahead][1])
                desired_output[desired_output > 0] = 1

                desired_output = desired_output.type(torch.float32).unsqueeze(0)
                loss = nn.functional.binary_cross_entropy(output, desired_output)
                losses.append(loss)

            if len(losses) > 0:
                optimizer.zero_grad()
                loss = sum(losses) / len(losses)
                loss.backward(retain_graph=True)
                optimizer.step()
                count +=1

            if count%print_interval ==0:
                print(loss)
            if count%(print_interval*2)==0:
                torch.save(model.state_dict(), 'modelfile2_lookahead')

            state = env.reset()
            history = [[torch.tensor(state[0]).to(device).type(torch.float32),
                        torch.tensor(state[-1]).to(device).type(torch.float32)]]