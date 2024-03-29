from mb_agg import *
from agent_utils import *
import torch
import numpy as np
import argparse
from Params import configs
import time
from scipy import stats as st
import math

device = configs.device

parser = argparse.ArgumentParser(description='Arguments for test_learned_on_benchmark')
parser.add_argument('--Pn_j', type=int, default=100, help='Number of jobs of instances to test')
parser.add_argument('--Pn_m', type=int, default=20, help='Number of machines instances to test')
parser.add_argument('--Nn_j', type=int, default=10, help='Number of jobs on which to be loaded net are trained')
parser.add_argument('--Nn_m', type=int, default=10, help='Number of machines on which to be loaded net are trained')
parser.add_argument('--which_benchmark', type=str, default='tai', help='Which benchmark to test')
parser.add_argument('--low', type=int, default=1, help='LB of duration')
parser.add_argument('--high', type=int, default=99, help='UB of duration')
parser.add_argument('--seed', type=int, default=200, help='Seed for validate set generation')
params = parser.parse_args()

N_JOBS_P = params.Pn_j
N_MACHINES_P = params.Pn_m
benchmark = params.which_benchmark
N_JOBS_N = params.Nn_j
N_MACHINES_N = params.Nn_m
LOW = params.low
HIGH = params.high
SEED = params.seed

from JSSP_Env import SJSSP
from PPO_jssp_multiInstances import PPO
env = SJSSP(n_j=N_JOBS_P, n_m=N_MACHINES_P)

ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
          n_j=N_JOBS_P,
          n_m=N_MACHINES_P,
          num_layers=configs.num_layers,
          neighbor_pooling_type=configs.neighbor_pooling_type,
          input_dim=configs.input_dim,
          hidden_dim=configs.hidden_dim,
          num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
          num_mlp_layers_actor=configs.num_mlp_layers_actor,
          hidden_dim_actor=configs.hidden_dim_actor,
          num_mlp_layers_critic=configs.num_mlp_layers_critic,
          hidden_dim_critic=configs.hidden_dim_critic)
path = './SavedNetwork/{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH))
ppo.policy.load_state_dict(torch.load(path))
g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                         batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                         n_nodes=env.number_of_tasks,
                         device=device)

from uniform_instance_gen import uni_instance_gen
np.random.seed(SEED)

# dataLoaded = np.load('./DataGen/generatedData' + str(N_JOBS_P) + '_' + str(N_MACHINES_P) + '_Seed' + str(SEED) + '.npy')

# dataLoaded = np.load('./DataGen/generatedData' + str(N_JOBS_P) + '_' + str(N_MACHINES_P) + '_Seed' + str(SEED) + '.npy')
dataLoaded = np.load('./BenchDataNmpy/' + benchmark + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + '.npy')
dataset = []

for i in range(dataLoaded.shape[0]):
# for i in range(1):
    dataset.append((dataLoaded[i][0], dataLoaded[i][1]))

result = []
screen_count = 0
    
for i, data in enumerate(dataset):
    t1 = time.time()
    adj, fea, candidate, mask = env.reset(data)
    ep_reward = - env.max_endTime
    

    while True:
        # Running policy_old:
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
        
        action_list = []
        
        
        with torch.no_grad():
            pi, _ = ppo.policy(x=fea_tensor,
                               graph_pool=g_pool_step,
                               padded_nei=None,
                               adj=adj_tensor,
                               candidate=candidate_tensor.unsqueeze(0),
                               mask=mask_tensor.unsqueeze(0))
            # action = sample_select_action(pi, omega)
            # print(pi)
            # print("max")
            # _, index = pi.squeeze().max(0)
            # print(_, index)
            
            indices = pi.squeeze().argsort()
            numpy_ind = indices.cpu().numpy()
            np.random.shuffle(numpy_ind)
            # print(indices)
            # indeices_screened = []
            idx = 0
            machine_screened = []
            qt_0_75 = N_JOBS_P * 0.25
            
            for j in range(len(candidate)):
                action_list.append(candidate[numpy_ind[j]])
            
            # while True:
            #     # print(torch.mode(pi.squeeze())[0])
            #     # if idx == N_JOBS_P :
            #     # if pi.squeeze()[indices[-1-idx]] <= pi.squeeze().quantile(0.75):
            #     if pi.squeeze()[indices[-1-idx]] <= pi.squeeze()[indices[-1-math.ceil(qt_0_75)]]:
            #     # if idx >= qt_0_75:
            #     # if pi.squeeze()[indices[-1-idx]] <= pi.squeeze().median():
            #         break
            #     else:
            #         action = candidate[indices[-idx-1]]
            #         action_job = action // N_MACHINES_P
            #         action_operationID = action % N_MACHINES_P
            #         action_machine = data[1][action_job][action_operationID]
                    
            #         if action_machine not in machine_screened:
            #             action_list.append(action)
            #             # indeices_screened.append(indices[-j-1])

            #             #screen out neighborhood of action from candidate
            #             # machine_screened.append(action_machine)
                        
            #         else:
            #             screen_count += 1
                        
            #         idx += 1
                    
        
        for j in range(len(action_list)):
            adj, fea, reward, done, candidate, mask = env.step(action_list[j])
            ep_reward += reward

        if done:
            break
    # print(max(env.end_time))
    t2 = time.time()
    print('Instance' + str(i + 1) + ' makespan:', -ep_reward + env.posRewards)
    print('CPU time: ' + str(t2-t1))
    
    result.append(-ep_reward + env.posRewards)
    
print(screen_count/20)
# file_writing_obj = open('./' + 'drltime_'  + '_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + 'jp_WoSo_v0.5.txt', 'w')
file_writing_obj = open('./' + 'drltime_'  + '_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + '_batch1_jp_rand_v0.txt', 'w')
file_writing_obj.write(str((t2 - t1)/len(dataset)))

# print(result)
# print(np.array(result, dtype=np.single).mean())
# np.save('drlResult_' + '_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + 'jp_WoSo_v0.5', np.array(result, dtype=np.single))
np.save('drlResult_' + '_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + '_batch1_jp_rand_v0', np.array(result, dtype=np.single))

print("AVG Cmax:",  sum(result)/len(result))
print("AVG cpu time:", (t2 - t1)/len(dataset))

'''refer = np.array([1231, 1244, 1218, 1175, 1224, 1238, 1227, 1217, 1274, 1241])
refer1 = np.array([2006, 1939, 1846, 1979, 2000, 2006, 1889, 1937, 1963, 1923])
refer2 = np.array([5464, 5181, 5568, 5339, 5392, 5342, 5436, 5394, 5358, 5183])
gap = (np.array(result) - refer2)/refer2
print(gap)
print(gap.mean())'''
