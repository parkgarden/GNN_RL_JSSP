from mb_agg import *
from agent_utils import *
import torch
import argparse
from Params import configs
import time
import numpy as np


device = torch.device(configs.device)

parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
parser.add_argument('--Pn_j', type=int, default=100, help='Number of jobs of instances to test')
parser.add_argument('--Pn_m', type=int, default=20, help='Number of machines instances to test')
parser.add_argument('--Nn_j', type=int, default=30, help='Number of jobs on which to be loaded net are trained')
parser.add_argument('--Nn_m', type=int, default=20, help='Number of machines on which to be loaded net are trained')
parser.add_argument('--low', type=int, default=1, help='LB of duration')
parser.add_argument('--high', type=int, default=199, help='UB of duration')
parser.add_argument('--seed', type=int, default=200, help='Seed for validate set generation')
params = parser.parse_args()

N_JOBS_P = params.Pn_j
N_MACHINES_P = params.Pn_m
LOW = params.low
HIGH = params.high
SEED = params.seed
N_JOBS_N = params.Nn_j
N_MACHINES_N = params.Nn_m

batch = 20

w_j = 1
w_m_list = [0.5, 1.5]

cmax_results = []
cpu_time_results = []

from JSSP_Env import SJSSP
from PPO_jssp_multiInstances_SimpleWeight import PPO
env = SJSSP(n_j=N_JOBS_P, n_m=N_MACHINES_P)


for w_m in w_m_list:
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
    
    path = './{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH) + '_sw_'+str(w_j)+'_'+str(w_m)+'_v0')
    # path = './SavedNetwork/{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH))
    # ppo.policy.load_state_dict(torch.load(path))
    ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    # ppo.policy.eval()
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                             n_nodes=env.number_of_tasks,
                             device=device)
    # 34 41 41 57 40 56 63 35 67 66 45 67 51 68 68 41 67 30 65 64
    from uniform_instance_gen import uni_instance_gen
    np.random.seed(SEED)
    
    # dataLoaded = np.load('./DataGen/generatedData' + str(N_JOBS_P) + '_' + str(N_MACHINES_P) + '_Seed' + str(SEED) + '.npy')
    dataLoaded = np.load('./DataGen/generatedData' + str(N_JOBS_P) + '_' + str(N_MACHINES_P) + 'PT' + str(LOW) + '_' + str(HIGH) +'_Seed' + str(SEED) + '_batch'+ str(batch) +'.npy')
    dataset = []
    
    for i in range(dataLoaded.shape[0]):
    # for i in range(1):
        dataset.append((dataLoaded[i][0], dataLoaded[i][1]))
    
    # dataset = [uni_instance_gen(n_j=N_JOBS_P, n_m=N_MACHINES_P, low=LOW, high=HIGH) for _ in range(N_TEST)]
    # print(dataset[0][0])
    
    
    result = []
    # torch.cuda.synchronize()
    t1 = time.time()
    for i, data in enumerate(dataset):
        adj, fea, candidate, mask = env.reset(data)
        ep_reward = - env.max_endTime
        # delta_t = []
        t5 = time.time()
        while True:
            # t3 = time.time()
            fea_tensor = torch.from_numpy(fea).to(device)
            adj_tensor = torch.from_numpy(adj).to(device)
            candidate_tensor = torch.from_numpy(candidate).to(device)
            mask_tensor = torch.from_numpy(mask).to(device)
            # t4 = time.time()
            # delta_t.append(t4 - t3)
    
            with torch.no_grad():
                pi, _ = ppo.policy(x=fea_tensor,
                                   graph_pool=g_pool_step,
                                   padded_nei=None,
                                   adj=adj_tensor,
                                   candidate=candidate_tensor.unsqueeze(0),
                                   mask=mask_tensor.unsqueeze(0))
                # action = sample_select_action(pi, omega)
                action = greedy_select_action(pi, candidate)
    
            adj, fea, reward, done, candidate, mask = env.step(action)
            ep_reward += reward
    
            if done:
                break
        t6 = time.time()
        print("cpu time:", t6 - t5)
        # print(max(env.end_time))
        print('Instance' + str(i + 1) + ' makespan:', -ep_reward + env.posRewards)
        result.append(-ep_reward + env.posRewards)
        # print(sum(delta_t))
    # torch.cuda.synchronize()
    t2 = time.time()
    print(t2 - t1)
    # file_writing_obj = open('./' + 'drltime_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + '_sw_'+str(w_j)+'_'+str(w_m)+'_v0.txt', 'w')
    file_writing_obj = open('./' + 'drltime_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + 'PT' + str(LOW) + '_' + str(HIGH) + '_sw_'+str(w_j)+'_'+str(w_m)+'_v0_batch1.txt', 'w')
    file_writing_obj.write(str((t2 - t1)/len(dataset)))
    
    # print(result)
    # print(np.array(result, dtype=np.single).mean())
    # np.save('drlResult_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + '_Seed' + str(SEED) + '_sw_'+str(w_j)+'_'+str(w_m)+'_v0', np.array(result, dtype=np.single))
    
    np.save('drlResult_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + 'PT' + str(LOW) + '_' + str(HIGH) +'_Seed' + str(SEED) + '_sw_'+str(w_j)+'_'+str(w_m)+'_v0_batch1', np.array(result, dtype=np.single))
    # print(np.array(result, dtype=np.single).mean())
    
    print("AVG Cmax:",  sum(result)/len(result))
    print("AVG cpu time:", (t2 - t1)/len(dataset))
    
    cmax_results.append(sum(result)/len(result))
    cpu_time_results.append((t2 - t1)/len(dataset))
    
print(cmax_results)
print(cpu_time_results)

# if __name__ == '__main__':
#     import cProfile

#     cProfile.run('test(dataset)', filename='restats')