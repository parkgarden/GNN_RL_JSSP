import numpy as np
from uniform_instance_gen import uni_instance_gen

j = 100
m = 20
l = 1
h = 199
batch_size = 1
seed = 200

np.random.seed(seed)

data = np.array([uni_instance_gen(n_j=j, n_m=m, low=l, high=h) for _ in range(batch_size)])
print(data.shape)
np.save('generatedData{}_{}_PT{}_{}_Seed{}_batch{}.npy'.format(j, m, l, h, seed,batch_size), data)