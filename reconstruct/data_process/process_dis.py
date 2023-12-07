import numpy as np

region_distance = np.load('/home1/shanyanbo/PeriodicMove-main/Dataset/region_data/region_distance_no_less10.npy')
print('Shape of region distance:', region_distance.shape)
print(type(region_distance))
print(region_distance[0][1])
