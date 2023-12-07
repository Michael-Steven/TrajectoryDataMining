import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
from tqdm import tqdm
import numpy as np
import math
from PIL import Image

split_path = '/home1/shanyanbo/SignalTrajectoryPrediction/data/GPS_mark'
output_path = '/home1/shanyanbo/SignalTrajectoryPrediction/data/traj_img'

T = 1
WP = 0.1  # lng
HP = 0.08  # lat
WM = 40
HM = 40
# data_file = '/home1/shanyanbo/SignalTrajectoryPrediction/data/GPS_mark/summary_1.txt'
# df = pd.read_csv(data_file, header=None,
#                  names=['lat', 'lng', 'dis', 'time', 'type'],
#                  dtype={'lat': float, 'lng': float, 'dis': float, 'time': int, 'type': int})
# all_colums = ['lat', 'lng', 'dis', 'time', 'type']
# # print(df.head(5))
# # print(df.info())#查看后发现没有缺失值
# # print(df.nunique())#除了前两列，其余每列都有重复值
# print(df.describe())  # 查看数据的描述性信息


# def hist(df):
#     df.hist(figsize=(30, 20))
#     # plt.show()
#     plt.savefig('aa.png')


# hist(df[all_colums])


def sampling_points(p):
    index = 0
    # min_lat = p[0][1]
    # min_lng = p[0][2]
    # max_lat = p[0][1]
    # max_lng = p[0][2]
    time = p[0][0]
    Ps = [[p[0][1], p[0][2]]]
    while index < len(p) - 1:
        # min_lat = p[index][1] if p[index][1] < min_lat else min_lat
        # max_lat = p[index][1] if p[index][1] > max_lat else max_lat
        # min_lng = p[index][2] if p[index][2] < min_lng else min_lng
        # max_lng = p[index][2] if p[index][2] > max_lng else max_lng
        # if max_lat - min_lat > HP or max_lng - min_lng > WP:
        #     break
        time += +datetime.timedelta(seconds=T)
        if not (time >= p[index][0] and time <= p[index + 1][0]):
            index += 1
            continue
        if time - p[index][0] > time - p[index + 1][0]:
            Ps.append([p[index][1], p[index][2]])
        else:
            Ps.append([p[index + 1][1], p[index + 1][2]])
    return Ps


def main():
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i in range(1, 7):
        class_path = os.path.join(output_path, str(i))
        if not os.path.exists(class_path):
            os.mkdir(class_path)
    for user in tqdm(os.listdir(split_path)):
        user_path = os.path.join(split_path, user)
        if user == 'hash_1.txt' or user == 'summary_1.txt':
            continue
        for traj in os.listdir(user_path):
            traj_file = os.path.join(user_path, traj)
            # traj_file = '/home1/shanyanbo/SignalTrajectoryPrediction/data/GPS_mark/062/20080831010002_1.plt'
            with open(traj_file, 'r') as f_in:
                df = pd.read_csv(f_in, header=None,
                                 names=['time', 'lat', 'lng', 'asl', 'days', 'delta', 's', 'v', 'a', 'type'],
                                 dtype={'time': str, 'lat': float, 'lng': float, 'asl': float, 'days': float, 'delta': float,
                                        's': float, 'v': float, 'a': float, 'type': int})
                P = []
                time_begin = None
                time_end = None
                for index, row in df.iterrows():
                    type = row['type']
                    time = datetime.datetime.strptime(row['time'], "%Y-%m-%d %H:%M:%S")
                    if time_begin is None:
                        time_begin = datetime.datetime.strptime(row['time'], "%Y-%m-%d %H:%M:%S")
                    time_end = datetime.datetime.strptime(row['time'], "%Y-%m-%d %H:%M:%S")
                    P.append([time, row['lat'], row['lng']])
                if (time_end - time_begin).total_seconds() < 300:
                    continue
                class_path = os.path.join(output_path, str(type))
                PS = sampling_points(P)
                PS = np.array(PS)
                img = np.zeros((WM, HM))
                center_lng = np.mean(PS[:, 1])
                center_lat = np.mean(PS[:, 0])
                min_lng = np.min(PS[:, 1])
                min_lat = np.min(PS[:, 0])
                # print(min_lat, min_lng)
                offset_x = np.floor(WM / 2) - np.floor((center_lng - min_lng) * WM / WP)
                offset_y = np.floor(HM / 2) - np.floor((center_lat - min_lat) * HM / HP)
                # print(offset_x, offset_y)
                for point in PS:
                    x = int(np.floor((point[1] - min_lng) * WM / WP) + offset_x)
                    y = int(np.floor((point[0] - min_lat) * HM / HP) + offset_y)
                    # print(x, y)
                    if 0 <= x < WM and 0 <= y < HM and img[x, y] < 255:
                        img[x, y] += 3
                # print(img)
                img = Image.fromarray(img)
                img = img.convert('L')
                img.save(os.path.join(class_path, traj + '.jpg'))
                # exit(0)


if __name__ == '__main__':
    main()
