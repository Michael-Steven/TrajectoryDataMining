import os
import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd

data_path = '/home1/shanyanbo/SignalTrajectoryPrediction/data/Geolife'
gps_path = '/home1/shanyanbo/SignalTrajectoryPrediction/data/GPS2'
split_path = '/home1/shanyanbo/SignalTrajectoryPrediction/data/GPS_split'

transportation = {
    'walk': 1,
    'bike': 2,
    'car': 3,
    'taxi': 3,
    'bus': 4,
    'subway': 5,
    'train': 6,
}


def hav(theta):
    s = np.sin(theta / 2)
    return s * s


def get_distance_hav(lat0, lng0, lat1, lng1):
    EARTH_RADIUS = 6371
    lat0 = np.radians(lat0)
    lat1 = np.radians(lat1)
    lng0 = np.radians(lng0)
    lng1 = np.radians(lng1)

    dlng = np.fabs(lng0 - lng1)
    dlat = np.fabs(lat0 - lat1)
    h = hav(dlat) + np.cos(lat0) * np.cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(h))
    return distance * 1000


def _read_label_data(label_file):
    label_data = []
    with open(label_file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip().split()
            # print(line)
            if not line[4] in transportation.keys():
                continue
            time_begin = datetime.datetime.strptime(str(line[0]) + ' ' + str(line[1]), "%Y/%m/%d %H:%M:%S")
            time_end = datetime.datetime.strptime(str(line[2]) + ' ' + str(line[3]), "%Y/%m/%d %H:%M:%S")
            label_data.append([time_begin, time_end, transportation[line[4]]])
    return label_data


def _process_one_traj(traj_file, output_file, label_data):
    with open(traj_file, 'r') as f_in, open(output_file, 'w') as f_out:
        df = pd.read_csv(f_in, header=None, skiprows=6,
                         names=['lng', 'lat', 'zero', 'asl', 'days', 'date', 'time'],
                         dtype={'lng': float, 'lat': float, 'zero': int, 'asl': float, 'days': float, 'date': str, 'time': str})
        cal = []
        for index, row in df.iterrows():
            time = datetime.datetime.strptime(row['date'] + ' ' + row['time'], "%Y-%m-%d %H:%M:%S")
            if cal != []:
                time_pre = datetime.datetime.strptime(cal[-1]['date'] + ' ' + cal[-1]['time'], "%Y-%m-%d %H:%M:%S")
                if time == time_pre:
                    continue
            cal.append(row)
            if len(cal) < 3:
                continue
            if len(cal) > 3:
                cal.pop(0)
            time_0 = datetime.datetime.strptime(cal[0]['date'] + ' ' + cal[0]['time'], "%Y-%m-%d %H:%M:%S")
            time_1 = datetime.datetime.strptime(cal[1]['date'] + ' ' + cal[1]['time'], "%Y-%m-%d %H:%M:%S")
            time_2 = datetime.datetime.strptime(cal[2]['date'] + ' ' + cal[2]['time'], "%Y-%m-%d %H:%M:%S")
            dis_1 = get_distance_hav(cal[1]['lat'], cal[1]['lng'], cal[0]['lat'], cal[0]['lng'])
            dis_2 = get_distance_hav(cal[2]['lat'], cal[2]['lng'], cal[1]['lat'], cal[1]['lng'])
            v_1 = dis_1 / (time_1 - time_0).total_seconds()
            v_2 = dis_2 / (time_2 - time_1).total_seconds()
            a = abs(v_2 - v_1) / (time_2 - time_1).total_seconds()
            f_out.write("%s,%.6f,%.6f,%.4f,%.10f,%d,%.4f,%.4f,%.4f" % (str(time_2), cal[2]['lng'], cal[2]['lat'],
                                                                       cal[2]['asl'], cal[2]['days'],
                                                                       (time_2 - time_1).total_seconds(), dis_2, v_2, a))
            flag = False
            if label_data is not None:
                for label in label_data:
                    if time_2 >= label[0] and time_2 <= label[1]:
                        flag = True
                        f_out.write(",%d\n" % label[2])
                        break
            if flag is False:
                f_out.write(",0\n")


def add_label():
    if not os.path.exists(gps_path):
        os.mkdir(gps_path)
    for user in tqdm(os.listdir(data_path)):
        user_path = os.path.join(data_path, user)
        output_user_path = os.path.join(gps_path, user)
        if not os.path.exists(output_user_path):
            os.mkdir(output_user_path)
        user_data = os.listdir(user_path)
        label_file = os.path.join(
            user_path, 'labels.txt') if 'labels.txt' in user_data else None
        label_data = _read_label_data(label_file) if label_file else None
        traj_path = os.path.join(user_path, 'Trajectory')
        for traj in os.listdir(traj_path):
            traj_file = os.path.join(traj_path, traj)
            output_file = os.path.join(output_user_path, traj)
            _process_one_traj(traj_file, output_file, label_data)


def _process_one_split(traj_file, output_user_path):
    index = 0
    with open(traj_file, 'r') as f_in:
        # df = pd.read_csv(f_in, header=None,
        #                  names=['time', 'lng', 'lat', 'asl', 'days', 'delta', 's', 'v', 'a', 'type'],
        #                  dtype={'time': str, 'lat': float, 'asl': float, 'days': float, 'delta': float,
        #                         's': float, 'v': float, 'a': float, 'type': int})
        lines = f_in.readlines()
        output_file = os.path.join(output_user_path, traj_file[-18:-4] + '_' + str(index) + '.plt')
        pre_type = -1
        # for line in lines:
        #     df = line.strip().split(',')
        #     interval = int(df[5])
        #     if interval > 150 or int(df[9]) != pre_type:
        #         pre_type = int(df[9])
        #         index += 1
        #         output_file = os.path.join(output_user_path, traj_file[-18:-4] + '_' + str(index) + '.plt')
        #     with open(output_file, 'a') as f_out:
        #         f_out.write(line)
        for line in lines:
            df = line.strip().split(',')
            interval = int(df[5])
            if interval > 300 or int(df[9]) != pre_type:
                pre_type = int(df[9])
                index += 1
                output_file = os.path.join(output_user_path, traj_file[-18:-4] + '_' + str(index) + '.plt')
            if int(df[9]) != 0:
                if not os.path.exists(output_user_path):
                    os.mkdir(output_user_path)
                with open(output_file, 'a') as f_out:
                    f_out.write(line)


def split_traj():
    if not os.path.exists(split_path):
        os.mkdir(split_path)
    for user in tqdm(os.listdir(gps_path)):
        user_path = os.path.join(gps_path, user)
        output_user_path = os.path.join(split_path, user)
        # if not os.path.exists(output_user_path):
        #     os.mkdir(output_user_path)
        for traj in os.listdir(user_path):
            traj_file = os.path.join(user_path, traj)
            _process_one_split(traj_file, output_user_path)


def summary():
    summary_file = os.path.join(split_path, 'summary_1.txt')
    summaries = []
    hash_file = os.path.join(split_path, 'hash_1.txt')
    hashes = []
    for user in tqdm(os.listdir(split_path)):
        user_path = os.path.join(split_path, user)
        if user == 'hash_1.txt' or user == 'summary_1.txt':
            continue
        for traj in os.listdir(user_path):
            traj_file = os.path.join(user_path, traj)
            # traj_file = '/home1/shanyanbo/SignalTrajectoryPrediction/data/GPS_mark/020/20110911000506_5.plt'
            with open(traj_file, 'r') as f_in:
                df = pd.read_csv(f_in, header=None,
                                 names=['time', 'lat', 'lng', 'asl', 'days', 'delta', 's', 'v', 'a', 'type'],
                                 dtype={'time': str, 'lat': float, 'lng': float, 'asl': float, 'days': float, 'delta': float,
                                        's': float, 'v': float, 'a': float, 'type': int})
                min_lat = 1000
                max_lat = -1000
                min_lng = 1000
                max_lng = -1000
                cnt = 0
                way = None
                time_begin = None
                time_end = None
                distance = 0
                pre_lat = None
                pre_lng = None
                for index, row in df.iterrows():
                    if time_begin is None:
                        time_begin = datetime.datetime.strptime(row['time'], "%Y-%m-%d %H:%M:%S")
                    time_end = datetime.datetime.strptime(row['time'], "%Y-%m-%d %H:%M:%S")
                    # if pre_lat is None:
                    #     pre_lat = row['lat']
                    #     pre_lng = row['lng']
                    # else:
                    #     distance += get_distance_hav(pre_lat, pre_lng, row['lat'], row['lng'])
                    #     pre_lat = row['lat']
                    #     pre_lng = row['lng']
                    cnt += 1
                    min_lat = row['lat'] if row['lat'] < min_lat else min_lat
                    max_lat = row['lat'] if row['lat'] > max_lat else max_lat
                    min_lng = row['lng'] if row['lng'] < min_lng else min_lng
                    max_lng = row['lng'] if row['lng'] > max_lng else max_lng
                    way = row['type']
                distance = max(get_distance_hav(max_lat, max_lng, max_lat, min_lng), get_distance_hav(max_lat, max_lng, min_lat, max_lng))
                if cnt > 10:
                    # print(max_lat, min_lat, max_lng, min_lng)
                    summaries.append("%.6f,%.6f,%.4f,%d,%d" % (
                        max_lat - min_lat,
                        max_lng - min_lng,
                        distance,
                        (time_end - time_begin).total_seconds(),
                        way))
                    # summaries.append("%.2f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%d" % (
                    #     distance,
                    #     (time_end - time_begin).total_seconds(),
                    #     df['v'].mean(),
                    #     df['v'].quantile(0.95),
                    #     df['v'].var(),
                    #     df['a'].mean(),
                    #     df['a'].max(),
                    #     way))
                    hashes.append(traj_file)
            # print(summaries)
            # exit(0)
    with open(summary_file, 'w') as f_out:
        for item in summaries:
            print(item, file=f_out)
    with open(hash_file, 'w') as f_out:
        for item in hashes:
            print(item, file=f_out)


def to_csv():
    with open("geolife.csv", "w") as f:
        f.write("\"TIMESTAMP\",\"POLYLINE\"\n")
        for user in tqdm(os.listdir(split_path)):
            user_path = os.path.join(split_path, user)
            for traj in os.listdir(user_path):
                traj_file = os.path.join(user_path, traj)
                # print(traj_file)
                times = traj_file.split("/")[-1][:8]
                # print(times)
                out = []
                with open(traj_file, 'r') as f_in:
                    lines = f_in.readlines()
                    for line in lines:
                        df = line.strip().split(',')
                        if float(df[2]) < 116.25 or float(df[2]) > 116.5 or float(df[1]) < 39.8 or float(df[1]) > 40.1:
                            continue
                        out.append([df[2], df[1]])
                n = len(out)
                for i in range(n//100 + 1):
                    if n - 100 * i < 20:
                        break
                    f.write("\"%s\"," % times)
                    f.write("\"[")
                    for j in range(100):
                        if 100 * i + j >= n:
                            break
                        if j == 0:
                            f.write("[%s,%s]" % (out[100 * i + j][0], out[100 * i + j][1]))
                        else:
                            f.write(",[%s,%s]" % (out[100 * i + j][0], out[100 * i + j][1]))
                    f.write("]\"\n")
            # exit(0)


                
                

if __name__ == '__main__':
    # add_label()
    # split_traj()
    # summary()
    to_csv()
