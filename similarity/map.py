import pickle
import torch
import math
from tqdm import tqdm
from models import LSTMSimCLR
from torch.utils.data import DataLoader
import folium
import numpy as np


x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 偏心率平方


def gcj02_to_bd09(lng, lat):
    """
    火星坐标系(GCJ-02)转百度坐标系(BD-09)
    谷歌、高德——>百度
    :param lng:火星坐标经度
    :param lat:火星坐标纬度
    :return:
    """
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * x_pi)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * x_pi)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return [bd_lng, bd_lat]


def bd09_to_gcj02(bd_lon, bd_lat):
    """
    百度坐标系(BD-09)转火星坐标系(GCJ-02)
    百度——>谷歌、高德
    :param bd_lat:百度坐标纬度
    :param bd_lon:百度坐标经度
    :return:转换后的坐标列表形式
    """
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
    gg_lng = z * math.cos(theta)
    gg_lat = z * math.sin(theta)
    return [gg_lng, gg_lat]


def wgs84_to_gcj02(lng, lat):
    """
    WGS84转GCJ02(火星坐标系)
    :param lng:WGS84坐标系的经度
    :param lat:WGS84坐标系的纬度
    :return:
    """
    if out_of_china(lng, lat):  # 判断是否在国内
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [mglng, mglat]


def gcj02_to_wgs84(lng, lat):
    """
    GCJ02(火星坐标系)转GPS84
    :param lng:火星坐标系的经度
    :param lat:火星坐标系纬度
    :return:
    """
    if out_of_china(lng, lat):
        return [lng, lat]
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def bd09_to_wgs84(bd_lon, bd_lat):
    lon, lat = bd09_to_gcj02(bd_lon, bd_lat)
    return gcj02_to_wgs84(lon, lat)


def wgs84_to_bd09(lon, lat):
    lon, lat = wgs84_to_gcj02(lon, lat)
    return gcj02_to_bd09(lon, lat)


def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def out_of_china(lng, lat):
    """
    判断是否在国内，不在国内不做偏移
    :param lng:
    :param lat:
    :return:
    """
    return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)


class Dataset:
    def __init__(self, filepath, max_len):
        self.filepath = filepath
        self.max_len = max_len
        self.trajs, self.trajs_len = self.read_data(filepath, max_len)

    def read_data(self, filepath, max_len):
        trajs = []
        trajs_len = []
        with open(filepath) as f:
            for traj in f:
                traj = [int(point) for point in traj.strip().split(" ")]
                if len(traj) > max_len:
                    traj = traj[:max_len]
                    traj_len = max_len
                else:
                    traj_len = len(traj)
                    traj = traj + [0] * (max_len - traj_len)
                trajs_len.append(traj_len)
                trajs.append(traj)
        return trajs, trajs_len

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, i):
        return torch.tensor(self.trajs[i], dtype=torch.long), \
            torch.tensor(self.trajs_len[i], dtype=torch.long)

max_len = 100
hidden_size = 128
batch_size = 128
bidirectional = 0
n_layers = 1
freeze = 1
infer = 1
max_vocab_size = 18924
device = "cuda:5"
features_type = "encoder"
test_data_path = "./experiment/geolife/exp1/exp1-trj.t"
test_model_path = "/home1/shanyanbo/design/similarity/log/geolife_tcn/checkpoint_0099_hiddensize_128_batchsize_128_bidirectional_0_nlayers_1_freeze_1.pth.tar"

if infer == 1:
    checkpoint = torch.load(test_model_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    model = LSTMSimCLR(max_vocab_size, hidden_size, bidirectional, n_layers)
    model.load_state_dict(state_dict)
    model.to(device)

    test_dataset = Dataset(test_data_path, max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    features_list = []
    # grids_list = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_dataloader):
            trajs, trajs_len = [b.to(device) for b in batch]
            # batch_size, hidden_size * n_direction
            features = model.encode_by_encoder(trajs, trajs_len)
            features_list.append(features.cpu())
        features_list = torch.cat(features_list, dim=0).numpy()

    # with open(test_data_path) as f:
    #     for traj in f:
    #         traj = [int(point) for point in traj.strip().split(" ")]
    #         if len(traj) > max_len:
    #             traj = traj[:max_len]
    #             traj_len = max_len
    #         grids_list.append(traj)

    pickle.dump(features_list, open("./outputs_encoder/geolife_features.pkl", "wb"))
    # pickle.dump(grids_list, open("./outputs_encoder/geolife_grids.pkl", "wb"))
    trajs_list = []
    with open("/home1/shanyanbo/design/similarity/experiment/geolife/exp1/exp1-gps.t", "r") as f:
        lines = f.readlines()
        for line in lines:
            traj = []
            nums = line.split()
            for i in range(0, len(nums), 2):
                lng, lat = float(nums[i]), float(nums[i+1])
                traj.append([lat, lng])
            trajs_list.append(traj)
    print(len(trajs_list))
    pickle.dump(trajs_list, open("./outputs_encoder/geolife_trajs.pkl", "wb"))
else:
    features_list = pickle.load(open("./outputs_encoder/geolife_features.pkl", "rb"))
    trajs_list = pickle.load(open("./outputs_encoder/geolife_trajs.pkl", "rb"))
# print(features_list.shape)
# print(len(trajs_list))
# print(features_list[0])
# print(trajs_list[0])

pairs = [[6, 8],[26, 1026],[112, 136],[151, 1151],[152, 1152],[155, 162],[171, 1171],[980, 546],
         [992, 1992],[416, 1416],[819, 1819],[122, 987],[808, 981],[200, 201],[119, 968],[110, 317],[101, 413],
         [2, 527],[456, 574]]
# map = folium.Map(location=[40.05, 116.4], zoom_start=13, max_zoom=25, tiles="cartodb positron")
# for i in range(1000):
#     traj = trajs_list[i]
#     folium.PolyLine(locations=traj, color='blue', weight=2).add_to(map)
#     folium.Marker(location=traj[0], popup=str(i), icon=folium.Icon(color='green')).add_to(map)
#     # folium.Marker(location=traj[-1], popup='End', icon=folium.Icon(color='red', icon='info-sign')).add_to(map)
#     traj = trajs_list[i+1000]
#     folium.PolyLine(locations=traj, color='red', weight=2).add_to(map)
# map.save("track/test.html")
i = 0
for x, y in pairs:
    map = folium.Map(location=[40.05, 116.4], zoom_start=13, max_zoom=25, tiles="cartodb positron")
    traj_x = trajs_list[x]
    traj_y = trajs_list[y]
    ans = np.linalg.norm(features_list[x] - features_list[y])
    folium.PolyLine(locations=traj_x, color='blue', weight=2).add_to(map)
    folium.Marker(location=traj_x[0], popup=str(ans), icon=folium.Icon(color='green'), weight=1).add_to(map)
    folium.Marker(location=traj_x[-1], popup='End', icon=folium.Icon(color='red', icon='info-sign')).add_to(map)
    folium.PolyLine(locations=traj_y, color='red', weight=2).add_to(map)
    folium.Marker(location=traj_y[0], popup=str(ans), icon=folium.Icon(color='green')).add_to(map)
    folium.Marker(location=traj_y[-1], popup='End', icon=folium.Icon(color='red', icon='info-sign')).add_to(map)
    map.save("track/test" + str(i) + ".html")
    print(i, ans)
    i += 1
