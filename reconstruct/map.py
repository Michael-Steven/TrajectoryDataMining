import pickle
from pathlib import Path
import folium
from utils import load_trajfile, convert_traj_to_index, split_traj_index
import numpy as np
import torch
from transformers import HfArgumentParser, set_seed
from config import GeolifConfig
from model import LSTM_Attention_Model
import json
import math
from tqdm import tqdm


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


def construct_graph(inputs):
    items, n_node, A, alias_inputs = [], [], [], []
    for u_input in inputs:
        n_node.append(len(np.unique(u_input)))
    max_n_node = np.max(n_node)
    for u_input in inputs:
        node = np.unique(u_input)
        items.append(node.tolist() + (max_n_node - len(node)) * [0])
        u_A = np.zeros((max_n_node, max_n_node))
        for i in range(len(u_input) - 1):
            if u_input[i + 1] == 0:
                break
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        A.append(u_A)
        alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
    return alias_inputs, A, items


def get_batch(w2i_dict, i):
    data_dict = load_trajfile("/home1/shanyanbo/design/reconstruct/Dataset/GeoLife/pos.test.txt")
    data_dict = convert_traj_to_index(w2i_dict, data_dict)
    data_dict = split_traj_index(data_dict, w2i_dict)

    history_trajs = np.array(data_dict.history_trajs)
    history_trajs_masks = np.array(data_dict.history_trajs_masks)
    masked_current_trajs = np.array(data_dict.masked_current_trajs)
    masked_indexes = np.array(data_dict.mask_indexes)
    masked_true_traj_points = np.array(data_dict.masked_true_traj_points)

    history_trajs, history_trajs_masks, masked_current_trajs, masked_indexes, targets = \
                history_trajs[i], history_trajs_masks[i], masked_current_trajs[i], masked_indexes[i], masked_true_traj_points[i]
    # reshape history trajs: batch_size * history_length, seq_length
    reshaped_history_trajs = history_trajs.reshape(-1, history_trajs.shape[-1])
    # print(reshaped_history_trajs.shape)

    reshaped_history_alias_inputs, reshaped_history_A, reshaped_history_items = construct_graph(reshaped_history_trajs)
    masked_current_alias_inputs, masked_current_A, masked_current_items = construct_graph(masked_current_trajs)
    # convert to tensor
    history_trajs_masks = torch.tensor(history_trajs_masks, dtype=torch.long)
    reshaped_history_alias_inputs = torch.tensor(reshaped_history_alias_inputs, dtype=torch.long)
    reshaped_history_A = torch.tensor(reshaped_history_A, dtype=torch.float)
    reshaped_history_items = torch.tensor(reshaped_history_items, dtype=torch.long)
    masked_current_alias_inputs = torch.tensor(masked_current_alias_inputs, dtype=torch.long)
    masked_current_A = torch.tensor(masked_current_A, dtype=torch.float)
    masked_current_items = torch.tensor(masked_current_items, dtype=torch.long)
    # batch_size, mask_num
    masked_indexes = torch.tensor(masked_indexes, dtype=torch.long)
    # batch_size * mask_num
    targets = torch.tensor(targets, dtype=torch.long).view(-1)

    batch = [history_trajs_masks, reshaped_history_alias_inputs, reshaped_history_A, reshaped_history_items, masked_current_alias_inputs,
                        masked_current_A, masked_current_items, masked_indexes, targets]
    return batch


if __name__ == "__main__":
    parser = HfArgumentParser(GeolifConfig)
    config = parser.parse_args_into_dataclasses()[0]
    set_seed(config.seed)
    device = torch.device(config.device)

    emb_w2i_dict_path = Path("/home1/shanyanbo/design/reconstruct/exp/geolife/dataset_geolife_hiddensize_128_nheads_12_selfheads_4_distloss_True_dropout_0.3_alpha_0.25_bs_50_lr_0.0010_nopos.emb")
    embedding_matrix, w2i_dict, i2w_dict = pickle.load(emb_w2i_dict_path.open("rb"))

    # model definition
    model = LSTM_Attention_Model(config, max(w2i_dict.values())+1, 4)
    model.to(device)
    model.load_state_dict(torch.load("/home1/shanyanbo/design/reconstruct/exp/geolife/dataset_geolife_hiddensize_128_nheads_12_selfheads_4_distloss_True_dropout_0.3_alpha_0.25_bs_50_lr_0.0010_nopos.chkpt"))
    model.eval()

    ind_list = [2, 43, 44, 107, 120, 124, 144, 157, 159, 160, 190, 192, 224, 238, 250, 267, 285, 306, 312, 316, 334, 351, 377, 388, 431, 449, 462, 477, 175, 252, 366, 412, 413, 446, 452, 476, 503]
    # ind_list = [120]

    for batch_num in tqdm(ind_list):
        ori_traj = []
        reconstruct_traj = []
        batch = get_batch(w2i_dict, [batch_num])
        batch = [d.to(device) for d in batch]
        inputs, targets = batch[:-1], batch[-1]
        targets = targets - 3
        # batch_size * mask_num, n
        scores = model(*inputs)
        masked_current_alias_inputs = inputs[4][0].detach().cpu().numpy().tolist()
        masked_current_items = inputs[6][0].detach().cpu().numpy().tolist()
        masked_indexes = inputs[7][0].detach().cpu().numpy().tolist()

        tmp = [i2w_dict[masked_current_items[masked_current_alias_inputs[i]]] for i in range(len(masked_current_alias_inputs))]
        targets = targets.detach().cpu().numpy().tolist()
        for i in range(len(masked_indexes)):
            tmp[masked_indexes[i]] = i2w_dict[targets[i] + 3]
        ori_traj.extend(tmp)

        max_list = torch.argmax(scores, dim=-1).detach().cpu().numpy().tolist()
        tmp = [i2w_dict[masked_current_items[masked_current_alias_inputs[i]]] for i in range(len(masked_current_alias_inputs))]
        for i in range(len(masked_indexes)):
            tmp[masked_indexes[i]] = i2w_dict[max_list[i] + 3]
        reconstruct_traj.extend(tmp)
        # print(ori_traj)
        # print(reconstruct_traj)

        # 区域id,边界和中心
        regions = json.load(open("/home1/shanyanbo/design/reconstruct/Dataset/region_data/region_beijing_countfilter10.json"))
        center = regions['center']
        boundary = regions['boundary']
        ori = []
        ori_region = []
        for region in ori_traj:
            if region != '*':
                lng, lat = gcj02_to_wgs84(center[int(region)][0], center[int(region)][1])
                ori.append([lat, lng])
                tmp = []
                for x, y in boundary[int(region)]:
                    lng, lat = gcj02_to_wgs84(x, y)
                    tmp.append([lat, lng])
                ori_region.append(tmp)
        res = []
        res_region = []
        res_mask = []
        for i in range(len(reconstruct_traj)):
            region = reconstruct_traj[i]
            if region != '*':
                lng, lat = gcj02_to_wgs84(center[int(region)][0], center[int(region)][1])
                res.append([lat, lng])
                if i in masked_indexes:
                    res_mask.append([lat, lng])
                tmp = []
                for x, y in boundary[int(region)]:
                    lng, lat = gcj02_to_wgs84(x, y)
                    tmp.append([lat, lng])
                res_region.append(tmp)

        # 创建地图对象，设置中心点和初始缩放级别
        map = folium.Map(location=[39.95, 116.4], zoom_start=13, tiles="cartodb positron")
        # 添加轨迹
        for b in ori_region:
            folium.PolyLine(locations=b, color='blue', weight=2, dash_array='5').add_to(map)
        for b in res_region:
            folium.PolyLine(locations=b, color='red', weight=2, dash_array='5').add_to(map)
        folium.PolyLine(locations=ori, color='blue', weight=3).add_to(map)
        folium.PolyLine(locations=res, color='red', weight=3).add_to(map)
        for point in res_mask:
            folium.CircleMarker(location=point, color='green', radius=4, fill=True, fill_opacity=1).add_to(map)
        folium.Marker(location=res[0], popup='Start', icon=folium.Icon(color='green')).add_to(map)
        folium.Marker(location=res[-1], popup='End', icon=folium.Icon(color='red', icon='info-sign')).add_to(map)

        # 保存地图为 HTML 文件
        map.save("track/track_" + str(batch_num) + ".html")
