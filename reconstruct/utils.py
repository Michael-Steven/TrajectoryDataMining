import pandas as pd
from easydict import EasyDict
import numpy as np
from tqdm import tqdm


def softmax(arr):
    exp_arr = np.exp(-arr)
    return exp_arr / np.sum(exp_arr)


def make_dist_index_dict(w2i_dict, dist_dict):
    dist_index_dict = dict()
    for gid1, gid2 in dist_dict:
        dist_index_dict[w2i_dict[gid1], w2i_dict[gid2]] = dist_dict[gid1, gid2]
        dist_index_dict[w2i_dict[gid2], w2i_dict[gid1]] = dist_dict[gid1, gid2]
    return dist_index_dict


def load_dist_dict(filepath):
    dist_dict = dict()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            dist_dict[line[0], line[1]] = float(line[-1])
            dist_dict[line[1], line[0]] = float(line[-1])
    return dist_dict


def load_dist_dict_geolife(filepath, w2i_dict):
    print("begin load dist dict")
    dist_dict = dict()
    region_distance = np.load(filepath)
    w, h = region_distance.shape
    res = []
    for key in w2i_dict:
        if key in ["<m>", "*", "pad"]:
            continue
        for key2 in w2i_dict:
            if key2 in ["<m>", "*", "pad"]:
                continue
            res.append([int(key), int(key2), region_distance[int(key)][int(key2)]])
    with open("/home1/shanyanbo/design/reconstruct/Dataset/GeoLife/vocabs_dist.txt", "w") as f:
        for a, b, dis in tqdm(res):
            f.write("%d\t%d\t%f\n" % (a, b, dis))
    print("end load dist dict")
    return dist_dict


def load_w2ifile(filepath):
    w2i_dict = EasyDict()
    w2i_dict["pad"] = 0
    w2i_dict["*"] = 1
    w2i_dict["<m>"] = 2
    index = 3
    with open(filepath, "r") as f:
        for line in f:
            if line.strip() not in w2i_dict:
                w2i_dict[line.strip()] = index
                index += 1
    i2w_dict = dict([(item[1], item[0]) for item in w2i_dict.items()])
    return w2i_dict, i2w_dict


def load_trajfile(filepath):
    df = pd.read_csv(filepath, header=None, index_col=False, names=["user", "time", "traj"], sep="\t")
    data_dict = EasyDict()
    data_dict["user"] = df["user"].values
    data_dict["time"] = df["time"].values
    data_dict["traj"] = [traj.split() for traj in df["traj"].values.tolist()]  # list of list.
    return data_dict


def convert_traj_to_index(w2i_dict, data_dict):
    traj_index = []
    for traj in data_dict.traj:
        traj_index.append([w2i_dict[w] for w in traj])
    data_dict["traj_index"] = traj_index
    return data_dict


def split_traj_index(data_dict, w2i_dict):
    history_trajs = []
    history_trajs_masks = []
    current_trajs = []
    masked_current_trajs = []
    mask_indexes = []
    masked_true_traj_points = []
    for traj_index in data_dict.traj_index:
        splited_traj_index = [traj_index[i * 48:(i + 1) * 48] for i in range(int(len(traj_index) / 48))]
        hist_masks = []
        for his_traj in splited_traj_index[:-2]:
            if w2i_dict["pad"] in his_traj:
                hist_masks.append(0)
            else:
                hist_masks.append(1)
        history_trajs_masks.append(hist_masks)
        history_trajs.append(splited_traj_index[:-2])
        current_trajs.append(splited_traj_index[-2])
        masked_current_trajs.append(splited_traj_index[-1])
        mask_index = []
        masked_true_traj_point = []
        for i, point in enumerate(splited_traj_index[-2]):
            if splited_traj_index[-2][i] != splited_traj_index[-1][i]:
                mask_index.append(i)
                masked_true_traj_point.append(splited_traj_index[-2][i])
        mask_indexes.append(mask_index)
        masked_true_traj_points.append(masked_true_traj_point)
    data_dict["history_trajs"] = history_trajs
    data_dict["history_trajs_masks"] = history_trajs_masks
    data_dict["current_trajs"] = current_trajs
    data_dict["masked_current_trajs"] = masked_current_trajs
    data_dict["mask_indexes"] = mask_indexes
    data_dict["masked_true_traj_points"] = masked_true_traj_points
    return data_dict


def preprocess(filepath, w2i_dict):
    data_dict = load_trajfile(filepath)
    data_dict = convert_traj_to_index(w2i_dict, data_dict)
    data_dict = split_traj_index(data_dict, w2i_dict)

    return data_dict
