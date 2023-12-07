# %%
# jupyter中使用hyperparams需要添加
import datetime
import pandas as pd
import os
import sys
import json
sys.argv = ['run.py']

# %% [markdown]
# 根据经纬度映射出区域号.
# - xlim:经度范围
# - ylim:纬度范围
#
# 北京：
# - lng_ld = 116.1
# - lat_ld = 39.7
# - lng_max = 116.7
# - lat_max = 40.2

# %%
lng_ld = 116.1
lat_ld = 39.7
lng_max = 116.7
lat_max = 40.2


def get_gridID(longitude, latitude):
    # xlim = [116.1, 116.7]
    # ylim = [39.7, 40.2]

    grid_length = 0.0002
    grid_width = 0.0002
    # regionID = (xlim[1] - xlim[0]) // grid_length * (latitude - ylim[0]) // grid_width + (
    #         longitude - xlim[0]) // grid_length + 1
    gridID = str(int((longitude - lng_ld) / 0.0002)) + "-" + str(int((latitude - lat_ld) / 0.0002))
    # print(gridID)
    # if gridID == '26498--42657':
    #     print(longitude, latitude)
    return gridID


print(get_gridID(116.2345, 39.8744))


# %% [markdown]
# 轨迹过滤：过滤掉timeslots小于m的轨迹，过滤掉总条数少于5条的轨迹。

# %%
file_list = sorted(os.listdir('../Geolife/Data'))
file_num = len(file_list)
# file_list = file_list[1:]
print(file_list)
# print(len(file_list))


for user_id in file_list:
    u_path = os.getcwd() + "../Geolife" + "/Data/" + user_id + "/Trajectory"


# path = os.getcwd() + "/data" + "/Geolife" + "/Data" + "/001" + "/Trajectory"
# print(path)
# user_path = os.scandir(path)
# print(user_path)

# %%
output_path = './Results/size%d' % (1)
if not os.path.exists(output_path):
    # 目录不存在，进行创建操作
    os.makedirs(output_path)


# %%
time0 = datetime.datetime.now()
start_time = datetime.datetime.now()
result_tmp = []
data_tmp = pd.DataFrame()
data_all = pd.DataFrame()
# data_all.columns=['UID','timestamp','longitude','latitude']
for filename in file_list:
    path = ".." + "/Geolife" + "/Data/" + filename + "/Trajectory"
    # user_path: one user's all trajectories
    user_path = os.scandir(path)
    for item in user_path:
        # path_item: one of the trajactories
        path_item = path + "/" + item.name
        with open(path_item, 'r+') as fp:
            # every item is a point
            # lat, lnt, 0, alt, date, date(日期), time（时间）
            # 39.999383,116.326916,0,134,39915.3152083333,2009-04-12,07:33:54
            for item in fp.readlines()[6::10]:
                item_list = item.strip().split(',')

                uid = filename
                lat = item_list[0]
                lnt = item_list[1]
                date = item_list[5]
                timestamp = item_list[6]

                temp = []
                temp.append(uid)
                temp.append(lat)
                temp.append(lnt)
                temp.append(date)
                temp.append(timestamp)
                result_tmp.append(temp)
end_time = datetime.datetime.now()
print(len(result_tmp))
print("Loading Time:", str((end_time - start_time))[0:7])

# %%
data_all = pd.DataFrame(result_tmp, columns=['uid', 'latitude', 'longitude', 'date', 'time']).dropna(axis=0)

tmp = data_all.copy(deep=True)
tmp = tmp.astype({'latitude': float, 'longitude': float})
# tmp = tmp['longitude'].astype('float')
print(tmp.shape)
print("dropping lat lng...")
tmp = tmp.drop(index=tmp[tmp['latitude'] < lat_ld].index)
tmp = tmp.drop(index=tmp[tmp['latitude'] > lat_max].index)
tmp = tmp.drop(index=tmp[tmp['longitude'] < lng_ld].index)
tmp = tmp.drop(index=tmp[tmp['longitude'] > lng_max].index)
print(tmp.shape)
# 经纬度
# UID
tmp['uid'] = tmp['uid'].apply(lambda x: int(x))
# 获取日期
# 获取具体时间（24小时制）
# 获取时间编号（48个半小时中的第几个）
tmp['timeNo'] = tmp['time'].apply(lambda x: int(x[0:2]) * 2 + int(x[3:5]) // 30)
# 获得区域编号
tmp['gridID'] = tmp.apply(lambda x: get_gridID(float(x['longitude']), float(x['latitude'])), axis=1)
# tmp

# %% [markdown]
# ```
# ['000', '40.000017', '116.327479', '2009-04-12', '07:33:03']
# ['000', '39.999383', '116.326916', '2009-04-12', '07:33:54']
# ['000', '39.998139', '116.327237', '2009-04-12', '07:34:44']
# ['000', '39.996225', '116.32658', '2009-04-12', '07:35:39']
# ['000', '39.995637', '116.326712', '2009-04-12', '07:36:24']
# ```

# %%
# 给区域重新编号
print("reindex regionID...")

grid2region = json.load(open('/home1/shanyanbo/PeriodicMove-main/Dataset/region_data/region_beijing_grid2region_REID_no_less10.json'))  # GRID2REGION lookup


def transgrid2region(grid):
    count = 0
    while (1):
        if grid == '0-0':
            return -1
        if grid in grid2region:
            return grid2region[grid]
        else:
            # print(grid)
            grid_lng, grid_lat = int(grid.split('-')[0]), int(grid.split('-')[1])
            if count % 2 == 0:
                if grid_lat >= 1:
                    grid_lat -= 1
            else:
                if grid_lng >= 1:
                    grid_lng -= 1
            grid = str(grid_lng) + '-' + str(grid_lat)
            count += 1


# keys = []
# keys += set(tmp['gridID'])
# # keys = [i for n, i in enumerate(keys) if i not in keys[:n]]
# print("After loading the data files, region numbers: ", len(keys))
# values = [i for i in range(1, len(keys) + 1)]
# tmp_dict = dict(zip(keys, values))
# tmp['regionID'] = tmp.apply(lambda x: tmp_dict[x['gridID']], axis=1)
tmp['regionID'] = tmp.apply(lambda x: transgrid2region(x['gridID']), axis=1)
tmp = tmp.drop(index=tmp[tmp['regionID'] == -1].index)
print(tmp.shape)
tmp.head(5)


# %%
# 转换成AttnMove的数据格式：{uid, date, trajectory}
print("generate raw_pos.csv...")
tmp3 = tmp[['uid', 'date', 'regionID', 'timeNo']].copy(deep=True)

# aggregate the regionID for every time intervel(30 mins)
# tmp3 = tmp3.groupby(['uid', 'date', 'timeNo']).apply(lambda x: x[:])#.agg(lambda x: x.value_counts().index[0]).reset_index()
tmp3 = tmp3.groupby(['uid', 'date', 'timeNo']).agg(lambda x: x.value_counts().index[0]).reset_index()
tmp3.to_csv('./Results/tmp3.txt', sep='\t', index=False)
tmp3

# %%
# 获得user_list和date_list
tmp3 = pd.read_csv('Results/tmp3.txt', sep="\t").groupby(['uid', 'date']).apply(lambda x: x[:])
user_list = tmp3['uid'].drop_duplicates().values
date_list = tmp3['date'].drop_duplicates().values
print(user_list)
print(date_list)

# %%
# result_final是符合AttnMove数据格式的dataframe
result_final = pd.DataFrame()
no_use = 0
result_list = []
for uid in user_list:
    for date in date_list:
        try:
            tmp4 = pd.DataFrame(tmp3.loc[(uid, date), ['timeNo', 'regionID']].values.T)
            tmp4 = tmp4.set_axis(tmp4.iloc[0], axis=1).drop(index=0)
            tmp4.index = pd.MultiIndex.from_product([[uid], [date]])
            result_list.append(tmp4)
        except:
            # print("No record for UID=%d, date=%s" % (uid, date))
            no_use = no_use + 1
            continue
result_final = pd.concat(result_list)

# 没有出现的时段列也补上相应的列
seq_len = 48
column_names = result_final.columns
for index in range(0, seq_len):
    if index not in column_names:
        result_final.insert(index, index, None)

result_final = result_final.fillna(-1)
result_final = result_final.astype('int')
# result_final = result_final.fillna('*')
result_final = result_final.replace(to_replace=-1, value='*')
result_final = result_final.sort_index(axis=1)
print(result_final.shape)

# %%
# 运行一次太慢了，因此跑完之后存下来方便之后直接用
result_final.to_csv(output_path + '/raw_pos.csv', index=True, header=True)
df_tmp = pd.read_csv(output_path + '/raw_pos.csv', index_col=0)
df_tmp.to_csv(output_path + '/raw_pos.txt', sep='\t', index=True, header=True)

# %%
