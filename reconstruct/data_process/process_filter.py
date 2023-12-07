# %%
import pandas as pd
from datetime import datetime
import os
import argparse

# %%
# 读取raw_data
output_path = './Results/size%d' % (1)
if not os.path.exists(output_path):
    # 目录不存在，进行创建操作
    os.makedirs(output_path)

startTime = datetime.now()
traj_undropped = pd.read_csv(output_path + '/raw_pos.txt', sep="\t")
# 删除日期未知的轨迹
traj_undropped.drop(traj_undropped[traj_undropped['Unnamed: 0'] == '*'].index, inplace=True)
traj_undropped.shape

# %%
# 删除采样点少于12个的轨迹
columns = traj_undropped.columns
traj_dropped = pd.DataFrame(columns=columns)
traj_dropped_list = []
for i in range(len(traj_undropped)):
    if '*' in traj_undropped.iloc[i].value_counts():
        count = traj_undropped.iloc[i].value_counts()['*']
    else:
        count = 0
    if count <= 36:
        tmp = pd.DataFrame(traj_undropped.iloc[i]).T
        # tmp.set_axis(tmp.iloc[0], axis=1).drop(index=0)
        traj_dropped_list.append(tmp)
        # traj_dropped = traj_dropped.append(traj_undropped.iloc[i])
        # break
traj_dropped = pd.concat(traj_dropped_list)
# traj_undropped.index = list(traj_undropped.iloc[:,0])
print(traj_dropped.shape)


# %%
# 删除轨迹少于5条的用户
traj_drop_uid = pd.DataFrame(columns=columns)
# 每个用户的天数
traj_dropped.index = list(traj_dropped.iloc[:, 0])
date_counts = traj_dropped.index.value_counts()
traj_drop_uid_list = []
for uid in date_counts.index:
    if date_counts[uid] > 5:
        tmp = traj_dropped.loc[uid]
        traj_drop_uid_list.append(tmp)
        # traj_drop_uid = traj_drop_uid.append(traj_dropped.loc[uid])
traj_drop_uid = pd.concat(traj_drop_uid_list)
print(traj_drop_uid.shape)

# %%
endTime = datetime.now()
print("Dropping time:", str(endTime - startTime)[0:7])

# %%
# 删除小数点
traj_drop_uid.replace('*', -1, inplace=True)
traj_drop_uid.iloc[:, 2:] = traj_drop_uid.iloc[:, 2:].astype(float)
traj_drop_uid.iloc[:, 2:] = traj_drop_uid.iloc[:, 2:].astype(int)
traj_drop_uid.replace(-1, '*', inplace=True)
print(set(traj_drop_uid['Unnamed: 0']))
print(len(set(traj_drop_uid['Unnamed: 0'])))

# %%
# 给用户重新编号
# keys = set(traj_drop_uid.index)
# values = [i for i in range(1, len(keys) + 1)]
# tmp_dict = dict(zip(keys, values))

# traj_drop_uid['Unnamed: 0'] = traj_drop_uid['Unnamed: 0'].replace(tmp_dict)
traj_drop_uid.sort_values(by=["Unnamed: 0", "Unnamed: 1"], inplace=True, ascending=True)
# print(traj_drop_uid.shape)
# print(set(traj_drop_uid['Unnamed: 0']))
# print(len(set(traj_drop_uid['Unnamed: 0'])))


# %%
print("generate dropped_pos.txt...")
traj_drop_uid.to_csv(output_path + '/dropped_pos.txt', sep='\t', index=False, header=1)
print("generatation completed.")

# %%
