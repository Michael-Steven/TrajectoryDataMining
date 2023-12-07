# %%
import pandas as pd
from datetime import datetime
import argparse
import os

# %%
output_path = './Results/size%d' % (1)
if not os.path.exists(output_path):
    # 目录不存在，进行创建操作
    os.makedirs(output_path)

startTime = datetime.now()
print("converting delimiters...")
dropped_pos = pd.read_csv(output_path + '/dropped_pos.txt', sep="\t")
dropped_pos.head(5)

# %%
# 将日期转化为连续数字
dropped_pos["Unnamed: 1"] = dropped_pos["Unnamed: 1"].str.replace("-", "")
dropped_pos.sort_values(by=["Unnamed: 0", "Unnamed: 1"], inplace=True, ascending=True)
dropped_pos = dropped_pos.reset_index(drop=True)
dropped_pos.head(5)

# %%
# 把采样点的分隔符从\t换成' '
dropped_pos.iloc[:, 2:].to_csv(output_path + '/position.txt', sep=' ', index=False)
dropped_pos.iloc[:, 0:2].to_csv(output_path + '/uid_and_date.txt', sep='\t', index=False)

# %%
# 合并txt
file1path = output_path + '/uid_and_date.txt'
file2path = output_path + '/position.txt'

file_1 = open(file1path, 'r')
file_2 = open(file2path, 'r')

list1 = []
for line in file_1.readlines():
    ss = line.strip()
    list1.append(ss)
file_1.close()

list2 = []
for line in file_2.readlines():
    ss = line.strip()
    list2.append(ss)
file_2.close()

file_new = open("./Results/result%d.txt" % (1), 'w')
for i in range(1, len(list1)):
    sline = list1[i] + '\t' + list2[i]
    file_new.write(sline + '\n')
file_new.close()

endTime = datetime.now()
print("converting completed.")
print("Total time:", str((endTime - startTime))[5:11] + 's.')
