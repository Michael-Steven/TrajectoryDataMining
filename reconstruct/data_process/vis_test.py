import os
import matplotlib.pyplot as plt


def trajPlt():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    lat = []  # 总维度
    lng = []  # 总经度
    lat_spl = []
    lng_spl = []
    # 总路径
    path = os.getcwd() + "/data" + "/Geolife" + "/Data" + "/001" + "/Trajectory"
    print(path)
    user_path = os.scandir(path)

    plts = os.listdir(path)

    # 每一个文件的绝对路径
    for item in user_path:
        path_item = path + "/" + item.name
        with open(path_item, 'r+') as fp:
            lat_tmp = []
            lng_tmp = []
            for item in fp.readlines()[6::10]:
                item_list = item.split(',')
                lat.append(item_list[0])
                lng.append(item_list[1])
                lat_tmp.append(item_list[0])
                lng_tmp.append(item_list[1])
            lat_tmp_plt = [float(x) for x in lat_tmp]
            lng_tmp_plt = [float(x) for x in lng_tmp]
            lat_spl.append(lat_tmp_plt)
            lng_spl.append(lng_tmp_plt)
            lat_tmp.clear()
            lng_tmp.clear()

    print(lat_spl)
    print(lng_spl)

    lat_new = [float(x) for x in lat]
    lng_new = [float(x) for x in lng]
    plt.ylim((min(lat_new), max(lat_new)))
    plt.xlim((min(lng_new), max(lng_new)))

    plt.title("003轨迹测试")
    plt.axis('equal')
    plt.xlabel("经度")  # 定义x坐标轴名称
    plt.ylabel("维度")  # 定义y坐标轴名称
    for x, y in zip(lng_spl, lat_spl):
        plt.plot(x, y, marker='o', linestyle=':')
    # plt.plot(lng_new, lat_new, '.')  # 绘图
    plt.show()  # 展示


if __name__ == "__main__":
    trajPlt()
