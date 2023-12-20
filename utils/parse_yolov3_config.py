def parse_cfg(cfg_file):
    """
    提取cfg文件中定义的网络模块
    :param cfg_file: cfg文件
    :return: 一系列的模块。每一个模块表示一个搭建好的神经网络。存储在一个字典列表中
    """
    file = open(cfg_file, 'r', encoding='UTF-8')
    lines = file.read().split('\n')  # 储存一个列表
    lines = [x for x in lines if len(x) > 0]  # 去掉空行
    lines = [x for x in lines if x[0] != '#']  # 去掉注释
    lines = [x.rstrip().lstrip() for x in lines]  # 去掉空格
    # 循环遍历列表，获取网络模块
    block = {}  # 存储当前的网络模块
    blocks = []
    for line in lines:
        if line[0] == "[":  # 这个标志着这是一个新的模块
            if len(block) != 0:  # 如果当前的模块不为空，则表示这是上一个模块
                blocks.append(block)  # 将上一个模块放入总的模块列表
                block = {}  # 清空字典
            block["type"] = line[1:-1].rstrip()  # 模块的名字
        else:
            key, value = line.split("=")  # 将值和键从字符串中提取出来
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


def parse_data_config(path):
    """返回一个字典，该字典为数据配置文件的信息"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r', encoding='UTF-8') as fp:
        lines = fp.readlines()
    for line in lines:  # 一行一行的地解析配置文件
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


if __name__ == '__main__':
    cfg = './yolov3.cfg'
    print(parse_cfg(cfg))