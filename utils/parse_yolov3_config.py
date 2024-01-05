from config import cfg

def parse_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            key = key.rstrip()
            value = value.strip()
            module_defs[-1][key] = value
            if module_defs[-1]['type'] == 'yolo' and key == 'classes':
                module_defs[-2]['filters'] = (cfg.YOLO.CLASSNUM + 5) * 3
                module_defs[-1][key] = cfg.YOLO.CLASSNUM
    return module_defs


if __name__ == '__main__':
    model_cfg = '../model/model-config/yolov3.cfg'
    a = parse_config(model_cfg)
    print(a)
