from easydict import EasyDict as edict

__C = edict()
# RTTS dataset
__C.RTTS = edict()
__C.RTTS.path = r'./dataset/RTTS_test.txt'
__C.RTTS.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.RTTS.BATCH_SIZE = 1
__C.times = 'exp03'
__C.MODEL = edict()
# fog settings
__C.MODEL.filters = True
__C.MODEL.model_def = r"./model/model_config/yolov3.cfg"
__C.MODEL.weight = './checkpoints/exp03/FY_ckpt_10.pth'
# YOLO options
__C.YOLO = edict()
# Set the class name
__C.YOLO.CLASSES = './data/classes/vocfog.names'
__C.YOLO.CLASSNUM = 5
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.pretrained_weights = r'./model/model_config/yolov3.weights'
__C.YOLO.ANCHORS = "./data/anchors/coco_anchors.txt"
# Train options
__C.TRAIN = edict()
__C.TRAIN.ANNOT_PATH = './data/data_fog/voc_norm_train.txt'
__C.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.BATCH_SIZE = 1
__C.TRAIN.DATA_AUG = True
__C.TRAIN.vocfog_traindata_dir = './data/data_fog/train/JPEGImages/'
__C.TRAIN.scratch = False
__C.TRAIN.n_cpu = 16
__C.TRAIN.epochs = 80
# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = './data/data_fog/voc_norm_test.txt'
__C.TEST.INPUT_SIZE = 416
__C.TEST.BATCH_SIZE = 4
__C.TEST.DATA_AUG = False
__C.TEST.vocfog_valdata_dir = './data/data_fog/test/JPEGImages/'
__C.TEST.n_cpu = 16
__C.TEST.evaluation_interval = 1
__C.TEST.checkpoint_interval = 5
__C.TEST.iou_threshold = 0.5
__C.TEST.nms_threshold = 0.4
__C.TEST.conf_threshold = 0.01

cfg = __C
