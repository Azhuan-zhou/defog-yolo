import argparse
from easydict import EasyDict as edict

def arg_parse():
    parser = argparse.ArgumentParser()
    # 数据集遍历次数
    parser.add_argument("--epochs", type=int, default=80, help="number of epochs")
    #  梯度累计
    parser.add_argument("--gradient_accumulations", type=int, default=1, help="number of gradient accums before step")
    # 模型的配置文件地址
    parser.add_argument("--model_def", type=str, default=r".\model\model_config\yolov3.cfg",
                        help="path to model definition file")
    # 预训练权重文件地址
    parser.add_argument("--yolov3_pretrained_weights", type=str,
                        default=r'.\model\model_config\yolov3.weights',
                        help="if specified starts from checkpoints model")
    #  在生成batch的时候，指定核数
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    # 指定图片的输入尺寸，这里必须是32的倍数
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # 每遍历多少次（数据）保存一次模型
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
    # 每遍历多少次（数据）评估一次模型
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    # 是否每10个batch计算一次mAP
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    # 是否使用多尺度训练
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for training logs files (e.g. for "
                                                                   "TensorBoard)")
    parser.add_argument('--class_name', default='./data/classes/vocfog.names',
                        help='folder of the training data')
    parser.add_argument('--class_num', default=5, type=int)
    parser.add_argument('--ISP_FLAG', dest='ISP_FLAG', type=bool, default=True, help='whether use DIP Module')
    parser.add_argument('--train_path', default='./data/data_fog/voc_norm_train.txt',
                        help='folder of the training data')
    parser.add_argument('--epoch_first_stage', dest='epoch_first_stage', type=int, default=0, help='# of epochs')
    parser.add_argument('--epoch_second_stage', dest='epoch_second_stage', type=int, default=1, help='# of epochs')
    parser.add_argument('--pre_train', dest='pre_train', default=None,
                        help='the path of pretrained models if '
                             'is not null. not used for now')
    parser.add_argument('--val_path', default='./data/data_fog/voc_norm_test.txt',
                        help='folder of the training data')
    parser.add_argument('--WEIGHT_FILE', dest='WEIGHT_FILE', nargs='*',
                        default=None,
                        help='weight file')
    parser.add_argument('--WRITE_IMAGE_PATH', dest='WRITE_IMAGE_PATH', nargs='*',
                        default='./data/result', help='folder of prediction')
    parser.add_argument('--vocfog_traindata_dir', dest='vocfog_traindata_dir',
                        default='./data/data_fog/train/JPEGImages/',
                        help='the dir contains ten levels synthetic foggy images')
    parser.add_argument('--vocfog_valdata_dir', dest='vocfog_valdata_dir',
                        default='./data/data_fog/test/JPEGImages/',
                        help='the dir contains ten levels synthetic foggy images')
    return parser.parse_args()


args = arg_parse()
__C = edict()

# fog settings
__C.model_def = args.model_def
__C.n_cpu = args.n_cpu
__C.epochs = args.epochs
__C.gradient_accumulations = args.gradient_accumulations
__C.evaluation_interval = args.evaluation_interval
__C.val_path = args.val_path
__C.train_path = args.train_path
__C.checkpoint_interval = args.checkpoint_interval
__C.vocfog_traindata_dir = args.vocfog_traindata_dir
__C.vocfog_valdata_dir = args.vocfog_valdata_dir
__C.multiscale_training = args.multiscale_training

# YOLO options
__C.YOLO = edict()

# Set the class name
__C.YOLO.CLASSES = args.class_name
__C.YOLO.CLASSNUM = args.class_num
__C.YOLO.ANCHORS = "./data/anchors/coco_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY = 0.9995
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5
__C.YOLO.UPSAMPLE_METHOD = "resize"
__C.YOLO.ISP_FLAG = args.ISP_FLAG
__C.YOLO.pretrained_weights = args.yolov3_pretrained_weights

# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = args.train_path
__C.TRAIN.BATCH_SIZE = 1
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]

# __C.TRAIN.INPUT_SIZE            = [512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896]
__C.TRAIN.DATA_AUG = True
__C.TRAIN.LEARN_RATE_INIT = 1e-4
__C.TRAIN.LEARN_RATE_END = 1e-6
__C.TRAIN.WARMUP_EPOCHS = 2
__C.TRAIN.FISRT_STAGE_EPOCHS = args.epoch_first_stage
__C.TRAIN.SECOND_STAGE_EPOCHS = args.epoch_second_stage
__C.TRAIN.INITIAL_WEIGHT = args.pre_train

# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = args.val_path
__C.TEST.BATCH_SIZE = 2
__C.TEST.INPUT_SIZE = 416
__C.TEST.DATA_AUG = False
__C.TEST.WRITE_IMAGE = True
__C.TEST.WRITE_IMAGE_PATH = args.WRITE_IMAGE_PATH
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE = args.WEIGHT_FILE
__C.TEST.SHOW_LABEL = True
__C.TEST.iou_threshold = 0.5
__C.TEST.nms_threshold = 0.4
__C.TEST.conf_threshold = 0.5

cfg = __C
