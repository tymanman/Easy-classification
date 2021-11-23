##数据集的类别
NUM_CLASSES = 2
#训练时batch的大小
BATCH_SIZE = 16
#网络默认输入图像的大小
INPUT_SIZE = [500, 500]
#训练最多的epoch
MAX_EPOCH = 100
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
EVAL_PERIOD = 5
# 初始学习率
LR = 1e-3
# 冻结层
FREEZE_LAYER_NUM = 0

BASE = "/home/fcq/Li/dataset/rotation_cls/"
# 训练好模型的保存位置
SAVE_FOLDER = "./checkpoints"
#数据集的存放位置
TRAIN_TXT =BASE + 'train.txt'     
VAL_TXT = BASE + 'val.txt'
