from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'DeMo'
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH_T = '/path/to/your/vitb_16_224_21k.pth'
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
# The loss type of metric loss
# options:['triplet'](without center loss) or ['center','triplet_center'](with center loss)
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
_C.MODEL.PROMPT = False # From MambaPro
_C.MODEL.ADAPTER = False # From MambaPro
_C.MODEL.FROZEN = False # whether to freeze the backbone
_C.MODEL.HDM = False # whether to use HDM in DeMo
_C.MODEL.ATM = False # whether to use ATM in DeMo
# SACR: Scale-Adaptive Contextual Refinement
_C.MODEL.USE_SACR = False # whether to use SACR before SDTPS
_C.MODEL.SACR_DILATION_RATES = [2, 3, 4] # dilation rates (adjusted for 16×8 feature maps)
# SDTPS: Sparse and Dense Token-Aware Patch Selection
_C.MODEL.USE_SDTPS = False # whether to use SDTPS (replaces HDM+ATM)
_C.MODEL.SDTPS_SPARSE_RATIO = 0.5 # token selection ratio (same as paper)
_C.MODEL.SDTPS_AGGR_RATIO = 0.4 # token aggregation ratio (same as paper)
_C.MODEL.SDTPS_BETA = 0.25 # score combination weight parameter
_C.MODEL.SDTPS_USE_GUMBEL = False # whether to use Gumbel-Softmax
_C.MODEL.SDTPS_GUMBEL_TAU = 1.0 # Gumbel temperature
_C.MODEL.SDTPS_LOSS_WEIGHT = 2.0 # loss weight for SDTPS branch (vs ori=1.0)
_C.MODEL.SDTPS_CROSS_ATTN_TYPE = 'cosine' # 'cosine' (原始余弦相似度) or 'attention' (真正的Cross-Attention)
_C.MODEL.SDTPS_CROSS_ATTN_HEADS = 4 # number of heads for Cross-Attention (when type='attention')
# Trimodal-LIF: Quality-aware multi-modal fusion (M2D-LIF framework)
_C.MODEL.USE_LIF = False # whether to use Trimodal-LIF for quality-aware fusion
_C.MODEL.LIF_BETA = 0.4 # fusion weight temperature for LIF
_C.MODEL.LIF_LOSS_WEIGHT = 0.1 # LIF loss weight (auxiliary loss)
_C.MODEL.LIF_LAYER = 3 # which layer to apply LIF (3, 4, or 5)
# DGAF: Dual-Gated Adaptive Fusion (AGFN paper)
# 用于 SDTPS 输出的自适应融合，替代简单的 concat
_C.MODEL.USE_DGAF = False # whether to use Dual-Gated Adaptive Fusion after SDTPS
_C.MODEL.DGAF_VERSION = 'v3' # 'v1' (需要mean后输入), 'v3' (直接接受tokens，内置attention pooling)
_C.MODEL.DGAF_TAU = 1.0 # temperature for entropy gate (lower = sharper)
_C.MODEL.DGAF_INIT_ALPHA = 0.5 # initial value for alpha (balance IEG and MIG)
_C.MODEL.DGAF_NUM_HEADS = 8 # number of attention heads for V3's attention pooling
# MultiModal-SACR: 多模态交互版本的 SACR
# 将三个模态沿 token 维度拼接，在 SACR 中进行跨模态交互，然后拆分
_C.MODEL.USE_MULTIMODAL_SACR = False # whether to use MultiModal SACR after SDTPS
_C.MODEL.MULTIMODAL_SACR_VERSION = 'v1' # 'v1' or 'v2' (v2 has cross-modal attention)
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with the contact feature
_C.MODEL.DIRECT = 1

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224'
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.GLOBAL_LOCAL = False # Whether to use the local information in PIFE for DeMo
_C.MODEL.HEAD = 12 # Number of heads in the ATMoE

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = True
_C.MODEL.SIE_VIEW = False  # We do not use this parameter


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('RGBNT201')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('./data')

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 14  # This may be affected by the order of data reading
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax_triplet'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16  # You can adjust it to 8 to save memory while the batch_size need to be 64 to ensure the number of ID

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "SGD"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 120
# Base learning rate
_C.SOLVER.BASE_LR = 0.009
# Factor of learning bias
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.BIAS_LR_FACTOR = 2
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Margin of cluster ;pss
_C.SOLVER.CLUSTER_MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
# Settings of range loss
_C.SOLVER.RANGE_K = 2
_C.SOLVER.RANGE_MARGIN = 0.3
_C.SOLVER.RANGE_ALPHA = 0
_C.SOLVER.RANGE_BETA = 1
_C.SOLVER.RANGE_LOSS_WEIGHT = 1
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0001
# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 10
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"
# Learning rate scheduler type: 'cosine' or 'multistep'
_C.SOLVER.LR_SCHEDULER = 'cosine'

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30
_C.SOLVER.SEED = 1111
_C.MODEL.NO_MARGIN = True
# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 10
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 1
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 128  # You can adjust it to 64

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 256
# If test with re-ranking, options: 'yes','no'
_C.TEST.RE_RANKING = 'no'
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'before'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# Pattern of test augmentation
_C.TEST.MISS = 'None'
# ----------------------------------------------------------a------------------ #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "./test"
