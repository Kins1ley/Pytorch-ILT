import torch

# class Constant(object):
OPC_ITERATION_THRESHOLD = 20    # ILT迭代次数
OPC_TILE_SIZE = 2048 * 2048     # MASK大小
LITHOSIM_OFFSET = 512           # glp文件中坐标的原点
OPC_TILE_X = 2048   # MASK行数
OPC_TILE_Y = 2048   # MASK列数

MASK_TILE_END_X = LITHOSIM_OFFSET + 1280    # ?
MASK_TILE_END_Y = LITHOSIM_OFFSET + 1280    # ?
OPC_LENGTH_CORNER_RESHAPE = 10

MASK_PRINTABLE_THRESHOLD = 0.5
MASKRELAX_SIGMOID_STEEPNESS = 4.0

# SRAF related constant parameters
OPC_SPACE_SRAF = 15
OPC_WIDTH_SRAF = 10
OPC_SPACE_FORBID_SRAF = 100

# Gradient Descent related parameters
OPC_INITIAL_STEP_SIZE = 1
OPC_JUMP_STEP_SIZE = OPC_INITIAL_STEP_SIZE / 2
GRADIENT_DESCENT_ALPHA = 0.5
GRADIENT_DESCENT_BETA = 0.75

# Simulate Image related parameters
MAX_DOSE = 1.02
MIN_DOSE = 0.98
NOMINAL_DOSE = 1.00
TARGET_INTENSITY = 0.225
PHOTORISIST_SIGMOID_STEEPNESS = 50
KERNEL_X = 35
KERNEL_Y = 35

# Other common constant value
ZERO_ERROR = 0.000001
WEIGHT_EPE_REGION = 0.5
WEIGHT_PVBAND = 1
WEIGHT_REGULARIZATION = 0.025
#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')