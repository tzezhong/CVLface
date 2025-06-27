# config.py

# 模型配置
RECOGNITION_MODEL_ID = 'minchul/cvlface_adaface_ir101_webface12m'
ALIGNER_ID = 'minchul/cvlface_DFA_mobilenet'

# 识别参数
THRESHOLD = 0.3
BATCH_SIZE = 200

# 文件和目录配置
OUTPUT_CSV_FILENAME = 'matched_results.csv'
DEALED_DIR_NAME = 'dealed_dir' # 处理后的输入图片存放目录名

# 运行参数
SCAN_INTERVAL_SECONDS = 1 # 扫描新图片的时间间隔（秒）

# 图像归一化参数 (一般是标准值，但也可以配置)
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]