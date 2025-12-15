import logging
import os
import sys

def get_logger(name, log_file='../prediction_result/run.log'):
    """
    创建一个配置好的logger，同时输出到控制台和文件
    """
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 文件 Handler
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        
        # 控制台 Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

def log_evaluation_callback(period=10, logger=None):
    """
    LightGBM callback for logging evaluation results
    """
    def _callback(env):
        if period > 0 and env.iteration % period == 0:
            result_list = env.evaluation_result_list
            # result structure: (dataset_name, metric_name, value, is_higher_better, stdv)
            msg = f"[{env.iteration}] "
            for res in result_list:
                msg += f"{res[0]}-{res[1]}: {res[2]:.6f} "
            if logger:
                logger.info(msg)
            else:
                print(msg)
    return _callback
