import os
import shutil
from tqdm import tqdm
import warnings
from argparse import ArgumentParser
from lob_data import lob_data
from norm_data import norm_data
from split_data import get_stock, split_data


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # 要预处理的数据类型
    args = ArgumentParser(description="data type for preprocess")
    args.add_argument("--data_type", "-type", default=["train_data"], nargs="+")
    options = args.parse_args()

    # 获得训练与测试数据
    if not os.path.exists(os.path.join("..", "dataset")):
        stock = get_stock()
        split_data(stock_name=stock)

    # 预处理数据
    for option in tqdm(options.data_type):
        norm_data(label=option)
        lob_data(label=option)

    # 删除初始数据
    train_folder_path = "../dataset/train_data"
    test_folder_path = "../dataset/test_data"

    # 使用try-except结构以处理可能出现的异常
    try:
        # 删除文件夹及其所有内容
        shutil.rmtree(train_folder_path)
        print(f"文件夹 '{train_folder_path}' 已成功删除")
    except FileNotFoundError:
        print(f"文件夹 '{train_folder_path}' 未找到")
    except PermissionError:
        print(f"没有足够的权限来删除文件夹 '{train_folder_path}'")
    except Exception as e:
        print(f"删除文件夹 '{train_folder_path}' 时出现错误：{e}")

    # 使用try-except结构以处理可能出现的异常
    try:
        # 删除文件夹及其所有内容
        shutil.rmtree(test_folder_path)
        print(f"文件夹 '{test_folder_path}' 已成功删除")
    except FileNotFoundError:
        print(f"文件夹 '{test_folder_path}' 未找到")
    except PermissionError:
        print(f"没有足够的权限来删除文件夹 '{test_folder_path}'")
    except Exception as e:
        print(f"删除文件夹 '{test_folder_path}' 时出现错误：{e}")
