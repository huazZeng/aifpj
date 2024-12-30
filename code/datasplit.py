import os
import lmdb
import cv2
import numpy as np

# def create_lmdb_dataset(output_path, image_dir, label_file):
#     """
#     将数据集转换为 LMDB 格式
#     :param output_path: LMDB 数据库输出路径
#     :param image_dir: 图像文件夹路径
#     :param label_file: 标签文件路径
#     """
#     # 确保路径格式正确
#     output_path = os.path.normpath(output_path)
    
#     # 检查并创建输出路径
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     # 打开LMDB数据库，设置最大内存映射大小（1GB）
#     try:
#         env = lmdb.open(output_path, map_size=109951162777)
#     except lmdb.Error as e:
#         print(f"Failed to open LMDB at {output_path}: {e}")
#         return

#     with env.begin(write=True) as txn:
#         with open(label_file, 'r', encoding='utf-8') as f:
#             lines = f.readlines()

#         # 写入样本总数
#         txn.put('num-samples'.encode(), str(len(lines)).encode())

#         for i, line in enumerate(lines):
#             line = line.strip()
#             if not line:
#                 continue

#             parts = line.split("\t", 1)
#             if len(parts) < 2:
#                 print(parts)
#                 print(f"Warning: Skipping invalid line: {line}")
#                 continue

#             img_name, label = parts
#             img_path = os.path.join(image_dir, img_name)

#             # 加载图像
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 print(f"Warning: Failed to read {img_path}")
#                 continue
#             _, img_encoded = cv2.imencode('.jpg', img)

#             # 写入图像数据和标签
#             img_key = f'image-{i+1:09d}'.encode()
#             label_key = f'label-{i+1:09d}'.encode()
#             txn.put(img_key, img_encoded.tobytes())
#             txn.put(label_key, label.encode())

#     print(f"LMDB dataset created at {output_path}")

# def create_train_test_lmdb(image_dir, label_train_file, label_test_file, output_dir):
#     """
#     分别为训练集和测试集创建 LMDB 数据库
#     :param image_dir: 图像文件夹路径
#     :param label_train_file: 训练集标签文件路径
#     :param label_test_file: 测试集标签文件路径
#     :param output_dir: LMDB 输出文件夹路径
#     """
#     # 创建训练集 LMDB
#     train_lmdb_path = os.path.join(output_dir, "lmdb_train")
#     create_lmdb_dataset(train_lmdb_path, os.path.join(image_dir, ""), label_train_file)

#     # 创建测试集 LMDB
#     test_lmdb_path = os.path.join(output_dir, "lmdb_test")
#     create_lmdb_dataset(test_lmdb_path, os.path.join(image_dir, ""), label_test_file)

# # 使用
# image_dir = "F:/workspace/AIFPJ/ic15_rec"  # 修改为实际的图像根目录
# label_train_file = "F:/workspace/AIFPJ/ic15_rec/train/labels.txt"
# label_test_file = "F:/workspace/AIFPJ/ic15_rec/test/labels.txt"
# output_dir = "F:/workspace/AIFPJ/lmdb_output"  # 输出的lmdb存储路径
# create_train_test_lmdb(image_dir, label_train_file, label_test_file, output_dir)
import os

def scan_and_create_vocab(label_file):
    """
    扫描标签文件并生成字符集合。
    
    :param label_file: 标签文件路径
    :return: 字符集合
    """
    char_set = set()  # 用集合来存储所有出现过的字符，自动去重
    
    # 打开标签文件并逐行读取
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) < 2:
            continue
        _, label = parts
        
        # 遍历标签中的每个字符并加入集合
        for char in label:
            char_set.add(char)
    
    return char_set

def save_vocab_to_file(char_set, output_file):
    """
    将字符集合保存到文件，每个字符在一行。
    
    :param char_set: 字符集合
    :param output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # 将字符集合转换为字符串，并直接连接，没有空格
        f.write(''.join(sorted(char_set)))

# 使用
label_train_file = "F:/workspace/AIFPJ/ic15_rec/train/labels.txt"  # 修改为实际的标签文件路径
vocab_file = "F:/workspace/AIFPJ/vocab.txt"  # 输出的字符集合文件路径

# 扫描标签并生成字符集合
char_set = scan_and_create_vocab(label_train_file)

# 保存字符集合到文件
save_vocab_to_file(char_set, vocab_file)

print(f"Character vocabulary saved to {vocab_file}")
