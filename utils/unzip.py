import zipfile
import os

def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# 使用示例
zip_file_path = '../data/VOC.zip'  # 替换为你的.zip文件路径
extract_to_path = '../data/'  # 替换为你想要解压到的文件夹路径

# 确保提供的路径存在，如果不存在则创建
os.makedirs(extract_to_path, exist_ok=True)

# 调用解压函数
unzip_file(zip_file_path, extract_to_path)
