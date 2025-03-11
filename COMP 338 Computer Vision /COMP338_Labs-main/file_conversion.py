import os

# 指定文件夹路径
folder_path = "C:/Users/Hailin/Documents/Obsidian-Vault/_legacy/comp124/labs"

# 指定输出文件名
output_file = "file_list.txt"

# 获取文件夹中的所有文件名
file_names = os.listdir(folder_path)

# 打开输出文件以写入模式
with open(output_file, "w") as file:
    # 遍历文件名列表，将每个文件名写入输出文件
    for file_name in file_names:
        file.write("[[ {} ]]\n".format(file_name))

print("文件名已写入 {} 文件中".format(output_file))
