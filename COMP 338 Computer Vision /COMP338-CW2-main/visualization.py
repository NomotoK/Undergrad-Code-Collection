import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件，假设文件名为data.csv
file_path = 'test.csv'  # 将文件路径替换为你的文件路径
data = pd.read_csv(file_path)

# 提取第一列作为横轴数据，第二列作为纵轴数据
epoch = data.iloc[:, 0]
accuracy = data.iloc[:, 1]

# 绘制折线图
plt.plot(epoch, accuracy, label='Accuracy')

# 添加标题和标签
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# 显示图例
plt.legend()

# 显示图形
plt.show()
