from PIL import Image

# 要转换的图片路径
img_path = 'image.jpg'

# 将图片调整为指定的宽度和高度
width, height = 100, 50
img = Image.open(img_path)
img = img.resize((width, height))

# 定义 ASCII 字符画所需的字符集
# 这里使用的是从黑到白的 70 个字符
char_set = list(" .,-'`:!1+*abcdefghijklmnopqrstuvwxyz<>()\/{}[]?234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ%&@#$")

# 将每个像素点转换为对应的 ASCII 字符
# 根据每个像素点的灰度值，选择合适的字符
ascii_img = ""
for y in range(height):
    for x in range(width):
        pixel = img.getpixel((x, y))
        if pixel == (255, 255, 255):  # 如果是白色像素，则用空格代替
            ascii_img += " "
        else:
            gray = (pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114)
            index = int((gray / 255) * (len(char_set) - 1))
            ascii_img += char_set[index]
    ascii_img += "\n"

# 输出 ASCII 字符画到控制台
print(ascii_img)
