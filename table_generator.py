import os
import random

def genSizeFile(fileName, fileSize):
    #file path
    filePath="./fast_tier/"+fileName+".txt"
    # 生成固定大小的文件
    # date size
    ds=0
    with open(filePath, "w", encoding="utf8") as f:
        while ds<fileSize:
            f.write(str(round(random.uniform(-1000, 1000),2)))
            f.write("\n")
            ds=os.path.getsize(filePath)
    # print(os.path.getsize(filePath))

# start here.
num=input('Enter number of files want to generate: ')
num=int(num)
for i in range(num):
    size=random.randint(1000,10000000)
    genSizeFile("%d"%i,size)
    print('Generating %dth file'%i)

print('Complete!')
