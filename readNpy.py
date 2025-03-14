# 该文件作用是将test文件中的groudtrueth npy文件转换成真彩色图像。
import numpy as np
import matplotlib.pyplot as plt
import os
import  glob
# depthmap = np.load('10023.npy')    #使用numpy载入npy文件
# plt.imshow(depthmap)              #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
# # plt.colorbar()                   #添加colorbar
# plt.savefig('1.jpg')       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
# plt.show()                        #在线显示图像

#若要将图像存为灰度图，可以执行如下两行代码
#import scipy.misc
#scipy.misc.imsave("depth.png", depthmap)

#遍历
path ='/media/admin1/hdd1/Data/crowdCounting/privateData/FishTotal/train_data/ground_truth'
filepath = os.path.join(path, 'GTTrueColor/')
if not os.path.exists(filepath):
    os.makedirs(filepath)
for img_path in glob.glob(os.path.join(path, '*.npy')):
    densitymap = np.load(img_path, allow_pickle=True)
    plt.imshow(densitymap)

    filename = os.path.basename(img_path.split('.')[0])
    plt.savefig(os.path.join(filepath, filename+'.png'), dpi=300)
