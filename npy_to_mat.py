import os
import numpy as np
import scipy.io
def convert_npy_to_mat(input_folder, output_folder):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        # 检查文件是否为 .npy 文件
        if file_name.endswith('.npy'):
            # 构造完整的文件路径
            npy_file_path = os.path.join(input_folder, file_name)
            # 加载 .npy 文件
            npy_data = np.load(npy_file_path)
            # 构造输出 .mat 文件路径
            mat_file_name = file_name.replace('.npy', '.mat')
            mat_file_path = os.path.join(output_folder, mat_file_name)
            # 保存为 .mat 文件
            scipy.io.savemat(mat_file_path, {'final_gt': npy_data})
            print(f'Converted {file_name} to {mat_file_name}')
# 指定输入文件夹和输出文件夹路径
input_folder = '/media/admin1/hdd1/Data/crowdCounting/privateData/FishTotal/usedData/train/ground_truth'
output_folder = '/media/admin1/hdd1/Data/crowdCounting/privateData/FishTotal/usedData/train/gt_mat'
convert_npy_to_mat(input_folder, output_folder)


# #csv to mat
# import os
# import numpy as np
# import scipy.io
# import pandas as pd
# def convert_npy_to_mat(input_folder, output_folder):
#     # 如果输出文件夹不存在，则创建
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     # 遍历输入文件夹中的所有文件
#     for file_name in os.listdir(input_folder):
#         # 检查文件是否为 .npy 文件
#         if file_name.endswith('.csv'):
#             # 加载csv
#             den = pd.read_csv(os.path.join(input_folder, os.path.splitext(file_name)[0] + '.csv'), sep=',',
#                               header=None).values
#             # 构造输出 .mat 文件路径
#             mat_file_name = file_name.replace('.csv', '.mat')
#             mat_file_path = os.path.join(output_folder, mat_file_name)
#             # 保存为 .mat 文件
#             scipy.io.savemat(mat_file_path, {'final_gt': den})
#             print(f'Converted {file_name} to {mat_file_name}')
# # 指定输入文件夹和输出文件夹路径
# # input_folder = '/media/admin1/hdd1/Data/crowdCounting/privateData/FishTotal/usedData/train/ground_truth'
# # output_folder = '/media/admin1/hdd1/Data/crowdCounting/privateData/FishTotal/usedData/train/gt_mat'
# input_folder = '/media/admin1/hdd1/Projects/crowdcount-mcnn-master/data/formatted_trainval/shanghaitech_part_A_patches_9/test_den'
# output_folder = '/media/admin1/hdd1/Projects/crowdcount-mcnn-master/data/formatted_trainval/shanghaitech_part_A_patches_9/test_den'
# convert_npy_to_mat(input_folder, output_folder)
