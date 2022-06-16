#  like others this is for data set

import os
# For static images:
path_in='/media/liang/ssd2/wash_hand_3/Domain-and-View-point-Agnostic-Hand-Action-Recognition-main/datasets/collect_data_batch_3/'
for step in os.listdir(path_in):
    os.chdir(path_in+step)  # 将当前工作目录修改为待修改文件夹的位置
    i=0
    for video in os.listdir(path_in+step):
        i=i+1
        os.rename(video,step+"_"+str(i)+".avi")


