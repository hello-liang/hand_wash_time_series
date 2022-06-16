#  like others this is for data set
import random

print(random.randint(0, 9))

import os
# For static images:
path_in='/media/liang/ssd2/wash_hand_3/Domain-and-View-point-Agnostic-Hand-Action-Recognition-main/datasets/handwash/HandWashDataset_self_one_hand/'
path_out='/media/liang/ssd2/wash_hand_3/Domain-and-View-point-Agnostic-Hand-Action-Recognition-main/datasets/handwash/HandWashDataset_after_clips/'


# create some folder

j=0
for subject in os.listdir(path_in):
    for action in os.listdir(path_in+subject):
        print(action)



j=0
for subject in os.listdir(path_in):
    for action in os.listdir(path_in+subject):
        print(action)
        ms=open(path_in+subject+"/"+action+"/joint.txt")
        temp=[]
        i=0
        for line in ms.readlines():

            temp.append(line)
            print(len(temp))
            if len(temp)==31 : #20%
                del temp[0]
                i=i+1
                if i>120:
                    continue
                if random.randint(0, 9)<1:
                    j = j + 1

                    if not os.path.exists(path_out+str(j)+"/"+action):
                        os.makedirs(path_out+str(j)+"/"+action)

                    write_file=open(path_out+str(j)+"/"+action+"/joint.txt","w")
                    for i in range(30):
                        write_file.write(temp[i])
                    write_file.close()
        ms.close()





