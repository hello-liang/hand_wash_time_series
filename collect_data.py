#!/usr/bin/python
# -*- coding: utf-8 -*-


import time
import cv2
import os
import tkinter as tk  # 使用Tkinter前需要先导入


def CollectData():
    # root='/media/arl/ssd2/collect_data_code'
    root = ''
    # root='/media/liang/ssd22/collect_data_code'
    video_name = os.listdir(root + 'collect_data')
    begin = 0
    if len(video_name) == 0:
        begin = 0
    else:
        for i in range(len(video_name)):
            num_video = int(video_name[i].split('.')[0])
            if begin < num_video:
                begin = num_video

    start_all_time = time.time()
    cost_control_time = time.time() - start_all_time

    wash_time = 11

    # This will return video from the first webcam on your computer.
    cap = cv2.VideoCapture(0)
    num_f=0

    #ret = cap.set(3, 320)


    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(root + 'collect_data/' + str(begin + 1) + '.avi', fourcc, 10, (640, 480))

    # loop runs if capturing has been initialized.
    while (cost_control_time < wash_time):
        num_f+=1
        # reads frames from a camera
        # ret checks return at each frame
        ret, frame = cap.read()
        out.write(frame)
        cv2.imshow('Original', frame)
        # Wait for 'a' key to stop the program
        cost_control_time = time.time() - start_all_time


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(num_f/wash_time)
    print(cap.get(3))
    print(cap.get(4))

    print(cap.get(5))

    # Close the window / Release webcam
    cap.release()
    cv2.destroyAllWindows()


CollectData()

# Python program to illustrate
# saving an operated video

# organize imports


# # 第1步，实例化object，建立窗口window
# window = tk.Tk()
#
# # 第2步，给窗口的可视化起名字
# window.title('wash hand monitor project')
#
# # 第3步，设定窗口的大小(长 * 宽)
# window.geometry('1000x600')  # 这里的乘是小x
#
# # 第4步，在图形界面上设定标签
# l = tk.Label(window, text='Want to wash hand?', bg='green', font=('Arial', 30), width=30, height=3)#, width=30, height=2
# # 说明： bg为背景，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
#
# # 第5步，放置标签
# l.pack()    # Label内容content区域放置位置，自动调节尺寸
# # 放置lable的方法有：1）l.pack(); 2)l.place();
# B = tk.Button(window, text ="yes,click here", command = CollectData,bg='white',width=30, height=3)
#
# B.pack()
# # 第6步，主窗口循环显示
# window.mainloop()
# # 注意，loop因为是循环的意思，window.mainloop就会让window不断的刷新，如果没有mainloop,就是一个静态的window,传入进去的值就不会有循环，mainloop就相当于一个很大的while循环，有个while，每点击一次就会更新一次，所以我们必须要有循环
# # 所有的窗口文件都必须有类似的mainloop函数，mainloop是窗口文件的关键的关键。
