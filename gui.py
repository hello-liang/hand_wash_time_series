import tkinter as tk
import tkinter.font as tkFont
import mutiprocessing_collect_data_3
import mutiprocessing_predict_and_collect_data_3


class SmartHandwashVisionSystem:
    def __init__(self, root):
        # setting title
        root.title("Smart Handwash Vision System")
        # setting window size
        width=1024
        height=768
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        welcome_label=tk.Label(root)
        welcome_label["anchor"] = "c"
        ft = tkFont.Font(family='Times',size=28)
        welcome_label["font"] = ft
        welcome_label["fg"] = "#333333"
        welcome_label["justify"] = "center"
        welcome_label["text"] = "Welcome to Smart Handwash Vision System"
        welcome_label.place(x=170,y=90,width=671,height=88)

        select_mode_label=tk.Label(root)
        ft = tkFont.Font(family='Times',size=18)
        select_mode_label["font"] = ft
        select_mode_label["fg"] = "#333333"
        select_mode_label["justify"] = "center"
        select_mode_label["text"] = "Please select a mode"
        select_mode_label.place(x=310,y=260,width=378,height=79)

        collect_data_button=tk.Button(root)
        collect_data_button["bg"] = "#01aaed"
        ft = tkFont.Font(family='Times',size=18)
        collect_data_button["font"] = ft
        collect_data_button["fg"] = "#000000"
        collect_data_button["justify"] = "center"
        collect_data_button["text"] = "Data collection mode"
        collect_data_button.place(x=120,y=410,width=353,height=77)
        collect_data_button["command"] = self.collect_data_button_command

        realtime_feedback_button=tk.Button(root)
        realtime_feedback_button["bg"] = "#01aaed"
        ft = tkFont.Font(family='Times',size=18)
        realtime_feedback_button["font"] = ft
        realtime_feedback_button["fg"] = "#000000"
        realtime_feedback_button["justify"] = "center"
        realtime_feedback_button["text"] = "Real-time Feedbacking mode"
        realtime_feedback_button.place(x=530,y=410,width=354,height=78)
        realtime_feedback_button["command"] = self.realtime_feedback_button_command

    def collect_data_button_command(self):
        mutiprocessing_collect_data_3.collect_data()

    def realtime_feedback_button_command(self):
        mutiprocessing_predict_and_collect_data_3.predict_collect()


if __name__ == "__main__":
    root = tk.Tk()
    SmartHandwashVisionSystem = SmartHandwashVisionSystem(root)
    root.mainloop()
