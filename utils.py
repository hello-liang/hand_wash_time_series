import re
import numpy as np
# 50 guys do 6 action
def process_output_skelenton_to_array(results):
    # not sure the type of mediapipe output ,I use this function convert it to array
    out = ['1'] * 63
    # Print handedness and draw hand landmarks on the image.
    if not results.multi_hand_landmarks:
        out = out
        # can not find a hand ,initialize to 0
    else:
        # only choose the first one hand
        hand_landmarks = str(results.multi_hand_landmarks[0])
        hand_landmarks = re.split('\n}\nlandmark {\n  x: |\n  y: |\n  z: |\n}\n|landmark {\n  x: ', hand_landmarks)
        out = hand_landmarks[1:64]
    return out



# 50 guys do 6 action
def process_output_skelenton_to_array_muti_hand(results):
    # not sure the type of mediapipe output ,I use this function convert it to array
    out = np.ones(126)
    # Print handedness and draw hand landmarks on the image.
    if not results.multi_hand_landmarks:
        out = out
        # can not find a hand ,initialize to 0
    elif len(results.multi_handedness) == 1:

        results_class = str(results.multi_handedness[0])
        results_class = re.split('label: "|"\n}\n', results_class)
        results_class = results_class[1]

        if results_class == "Left":
            hand_landmarks = str(results.multi_hand_landmarks[0])
            hand_landmarks = re.split('\n}\nlandmark {\n  x: |\n  y: |\n  z: |\n}\n|landmark {\n  x: ', hand_landmarks)
            out[0:63] = hand_landmarks[1:64]
        else:
            hand_landmarks = str(results.multi_hand_landmarks[0])
            hand_landmarks = re.split('\n}\nlandmark {\n  x: |\n  y: |\n  z: |\n}\n|landmark {\n  x: ', hand_landmarks)
            out[63:126] = hand_landmarks[1:64]

    elif len(results.multi_handedness) == 2:  # 2 hand right first then left

        hand_landmarks = results.multi_hand_landmarks[0]
        hand_landmarks = str(hand_landmarks)
        hand_landmarks = re.split('\n}\nlandmark {\n  x: |\n  y: |\n  z: |\n}\n|landmark {\n  x: ', hand_landmarks)
        out[0:63] = hand_landmarks[1:64]
        hand_landmarks = results.multi_hand_landmarks[1]
        hand_landmarks = str(hand_landmarks)
        hand_landmarks = re.split('\n}\nlandmark {\n  x: |\n  y: |\n  z: |\n}\n|landmark {\n  x: ', hand_landmarks)
        out[63:126] = hand_landmarks[1:64]
    else:
        print("have more than two hand？")
        print(len(results.multi_handedness))
    return str(out.tolist())
