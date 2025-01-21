import cv2
import numpy as np
import os

def edgedetection(Input, Output):

    img = cv2.imread(Input, 0)
    canny = cv2.Canny(img, 100, 200)

    output_file = f"{Output}{Input[16:]}.npy"

    np.save(output_file, canny)

    

if __name__ == "__main__":
    Output = "data/crack/edges/"
    path = "data/crack/imgs"
    dir_list = os.listdir(path)
    for img in dir_list:
        Input = f"data/crack/imgs/{img}"
        edgedetection(Input, Output)


