import os
from PIL import Image
import numpy as np

input_path = '/Users/bryanwgranger/Desktop/bryan/DS/thesis/code/dataset/resized/resized/'
output_path = '/Users/bryanwgranger/Desktop/bryan/DS/thesis/code/dataset/resized/resized_square/'

def make_square(img):
    img_arr = np.array(img)
    h = img_arr.shape[0]
    w = img_arr.shape[1]
    if w > h:
        diff = w - h
        new_img_arr = img_arr[:, int(diff/2):(w-int(diff/2))]
    else:
        diff = h - w
        new_img_arr = img_arr[int(diff/2):(h-int(diff/2)),:]

    return new_img_arr

def main():
    for i, f in enumerate(os.listdir(input_path)):
        if f[0] == '.':
            continue
        img = Image.open(input_path + f)
        new_img_arr = make_square(img)
        new_img = Image.fromarray(new_img_arr)

        title, ext = f.split(".")
        new_filename = title + "_square." + ext

        new_img.save(output_path + new_filename)

        if i % 100 == 0:
            print(f"{i} images done")

main()