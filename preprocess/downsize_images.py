import os
from PIL import Image
import numpy as np

input_path = '/Users/bryanwgranger/Desktop/bryan/DS/thesis/code/dataset/resized/resized_square/'
output_path = '/Users/bryanwgranger/Desktop/bryan/DS/thesis/code/dataset/resized/resized_square_small/'

img = Image.open(input_path + os.listdir(input_path)[0])
size = (128,128)
print(img)

new_img = img.resize(size)

print(new_img)

def main():
    size = (128,128)
    for i, f in enumerate(os.listdir(input_path)):
        if f[0] == '.':
            continue
        img = Image.open(input_path + f)
        new_img = img.resize(size)

        title, ext = f.split(".")
        new_filename = title + "_sm." + ext

        new_img.save(output_path + new_filename)

        if i % 100 == 0:
            print(f"{i} images done")

main()
