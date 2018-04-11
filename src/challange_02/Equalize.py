from PIL import ImageOps
from PIL import Image
import image_rangelling as ir
import numpy as np

path = "/home/prakash/Titan/padded_images"
path_equalize = "/home/prakash/Titan/padded_equalize"

def pil_equalize(img):
    return ImageOps.equalize(img)

def pil_read_image(path, name):
    img = Image.open(path + "/" + name)    
    return img

def pil_save_image(equalized_img, name, new_directory):
    equalized_img.save(new_directory + "/" + name + ".jpg")   

if __name__ == "__main__":
    directories = ir.get_directories(path)
    for directory in directories:
        new_directory = ir.create_directory(path=path_equalize, name=directory)
        images = ir.get_all_images(path + "/" + directory)
        for image in images:
            img = pil_read_image(path + "/" + directory , image)
            equalized_img = pil_equalize(img)
            pil_save_image(
                equalized_img = equalized_img,
                name = image,
                new_directory = new_directory
                )

            

