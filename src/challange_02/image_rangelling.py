import cv2
import matplotlib.pyplot as plt
import os

%matplotlib inline

path = "/home/prakash/Titan/images"

max_height = 128
max_width = 128

def resize_image(max_height, max_width, img):
    height, width = img.shape[:2]
    
    if max_height < height or max_width < width:
        scaling_factor = max_height / float(height)
    if max_width/float(width) < scaling_factor:
        scaling_factor = max_width / float(width)
    
    img = cv2.resize(
        img, None, fx=scaling_factor, 
        fy=scaling_factor, 
        interpolation=cv2.INTER_CUBIC
        )
    return img       


def padding(img):
    row, col= img.shape[:2]
    bottom= img[row-2:row, 0:col]
    mean= cv2.mean(bottom)[0]
    print("img in between: ", img.shape)
    top, bottom, left, right = get_padding_args(max_height, max_width, img)
    print("top {0}, bottom {1}, left {2}, right {3}".format(top, bottom, left, right))
    img = cv2.copyMakeBorder(img, top=top, bottom=bottom, left=left, right=right, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
    print("img after : ", img.shape)
    return img

def get_padding_args(max_height, max_width, img): 
    original_height, original_width = img.shape[:2]

    delta_height = max_height - original_height
    delta_width = max_height - original_width

    print("delta_height {0}, delta_width {1}".format(delta_height, delta_width))   

    top = 0
    bottom = 0
    if(delta_height % 2 != 0):
        top = (int)(delta_height / 2)
        bottom = ((int)(delta_height / 2)) + 1        
    else:
        top = (int)(delta_height / 2)
        bottom = (int)(delta_height / 2) 
    
    left = 0
    right = 0
    if(delta_width % 2 != 0):
        left = (int)(delta_width / 2)
        right = ((int)(delta_width / 2)) + 1  
    else:
        left = (int)(delta_width / 2)
        right = (int)(delta_width / 2)          

    return top, bottom, left, right

def get_directories(path):
    return os.listdir(path)

def create_directory(path, name):
    directory = path + "/" + name + "01"
    
    if not os.path.exists(directory):
        os.makedirs(directory)  
    return directory 

def get_all_images(path):
    included_extenstions = ['jpg', 'bmp', 'png', 'jpeg']
    file_names = [fn for fn in os.listdir(path)
        if any(fn.endswith(ext) for ext in included_extenstions)]
    return  file_names          

def save_image(padded_image, name, new_directory):
    file_path = new_directory + "/" + name, padded_image
    if not os.path.exists(file_path):
        cv2.imwrite(file_path) 

if __name__ == "__main__":
    directories = get_directories(path)
    for directory in directories:
        new_directory = create_directory(path=path, name=directory)
        images = get_all_images(path + "/" + directory)
        for image in images:
            img = cv2.imread(path + "/" + directory + "/" + image)
            print("Inital image shape: ", img.shape)
            img = resize_image(
                max_height=max_height,
                max_width=max_width,
                img=img
            )
            padded_image = padding(img=img)
            save_image(
                padded_image = padded_image, 
                name = image,
                new_directory = new_directory)
    
    
    #plt.imshow(img)
