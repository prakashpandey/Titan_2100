import image_rangelling as ir
import image_prediction as ip
from io import BytesIO
import requests
import numpy as np
from PIL import Image
import urllib.request

def read_img_from_url(url):
    print("url: ", url)
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image, np.asarray(image)

def get_pil_image(target):
    ir.get_pil_image(
        path = path, 
        name = name
    )

def download_image(url, target, name):
    target_dir = target + "/" + name
    urllib.request.urlretrieve(url, target_dir)
    return target_dir
 

max_height = 128
max_width = 128
target = "/home/prakash/Titan/testing"
name = "test_img.jpg"

if __name__ == "__main__":

    url = "https://shop.epictv.com/sites/default/files/ae42ad29e70ba8ce6b67d3bdb6ab5c6e.jpeg"
    
    download_image(
        url = url, 
        target = target, 
        name = name)

    img_cv2 = ip.read_img(
        path = target,
        name = name
    )    

    model = ip.main()
    img = ir.resize_image(
                max_height=max_height,
                max_width=max_width,
                img=img_cv2
            )
    padded_image = ir.padding(img=img) 
    
    ir.save_image(
        padded_image = padded_image,
        name = name,
        new_directory = target
    )

    img_pil = ir.get_pil_image(
        path = target,
        name = name
    )
    equalized_img = ir.equalize_img(img_pil) 

    raveled_image = ip.ravel_image(
        img = np.asarray(equalized_img)
    )

    features = []
    features.append(raveled_image)
    features = np.array(features)
    preds = ip.predict(
            model = model, 
            features_x = features
            )
    print("Printing preds...")
    print(preds) 

    print("Printing directory_hash_ keys...")
    print(ip.directory_hash_dic)


             