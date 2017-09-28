import PIL
from PIL import Image
from skimage.feature import canny
#from deron_preprocess import *
import deron_preprocess as der
import os
import datetime

start = datetime.datetime.now()

# os.makedirs(DEST_TRAIN_THUMB_DIR, exist_ok=True)
# slide_to_thumbs(1, 15)
# multiprocess_slide_to_thumbs()
# slide_stats()
# slide_info()



folder = "example_filters" + os.sep
img_path = der.get_train_thumb_path(2)
pil_img = Image.open(img_path)
# or_img.show()

orig_img = der.pil_to_np(pil_img)
gray_img = der.filter_rgb_to_grayscale(orig_img)
# der.np_to_pil(gray_img).show()
# der.np_to_pil(gray_img).save(folder + "gray.jpg")
der.ar_info(gray_img, "Gray")

canny_edges = canny(gray_img, 1, low_threshold=0, high_threshold=25)
der.ar_info(canny_edges, "Canny Edges")

# canny_img = 255*canny(gray_img, 1, low_threshold=0, high_threshold=50).astype(float) # pretty good
# canny_img = 255*canny(gray_img, 0, low_threshold=0, high_threshold=50).astype(float) # also pretty good
#canny_img = 255*canny(gray_img, 2, low_threshold=0, high_threshold=50).astype(float) # not good
canny_img = canny_edges.astype("uint8") * 255 # pretty good
# canny_img = 255*canny(gray_img, 1, low_threshold=0, high_threshold=10).astype(float) # pretty good
# canny_img = 255*canny(gray_img, 1, low_threshold=0, high_threshold=5).astype(float) # pretty good
# canny_img = 255*canny(gray_img, 1, low_threshold=255, high_threshold=255).astype(float) # not good
# canny_img = 255*canny(gray_img, 0).astype(float)
der.ar_info(canny_img, "Canny")
der.np_to_pil(canny_img).show()

end = datetime.datetime.now()
delta = end - start
print(str(delta))

# der.do_filters(4)