import PIL
from PIL import Image
from skimage.feature import canny
# from deron_preprocess import *
import deron_preprocess as der
import os
import datetime
from skimage import feature
from skimage import filters
from skimage.util import view_as_blocks

start = datetime.datetime.now()

# os.makedirs(DEST_TRAIN_THUMB_DIR, exist_ok=True)
# slide_to_thumbs(1, 15)
# multiprocess_slide_to_thumbs()
# slide_stats()
# slide_info()


folder = "example_filters" + os.sep
img_path = der.get_train_thumb_path(4)
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
# canny_img = 255*canny(gray_img, 2, low_threshold=0, high_threshold=50).astype(float) # not good
canny_img = canny_edges.astype("uint8") * 255  # pretty good
# canny_img = 255*canny(gray_img, 1, low_threshold=0, high_threshold=10).astype(float) # pretty good
# canny_img = 255*canny(gray_img, 1, low_threshold=0, high_threshold=5).astype(float) # pretty good
# canny_img = 255*canny(gray_img, 1, low_threshold=255, high_threshold=255).astype(float) # not good
# canny_img = 255*canny(gray_img, 0).astype(float)
der.ar_info(canny_img, "Canny")
# der.np_to_pil(canny_img).show()

# filt_real, filt_imag = filters.gabor(gray_img, frequency=0.7)
# der.ar_info(filt_real, "Real")
# der.ar_info(filt_imag, "Imaginary")
# der.np_to_pil(filt_real).show()
# der.np_to_pil(filt_imag).show()

# hessian_filt = filters.hessian(gray_img)
# der.ar_info(hessian_filt, "Hessian")
# hessian_img = (hessian_filt * 255).astype("uint8")
# der.np_to_pil(hessian_img).show()

# roberts_edges = filters.roberts(gray_img)
# der.ar_info(roberts_edges, "Roberts")
# roberts_img = (roberts_edges * 255).astype("uint8")
# der.np_to_pil(roberts_img).show()

# sobel_edges = filters.sobel(gray_img)
# der.ar_info(sobel_edges, "Sobel")
# sobel_img = (sobel_edges * 255).astype("uint8")
# der.np_to_pil(sobel_img).show()

# scharr_edges = filters.scharr(gray_img)
# der.ar_info(scharr_edges, "Scharr")
# scharr_img = (scharr_edges * 255).astype("uint8")
# der.np_to_pil(scharr_img).show()

# prewitt_edges = filters.prewitt(gray_img)
# der.ar_info(prewitt_edges, "Prewitt")
# prewitt_img = (prewitt_edges * 255).astype("uint8")
# der.np_to_pil(prewitt_img).show()

from skimage import data, io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt


# img = data.coffee()

labels1 = segmentation.slic(orig_img, compactness=10, n_segments=800)
segments_img = color.label2rgb(labels1, orig_img, kind='avg')
der.ar_info(segments_img, "Segments")
der.np_to_pil(segments_img).save(folder + "00-KMEANS-SEGMENTS.jpg")

g = graph.rag_mean_color(orig_img, labels1)
labels2 = graph.cut_threshold(labels1, g, 9)
rag_thresh_img = color.label2rgb(labels2, orig_img, kind='avg')
der.ar_info(rag_thresh_img, "RAG Threshold")
der.np_to_pil(rag_thresh_img).save(folder + "00-RAG-THRESHOLD.jpg")

fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True)
ax[0].imshow(orig_img)
ax[1].imshow(segments_img)
ax[2].imshow(rag_thresh_img)

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

# from skimage import color
# color.rgb2hed()

# block_shape = (64, 64)
# view = view_as_blocks(gray_img, block_shape=(2,2))
# der.ar_info(view)

end = datetime.datetime.now()
delta = end - start
print(str(delta))

# der.do_filters(3)
