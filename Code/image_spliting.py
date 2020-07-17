
from PIL import Image
import os


def split(image_path, out_name, outdir):
    """slice an image into parts slice_size wide"""
    img = Image.open(image_path)
    width, height = img.size
    upper = 0
    left = 0
    slice_size = width/4
    slices = 4

    count = 1
    for slice in range(slices):
        #if we are at the end, set the lower bound to be the bottom of the image
        if count == slices:
            right = width
        else:
            right = int(count * slice_size)  

        bbox = (left, upper, right, height)
        working_slice = img.crop(bbox)
        left += slice_size
        #save the slice
        working_slice.save(os.path.join(outdir, out_name + "_" + str(count)+".jpg"))
        count +=1