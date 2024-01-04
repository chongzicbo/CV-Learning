from PIL import Image


def set_image_dpi_resize(image):
    """
    Rescaling image to 300dpi while resizing
    :param image: An image
    :return: A rescaled image
    """
    length_x, width_y = image.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    image_resize = image.resize(size, Image.ANTIALIAS)
    temp_filename = "./1.jpg"
    image_resize.save(temp_filename, dpi=(300, 300))
    return temp_filename


if __name__ == "__main__":
    image_path = "/data/bocheng/data/image_extract/PDFs/outputs/pypdf/Elsevier/1-s2.0-S0165178119307024-main/page-5_img-1_Im0.jpg"
    image = Image.open(image_path)
    set_image_dpi_resize(image)
