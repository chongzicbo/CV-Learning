import os, cv2
import imghdr
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTFigure, LTLayoutContainer, LTTextBoxHorizontal

file_path = r"E:\data\test\pdf2img\sustainability-15-01881-v2.pdf"  # PDF 文件路径
pic_basepath = r"E:\data\test\pdf2img\imgs"
if not os.path.exists(pic_basepath):
    os.makedirs(pic_basepath)
num = 1
for page_layout in extract_pages(file_path):
    for element in page_layout:
        # print(element.__class__)

        if isinstance(element, LTFigure):
            # print(element.bbox)
            for i in element._objs:
                if hasattr(i, "stream"):
                    res = imghdr.what(None, i.stream.rawdata)
                    if res:
                        pic_name = os.path.join(pic_basepath, f"seg{num}.jpg")
                        with open(pic_name, 'wb') as f:
                            f.write(i.stream.rawdata)
                        num += 1
                        # cv2.imshow("cnt", i.stream.rawdata)
                        # cv2.waitKey()
                    else:
                        print("res is None")
                        print(i.stream.attrs.get('Subtype'))
