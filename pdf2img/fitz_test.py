import fitz
import re
import os

import fitz
import re
import os
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

file_path = r'C:\xxx\xxx.pdf'  # PDF 文件路径
dir_path = r'C:\xxx'  # 存放图片的文件夹


def pdf2image1(path, pic_path):
    checkIM = r"/Subtype(?= */Image)"
    pdf = fitz.open(path)
    lenXREF = pdf.xref_length()
    count = 1
    for i in range(1, lenXREF):
        text = pdf.xref_object(i)
        isImage = re.search(checkIM, text)
        if not isImage:
            continue
        pix = fitz.Pixmap(pdf, i)
        new_name = f"img_{count}.png"
        pix.save(os.path.join(pic_path, new_name))
        count += 1
        pix = None


def pdf2image1(path, pic_path):
    checkIM = r"/Subtype(?= */Image)"
    pdf = fitz.open(path)
    lenXREF = pdf.xref_length()
    count = 1
    for i in range(1, lenXREF):
        text = pdf.xref_object(i)
        isImage = re.search(checkIM, text)
        if not isImage:
            continue
        pix = fitz.Pixmap(pdf, i)
        if pix.size < 10000:  # 在这里添加一处判断一个循环
            continue  # 不符合阈值则跳过至下
        new_name = f"img_{count}.png"
        pix.save(os.path.join(pic_path, new_name))
        count += 1
        pix = None


def extract_pic_info(filepath, pic_dirpath):
    """
    提取PDF中的图片
    @param filepath:pdf文件路径
    @param pic_dirpath:要保存的图片目录路径
    @return:
    """
    if not os.path.exists(pic_dirpath):
        os.makedirs(pic_dirpath)
    # 使用正则表达式来查找图片
    check_XObject = r"/Type(?= */XObject)"
    check_Image = r"/Subtype(?= */Image)"
    img_count = 0

    """1. 打开pdf，打印相关信息"""
    pdf_info = fitz.open(filepath)
    # 1.16.8版本用法 xref_len = doc._getXrefLength()
    # 最新版本
    xref_len = pdf_info.xref_length()
    # 打印PDF的信息
    print("文件名：{}, 页数: {}, 对象: {}".format(filepath, len(pdf_info), xref_len - 1))

    """2. 遍历PDF中的对象，遇到是图像才进行下一步，不然就continue"""
    for index in range(1, xref_len):
        # 1.16.8版本用法 text = doc._getXrefString(index)
        # 最新版本
        text = pdf_info.xref_object(index)

        is_XObject = re.search(check_XObject, text)
        is_Image = re.search(check_Image, text)
        # 如果不是对象也不是图片，则不操作
        if is_XObject or is_Image:
            img_count += 1
            # 根据索引生成图像
            pix = fitz.Pixmap(pdf_info, index)
            pic_filepath = os.path.join(pic_dirpath, 'img_' + str(img_count) + '.png')
            """pix.size 可以反映像素多少，简单的色素块该值较低，可以通过设置一个阈值过滤。以阈值 10000 为例过滤"""
            if pix.size < 10000:
                continue

            """三、 将图像存为png格式"""
            if pix.n >= 5:
                # 先转换CMYK
                pix = fitz.Pixmap(fitz.csRGB, pix)

            # 存为PNG
            pix.save(pic_filepath)


from pdf2image import convert_from_path


def extract_images_by_paddleocr(pdf_path, pdf_page_dir, save_folder):
    convert_from_path(
        pdf_path=pdf_path,  # 要转换的pdf的路径
        dpi=200,  # dpi中的图像质量（默认200）
        output_folder=pdf_page_dir,  # 将生成的图像写入文件夹（而不是直接写入内存）#注意中文名的目录可能会出问题
        first_page=None,  # 要处理的第一页
        last_page=None,  # 停止前要处理的最后一页
        fmt="png",  # 输出图像格式
        jpegopt=None,  # jpeg选项“quality”、“progressive”和“optimize”（仅适用于jpeg格式）
        thread_count=4,  # 允许生成多少线程进行处理
        userpw=None,  # PDF密码
        use_cropbox=False,  # 使用cropbox而不是mediabox
        strict=False,  # 当抛出语法错误时，它将作为异常引发
        transparent=False,  # 以透明背景而不是白色背景输出。
        single_file=False,  # 使用pdftoppm/pdftocairo中的-singlefile选项
        poppler_path=None,  # 查找poppler二进制文件的路径
        grayscale=False,  # 输出灰度图像
        size=None,  # 结果图像的大小，使用枕头（宽度、高度）标准
        paths_only=False,  # 不加载图像，而是返回路径（需要output_文件夹）
        use_pdftocairo=False,  # 用pdftocairo而不是pdftoppm，可能有助于提高性能
        timeout=None,  # 超时
    )
    import datetime
    import os
    import fitz
    import cv2
    import shutil
    from paddleocr import PPStructure, draw_structure_result, save_structure_res
    table_engine = PPStructure(show_log=True)
    # save_folder = './result' # 图片保存地址
    img_dir = pdf_page_dir  # pdf转换为img图片的地址
    files = os.listdir(img_dir)
    for fi in files:
        fi_d = os.path.join(img_dir, fi)
        # print(fi_d)
        # for img in os.listdir(fi_d):
        # img_path = os.path.join(fi_d, img)
        img = cv2.imread(fi_d)
        result = table_engine(img)
        # 保存在每张图片对应的子目录下
        save_structure_res(result, save_folder, fi)


def extract_images_by_pdfminer(pdf_path: str, img_save_dir: str):
    # 打开一个pdf，使用二进制读取文件，读进来的是bytes类型
    pdf0 = open(pdf_path, 'rb')
    # 创建一个PDFParser对象
    parser = PDFParser(pdf0)
    # 这里可以输入文档密码，官方是doc=PDFDocument(parser,password),我没有密码，就没输入
    doc = PDFDocument(parser)
    parser.set_document(doc)

    # 这四行代码的目的就是初始化一个interpreter ，用于后面解析页面
    # pdf资源管理器
    resources = PDFResourceManager()
    # 参数分析器
    laparam = LAParams()
    # 聚合器
    device = PDFPageAggregator(resources, laparams=laparam)
    # 页面解释器
    interpreter = PDFPageInterpreter(resources, device)

    # 这里可以拆开获取pdf的每一页
    # PDFPage.create_pages(doc)的返回值是generator类型，可以用for来遍历
    pages = []
    for page in PDFPage.create_pages(doc):
        pages.append(page)

    # 准备把page解析出来的东西存一下，方便后面用
    texts = []
    images = []

    for page in pages:
        interpreter.process_page(page)  # 解析page
        layout = device.get_result()  # 获得layout，layout可遍历

        # 遍历layout，layout里面就是要拆的东西了
        for out in layout:
            if isinstance(out, LTTextBox):
                texts.append(out)
            if isinstance(out, LTImage):
                images.append(out)
            # 当是figure类型时，需要取出它里面的东西来。figure可遍历，所以for循环取。
            # 如果figure里面还是figure，就接着遍历(虽然我没见过多层figure的情况)
            if isinstance(out, LTFigure):
                figurestack = [out]
                while figurestack:
                    figure = figurestack.pop()
                    for f in figure:
                        if isinstance(f, LTTextBox):
                            texts.append(f)
                        if isinstance(f, LTImage):
                            images.append(f)
                        if isinstance(f, LTFigure):
                            figurestack.append(f)

    # 图片及图片另存：
    i = 0  # 文件名编号

    for image in images:
        with open(os.path.join(img_save_dir, 'pic_{}.jpg'.format(i)), 'wb') as f:
            f.write(image.stream.get_data())


if __name__ == '__main__':
    path = r"E:\data\test\pdf2img\cancers-15-00634-v2.pdf"
    save_path = r"E:\data\test\pdf2img\imgs2"
    ocr_saved_dir = r"E:\data\test\pdf2img\paddleocr_images_04"
    pdfminer_saved_dir = r"E:\data\test\pdf2img\imgs4"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(ocr_saved_dir):
        os.makedirs(ocr_saved_dir)
    if not os.path.exists(pdfminer_saved_dir):
        os.makedirs(pdfminer_saved_dir)

    # pdf2image1(path, save_path)
    # extract_pic_info(path, save_path)
    extract_images_by_paddleocr(path, save_path, ocr_saved_dir)

    # extract_images_by_pdfminer(path, pdfminer_saved_dir)
