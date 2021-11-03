from model import OcrHandle
import base64
from PIL import Image
from io import BytesIO

ocrhandle = OcrHandle()


# 压缩图片尺寸，0～9控制压缩比例
def condense_image(image_path):
    import cv2

    src = cv2.imread(image_path, 1)
    cv2.imwrite(image_path, src, [cv2.IMWRITE_PNG_COMPRESSION, 9])


# 图片转base64
def get_base64(image_path):
    import base64
    with open(image_path, "rb") as f:  # 转为二进制格式
        base64_data = base64.b64encode(f.read())  # 使用base64进行加密
        return base64_data


# 图片文字识别
def get_res(image_path):
    im = Image.open(image_path)  # 返回一个Image对象
    short_size = min(im.size[0], im.size[1])
    img_b64 = get_base64(image_path)
    raw_image = base64.b64decode(img_b64)
    img = Image.open(BytesIO(raw_image))

    img = img.convert("RGB")
    res = ocrhandle.text_predict(img, 32 * (short_size // 32))
    for i, r in enumerate(res):
        rect, txt, confidence = r
        print(txt)
    return res


if __name__ == '__main__':
    image_path = "Xnip2021-11-03_13-54-05.png"
    get_res(image_path)
