import os
from threading import Lock

import cv2
import numpy as np



def show_image(pic):
    cv2.imshow("p", pic)
    cv2.waitKey(0)


def distance_in_percentage(x, y):
    if (x / y) >= 1:
        return (x / y) - 1
    else:
        return (y / x) - 1


def clean_white_background(pic):
    # Convert to gray, and threshold
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    # Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    # Crop and and try to create square if possible
    x, y, w, h = cv2.boundingRect(cnt)
    if w < h:
        d = int(min(h - w, x * 2) / 2)
        x -= d
        w += 2 * d
    else:
        d = int(min(w - h, y * 2) / 2)
        y -= d
        h += 2 * d
    dst = pic[y:y + h, x:x + w]
    return dst


def resize_pic(pic: np.ndarray, res=256, squaring_threshold=0.1) -> list:
    pic = clean_white_background(pic)
    if min(pic.shape[0], pic.shape[1]) < 256:
        # This image is too small for our needs
        return []
    pics = []
    if pic.shape[0] == pic.shape[1]:
        # Normal resize
        resized_pic = cv2.resize(pic, (res, res), interpolation=cv2.INTER_AREA)
        pics.append(resized_pic)
    elif distance_in_percentage(pic.shape[0], pic.shape[1]) <= squaring_threshold:
        # We resize in the normal way, even that the original image is not exactly squared
        resized_pic = cv2.resize(pic, (res, res), interpolation=cv2.INTER_AREA)
        pics.append(resized_pic)
    elif distance_in_percentage(pic.shape[0], pic.shape[1]) <= squaring_threshold * 2:
        # TODO: Maybe off scale a bit for more image coverage
        # We don't want to window with this small overlap
        scale_to = 256 / (min(pic.shape[0], pic.shape[1]))
        resized_pic = cv2.resize(pic, dsize=(0, 0), fx=scale_to, fy=scale_to, interpolation=cv2.INTER_AREA)
        # Crop the middle of the image
        x = int((resized_pic.shape[0] / 2) - 0.5 * res)
        y = int((resized_pic.shape[1] / 2) - 0.5 * res)
        cropped_pic = resized_pic[x:x + res, y:y + res]
        pics.append(cropped_pic)
    else:
        # We need to resize, and then create picture windows
        scale_to = 256 / (min(pic.shape[0], pic.shape[1]))
        resized_pic = cv2.resize(pic, dsize=(0, 0), fx=scale_to, fy=scale_to, interpolation=cv2.INTER_AREA)

        # Crop 256*256 boxes
        for y in range(0, resized_pic.shape[0], res):
            for x in range(0, resized_pic.shape[1], res):
                pic_window = resized_pic.copy()
                boxed_pic = pic_window[x:x + res, y:y + res]
                pics.append(boxed_pic)
        # Now we want the last box in the proper size:
        pics = pics[:-1]
        pic_window = resized_pic.copy()
        boxed_pic = pic_window[pic_window.shape[0] - res:, pic_window.shape[1] - res:]
        pics.append(boxed_pic)

    return pics


def load_pic(pic_path):
    pic = cv2.imread(pic_path)
    return pic


def remove_low_colored_pics(pics, bw_threshold=20):
    colorful_pics = []
    for pic in pics:
        pic_hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
        # is the picture colorful enough? Check the mean value of the saturation.
        mean, std = cv2.meanStdDev(pic_hsv)
        if mean[1] < bw_threshold:
            continue
        colorful_pics.append(pic)
    return colorful_pics


def linerize1(pics):
    """
    This is the way to binaries the pictures as introduced in the original paper
    :param pics: The images to linearize
    :return: Linearized pics
    """
    line_pics = []
    neiborhood8 = np.ones((3, 3), dtype=np.uint8)
    for pic in pics:
        gray_form = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        # img_dilate = cv2.erode(gray_form, neiborhood8, iterations=2)
        img_dilate = cv2.dilate(gray_form, neiborhood8, iterations=1)

        img_diff = cv2.absdiff(gray_form, img_dilate)
        img_diff = cv2.multiply(img_diff, 3)
        img_line = cv2.bitwise_not(img_diff)

        # img_line = cv2.adaptiveThreshold(img_line, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 8)
        img_binary = cv2.adaptiveThreshold(img_line, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8)

        line_pics.append(img_binary)
    return line_pics


def linerize(pics):
    """
    This is the way to binaries the pictures as introduced in the original paper
    :param pics: The images to linearize
    :return: Linearized pics
    """
    line_pics = []
    for pic in pics:
        gray_form = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(gray_form, (21, 21), 0, 0)
        img_blend = cv2.divide(gray_form, img_blur, scale=256)
        img_binary = cv2.adaptiveThreshold(img_blend, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8)
        img_line_tmp = np.zeros_like(img_binary)
        img_line_tmp[img_binary > 180] = 255
        img_line = cv2.cvtColor(img_line_tmp, cv2.COLOR_GRAY2RGB)
        line_pics.append(img_line)
    return line_pics


def clean_color(pic_path, save_path) -> None:
    """
    Takes an image, checks that the resolution is good and resize it to 256 on at least one of the axes, then crops it
    to 256X256 chunks and picks a chunk if it colorful enough. Then the chunk is cleaned from color, and saved in it's
    colored form as ground truth (in save_path/GT/ folder) and the uncolored form as train data
    (in save_path/train_data/ folder).
    :param pic_path: The path of the original image
    :param save_path: The path of GT and train data folders for saving the output
    :return: None
    """
    pic = load_pic(pic_path)
    resize_pics = resize_pic(pic)
    only_colorful_pics = remove_low_colored_pics(resize_pics)

    for i, (pic) in enumerate(only_colorful_pics):
        gt_file_path = f"{save_path}/GT/{pic_path[pic_path.rindex('/') + 1:pic_path.rindex('.')]}_{str(i)}.gt.jpg"
        cv2.imwrite(gt_file_path, pic)

    lined_pics = linerize(only_colorful_pics)

    for i, (pic) in enumerate(lined_pics):
        train_file_path = f"{save_path}/train_data/{pic_path[pic_path.rindex('/') + 1:pic_path.rindex('.')]}_{str(i)}.train.jpg"
        cv2.imwrite(train_file_path, pic)
