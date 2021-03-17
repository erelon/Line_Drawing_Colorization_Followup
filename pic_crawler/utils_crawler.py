import multiprocessing
import os
from argparse import ArgumentParser
import tarfile
import urllib
from time import sleep

import cv2
import urllib3
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from posixpath import basename

import threading
import concurrent.futures

from color_cleaner import clean_color

max_thread = 10


def downloadPic(url, folderName):
    try:
        response = urllib.request.urlopen(url, timeout=10)
        soup = BeautifulSoup(response.read(), "html.parser")

        picObj = soup.find('div', id='content')
        linkObj = picObj.find('img')
        imgUrl = urljoin(url, linkObj['src'])

        outputName = '%s/%s.jpg' % (folderName, basename(url))
        urllib.request.urlretrieve(imgUrl, outputName)
        clean_color(outputName, folderName)
        os.remove(outputName)
        print("OK: " + url)
    except urllib.error.HTTPError as err:
        print("FAIL: " + url + " " + str(err))
    except cv2.error as err:
        print("FAIL: " + url + " " + str(err))
        os.remove(outputName)
    except Exception as exp:
        print("FATAL FAIL: " + str(exp))


def tarify(folderName):
    all_gts = os.listdir(f"{folderName}/GT")
    all_train_data = os.listdir(f"{folderName}/train_data")
    if len(all_gts) == 0 or len(all_train_data) == 0:
        return
    if len(all_gts) > len(all_train_data):
        all_gts = all_gts[:len(all_train_data)]
    elif len(all_train_data) > len(all_gts):
        all_train_data = all_train_data[:len(all_gts)]
    take_only = len(all_gts) - len(all_gts) % 10
    all_gts = all_gts[:take_only]
    all_train_data = all_train_data[:take_only]
    train_num = int(take_only * 0.8)

    all_gts_train = all_gts[:train_num]
    all_gts_test = all_gts[train_num:]
    all_train_data_train = all_train_data[:train_num]
    all_train_data_test = all_train_data[train_num:]

    def to_tar(file_name, data_to_save):
        tar_file = tarfile.open(folderName + "/" + file_name + ".tar", mode="a")
        if "GT" in file_name:
            folder = "GT"
        else:
            folder = "train_data"
        for file in data_to_save:
            tar_file.add(folderName + "/" + folder + "/" + file, file[file.rfind("\\") + 1:])
            os.remove(folderName + "/" + folder + "/" + file)
        tar_file.close()

    for data, tar_file in zip([all_gts_train, all_gts_test, all_train_data_train, all_train_data_test],
                              ["GT_train", "GT_test", "train_data_train", "train_data_test"]):
        to_tar(tar_file, data)

    return


def scrape(base_url, folderName):
    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_thread)
    ths = []
    i = 0

    for curr_img in reversed(range(0, 3192000)):
        if os.path.isfile(os.path.join(folderName, '%d.jpg' % curr_img)):
            continue
        if i % 10 == 0:
            sleep(1)
            th = thread_pool.submit(tarify, folderName)
            th.result()
        # sema.acquire(True)
        url = '%s%d' % (base_url, curr_img)
        th = thread_pool.submit(downloadPic, url, folderName)
        th.result()
        # ths.append(th)
        i += 1

    # wait for the threads to finish
    thread_pool.shutdown()


def parseCLI():
    desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)

    parser = ArgumentParser(description=desc)

    parser.add_argument('-u', '--url', type=str, dest='url', default="https://www.zerochan.net/full/",
                        help='URL of the website to crawl in')
    parser.add_argument('-f', '--folderName', type=str, dest='folderName', default="out",
                        help='Crawl output folder')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseCLI()

    # make the directory for the created files
    os.makedirs(args.folderName, exist_ok=True)
    os.makedirs(args.folderName + "/GT", exist_ok=True)
    os.makedirs(args.folderName + "/train_data", exist_ok=True)

    scrape(args.url, args.folderName)

    print("done")
