import cv2
import numpy as np
import skimage.io as io
from skimage import feature
import matplotlib.pyplot as plt

search_range = 500


def template_matching_ncc(src, temp):
    _, w = src.shape
    _, wt = temp.shape

    score = np.empty((w - wt))

    src = np.array(src, dtype="float")
    temp = np.array(temp, dtype="float")

    for dx in range(0, w - wt):
        # roi
        roi = src[:, dx:dx + wt]
        # NCC
        num = np.sum(roi * temp)
        den = np.sqrt((np.sum(roi ** 2))) * np.sqrt(np.sum(temp ** 2))
        if den == 0:
            score[dx] = 0
        score[dx] = num / den

    pt = np.unravel_index(score.argmax(), score.shape)

    return pt


def match(imL, imR, win_size, stride):
    h, w = imL.shape
    wh, ww = win_size
    m = (wh-1)//2
    n = (ww-1)//2
    imR = cv2.copyMakeBorder(imR, top=m, bottom=m, left=n, right=n,
                                     borderType=cv2.BORDER_CONSTANT, value=0)
    imL = cv2.copyMakeBorder(imL, top=m, bottom=m, left=n, right=n,
                                    borderType=cv2.BORDER_CONSTANT, value=0)
    disparity=[]
    for dy in range(0, h-m, stride):
        disparity_row = []
        for dx in range(0, w-n, stride):
            win = imR[dy:dy+wh, dx:dx+ww]
            search_aera = imL[dy:dy+wh, dx:min(dx+search_range+ww, w+ww)]
            # d = template_matching_ncc(search_aera, win)
            tmp = feature.match_template(search_aera, win)
            disparity_row.append(np.argmax(tmp))
        disparity.append(disparity_row)
        print('dy={:} \r'.format(dy))
    return disparity


def disp2depth(disp):
    baseline = 237.604
    f = 3962.004
    doffs = 107.911
    Z = baseline * f / (disp + doffs)
    return Z


def main():
    imageL = cv2.imread('Classroom1-perfect/im0.png', 0)
    imageR = cv2.imread('Classroom1-perfect/im1.png', 0)
    # imageL = cv2.imread('dataset/024_L.png', 0)
    # imageR = cv2.imread('dataset/024_R.png', 0)
    disp = match(imageL, imageR, [51,51], 2)
    disp = np.array(disp)
    disp = np.squeeze(disp)
    plt.imshow(disp, 'gray')
    Z = disp2depth(disp)
    Z = 2*Z/(Z.max()-Z.min())-1
    io.imsave('tmp.png', Z)
    n=0


