import cv2
import numpy as np
import scipy
from skimage import measure

Max_len_of_one_lung = 600
AreaThreshold = 50


def find_lung_contours(img):
    np.clip(img, 0, 255, out=img)
    img = img.astype('uint8')
    ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # sort according to area
    contours.sort(key=lambda x: int(len(x)), reverse=True)
    return contours


def find_contors_center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return [cX, cY]


def find_lung_center(img):
    # print('find_lung_center shape ', img.shape)
    contours = find_lung_contours(img)
    isErode = False
    if (len(contours) > 0):
        if (len(contours[0]) > Max_len_of_one_lung):
            isErode = True
    # If left and right lungs together, erode
    if (isErode):
        # erode size 20*20 is upto experience
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        eroded = cv2.erode(img, kernel)
        contours = find_lung_contours(eroded)

    # If the left and right lungs contain multiple areas
    Left_lung_contours = []
    Right_lung_contours = []

    for i in range(len(contours)):
        contour = contours[i]
        if (len(contour) > AreaThreshold):
            cX, cY = find_contors_center(contour)
            if (cX < 512 / 2):
                Left_lung_contours.append(contour)
            else:
                Right_lung_contours.append(contour)
    # Respectively, take the left and right of the two largest areas
    Left_contours = []
    Right_contours = []
    if (len(Left_lung_contours) > 1):
        Left_contours = np.concatenate((Left_lung_contours[0], Left_lung_contours[1]), axis=0)
    elif (len(Left_lung_contours) > 0):
        Left_contours = Left_lung_contours[0]

    if (len(Right_lung_contours) > 1):
        Right_contours = np.concatenate((Right_lung_contours[0], Right_lung_contours[1]), axis=0)
    elif (len(Right_lung_contours) > 0):
        Right_contours = Right_lung_contours[0]

    # init value
    if len(Left_contours) == 0:
        left_x = int(512 / 4.0)
        left_y = int(512 / 2.0)
    else:
        left_x, left_y = find_contors_center(Left_contours)
    if len(Right_contours) == 0:
        right_x = int(512 / 4.0 * 3.0)
        right_y = int(512 / 2.0)
    else:
        right_x, right_y = find_contors_center(Right_contours)
    return (int(left_x), int(left_y)), (int(right_x), int(right_y))


def getNoduleLocation(lung_mask, nodule_center):
    '''
    :param mask:
    :param nodule_center: np.array(x,y) in voxel coord
    :return: nodule location:
    '''
    left_center, right_center = find_lung_center(lung_mask)

    # get nodule position
    # 实际的方向与图像是反着的
    left_or_right = u'右'
    up_or_down = u'下'
    left2center = np.linalg.norm(np.array(left_center) - np.array(nodule_center))
    right2center = np.linalg.norm(np.array(right_center) - np.array(nodule_center))
    if left2center > right2center:
        left_or_right = u'左'
        if nodule_center[1] > right_center[1]:
            up_or_down = u'上'

    else:
        if nodule_center[1] > left_center[1]:
            up_or_down = u'上'
    return left_or_right, up_or_down


def lobe_locate_gmm(nodule, case, mask, spacing, left_gmm, right_gmm):
    # mask = np.transpose(mask)
    # print(mask.shape,nodule)
    x, y, z = nodule
    z = min(int(z), int(mask.shape[0]) - 1)
    mask_slice = mask[z,:, :]

    mask_slice[mask_slice > 0] = 255
    # print(mask_slice.shape)
    leftOrRight = u'左'
    nodule_center = np.array([x, y])
    right_center, left_center = find_lung_center(mask_slice)
    left2center = np.linalg.norm(np.array(left_center) - np.array(nodule_center))
    right2center = np.linalg.norm(np.array(right_center) - np.array(nodule_center))

    if left2center > right2center:
        leftOrRight = u'右'
    # locate lobe
    if leftOrRight == u'左':
        zz, yy, xx = np.where(mask == 4)
    else:
        zz, yy, xx = np.where(mask == 3)
    # print (zz)
    lung_coords = np.array([xx, yy, zz])
    nodule_coord = np.array([x, y, z]).reshape(3, 1)
    nodule_relative = np.sum(lung_coords < nodule_coord, axis=1) * 1.0 / lung_coords.shape[1]
    # print(nodule_relative)
    nodule_relative[2] = 1 - nodule_relative[2]
    # nodule_relative[1] = 1 - nodule_relative[1]
    # nodule_relative[0] = 1 - nodule_relative[0]
    if not len(xx) == 0:  # we can find this lung
        if leftOrRight == u'左':
            p = left_gmm.predict_proba(nodule_relative.reshape(1, -1))[0]
            # print(p)
            if p[0] > p[1]:
                upOrDown = u'上'
            else:
                upOrDown = u'下'
            if max(p) < 0.65:
                upOrDown = ''

        else:
            p = right_gmm.predict_proba(nodule_relative.reshape(1, -1))[0]
            # print(p)
            if p[0] > p[1]:
                if p[0] > p[2]:
                    upOrDown = u'上'
                else:
                    upOrDown = u'下'
            else:
                if p[1] > p[2]:
                    upOrDown = u'中'
                else:
                    upOrDown = u'下'
            if max(p) < 0.6:
                upOrDown = ''
            # center = np.mean(zz)
            # upOrDown = 'up' if z > center else 'down'
    else:  # we can't find this lung
        return leftOrRight, "", -1

    return leftOrRight, upOrDown


# #
# from prepare import *
# from read_dicom import load_dicom, read_data
# # from scipy.ndimage.interpolation import zoom
#
#
# if __name__ == '__main__':
#     case, orgin, spacing = load_dicom(
#         '/home/wly/1.2.840.113704.9.1000.16.1.2019050909013658800020001/')
#     im, m1, m2 = read_data(case, spacing)
#     zero_im =np.zeros(shape=im.shape)
#     dm1 = process_mask(m1)
#     dm2 = process_mask(m2)
#     zero_im[dm1] = 4
#     zero_im[dm2] = 3
#     Mask = zero_im

    # clean, Mask2 = savenpy(im, m1, m2, spacing)
    # # 86, 150, 155)
    # # 140, 350, 220)
    # weizhi = lobe_locate_gmm(clean[0],Mask , 86, 150, 155)
    # print(weizhi)
