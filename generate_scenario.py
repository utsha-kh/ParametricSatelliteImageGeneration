import data.airbus_loader
import numpy as np
import PIL
import cv2
import os

def move_ship(im, label, angle):
    im = np.array(im)
    (im_h, im_w) = im.shape[: 2]
    bbx = (np.array(label[:-1]) * [im_w, im_h, im_w, im_h]).astype(np.int)
    prev_angle = label[-1]
    if angle != prev_angle:
        im, bbx = rotate_ship(im, bbx, prev_angle - angle)
    else:
        x_shift = np.around(10 * np.cos(angle * np.pi / 180)).astype(np.int)
        y_shift = -np.around(10 * np.sin(angle  * np.pi / 180)).astype(np.int)
        x0 = max(0, int(bbx[0] - np.abs(x_shift)))
        x1 = min(im.shape[0] - 1, int(bbx[0] + bbx[2] + np.abs(x_shift)))
        y0 = max(0, int(bbx[1] - y_shift))
        y1 = min(im.shape[1] - 1, int(bbx[1] + bbx[3] + np.abs(y_shift)))
        im_slice = im[y0 : y1, x0 : x1]
        im_slice = np.roll(im_slice, y_shift, axis=0)
        im_slice = np.roll(im_slice, x_shift, axis=1)
        im[y0 : y1, x0 : x1] = im_slice
        bbx[0] = bbx[0] + x_shift
        bbx[1] = bbx[1] + y_shift
    bbx = bbx / [im_w, im_h, im_w, im_h]
    label = np.append(bbx, angle)
    return PIL.Image.fromarray(im), label

def rotate_ship(im, bbx, angle, border=0):
    ship = im.copy()[bbx[1] - border: bbx[1] + bbx[3] + border, 
                     bbx[0] - border : bbx[0] + bbx[2] + border]

    (h, w) = ship.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    new_ship = cv2.warpAffine(ship, M, (nW, nH))

    center_in_im = (int(bbx[0] + bbx[2] / 2), int(bbx[1] + bbx[3] / 2))
    x0 = int(center_in_im[0] - nW / 2)
    x1 = x0 + new_ship.shape[1]
    y0 = int(center_in_im[1] - nH / 2)
    y1 = y0 + new_ship.shape[0]

    mask = np.zeros(im.shape, dtype=np.uint8)
    mask[y0 : y1, x0 : x1] = new_ship

    im[np.where(mask != [0, 0, 0])] = mask[np.where(mask != [0, 0, 0])]

    bbx = np.array([x0, y0, nW, nH])
    return im, bbx

label_path = '/data/airbus/train_ship_segmentations_v2.csv'
im_dir = '/data/airbus/images'

ab = data.airbus_loader.AirbusDataset(im_dir, label_path)

im_id = '00113a75c.jpg'
all_labels = ab.labels
labels = all_labels[all_labels['ImageId'] == im_id]

im_path = os.path.join(im_dir, im_id)
with open(im_path, 'rb') as f:
    with PIL.Image.open(f) as image:
        image = image.convert('RGB')

outdir = '/workspaces/LostGANs/vis/'
if not os.path.exists(outdir): os.makedirs(outdir)

im = np.array(image.copy())
angles = [320, 320, 320, 320, 310, 310, 310, 300, 290, 290, 290, 290, 290, 290, 290, 270, 270]
bbx = labels.iloc[4]['BoundingBox']
label = np.append(bbx, 320)
for step, angle in enumerate(angles):
    im, label = move_ship(im, label, angle)
    im.save(os.path.join(outdir, f'{step}.png'))

PIL.Image.fromarray(im)


