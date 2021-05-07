from torchvision.utils import save_image
import data.airbus_loader
import pandas as pd
import numpy as np
import PIL
import cv2
import os

def move_ship(im, label, angle, border=3):
    """
    Takes a given image and advances the pixels located inside the label 
    bounding box, rolling the pixels at the front of the bounding box to the
    rear of the bounding box to simulate motion.
    Args:
        im: The input image in either PIL.Image or np.ndarray format.
        label: The bounding box of the pixels to move, the bounding box is
            expected in coco style with (x0, y0, width, height), where x0 and y0
            are the top left corner and all values are scaled by the image
            dimensions to be in the range [0, 1].
        angle: The angle to advance the bounding box. Angles should be given in
            degrees with 0 degrees being rightword.
        border: The width of the border around the ship, increase border if
            portions of the object are getting clipped when moving.
    Returns:
        Pil.Image containing the shifted bounding box.
        Updated bounding box that matches the shift of the object.
    """
    im = np.array(im)
    (im_h, im_w) = im.shape[: 2]
    bbx = (np.array(label[:-1]) * [im_w, im_h, im_w, im_h]).astype(np.int)
    prev_angle = label[-1]
    if angle != prev_angle:
        im, bbx = rotate_ship(im, bbx, prev_angle - angle)
    else:
        x_shift = np.around(3 * np.cos(angle * np.pi / 180)).astype(np.int)
        y_shift = -np.around(3 * np.sin(angle  * np.pi / 180)).astype(np.int)
        x0 = max(0, int(bbx[0] - np.abs(x_shift)))
        x1 = min(im.shape[0] - 1, int(bbx[0] + bbx[2] + np.abs(x_shift)))
        y0 = max(0, int(bbx[1] - y_shift))
        y1 = min(im.shape[1] - 1, int(bbx[1] + bbx[3] + np.abs(y_shift)))
        im_slice = im[y0 - border: y1 + border, x0 - border : x1 + border]
        im_slice = np.roll(im_slice, y_shift, axis=0)
        im_slice = np.roll(im_slice, x_shift, axis=1)
        im[y0 - border : y1 + border, x0 - border : x1 + border] = im_slice
        bbx[0] = bbx[0] + x_shift
        bbx[1] = bbx[1] + y_shift
    bbx = bbx / [im_w, im_h, im_w, im_h]
    label = np.append(bbx, angle)
    return PIL.Image.fromarray(im), label

def rotate_ship(im, bbx, angle, border=1):
    """
    Performs a rotation on the pixels located in the given bounding box by 
    applying a rotation matrix to the values located within the bounding box.
    Args:
        im: The image to be modified
        bbx: The bounding box in the style [x0, y0, width, height] containng the
            pixels to be rotated.
        angle: The angle to rotate the pixels contained in the bounding box.
        border: The border to increase the bounding box size in each direction.
            Increase the border size if portions of the object are clipped
            during rotation.
    Returns:
        Modified image as np.ndarray
        New bounding box of the rotated object
    """
    ship = im.copy()[bbx[1] - border: bbx[1] + bbx[3] + border, 
                     bbx[0] - border : bbx[0] + bbx[2] + border]
    ship_mask = np.zeros(ship.shape, dtype=np.uint8)
    ship_mask[2:-2,2:-2] = 255
    
    (h, w) = ship.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    """Grab the rotation matrix (applying the negative of the angle to rotate 
        clockwise), then grab the sine and cosine (i.e., the rotation components
        of the matrix) """
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # Compute the eenlarged bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix to take translation into account
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    """Perform the rotation, warped_ship_mask is used to remove the non-zero
        values around the edges of the rotated image due to artifacts of the
        matrix multiplication that performs that transformation"""
    warped_ship = cv2.warpAffine(ship, M, (nW, nH))
    warped_ship_mask = cv2.warpAffine(ship_mask, M, (nW, nH))
    warped_ship[warped_ship_mask == [0, 0, 0]] = 0

    # Coordinates to place the rotated box back into the original image
    center_in_im = (int(bbx[0] + bbx[2] / 2), int(bbx[1] + bbx[3] / 2))
    x0 = int(center_in_im[0] - nW / 2)
    x1 = x0 + warped_ship.shape[1]
    y0 = int(center_in_im[1] - nH / 2)
    y1 = y0 + warped_ship.shape[0]

    # Mask is used to place only the non-zero ship pixels back into the image
    mask = np.zeros(im.shape, dtype=np.uint8)
    mask[y0 : y1, x0 : x1] = warped_ship

    # Only write non-zero ship pixels so as not to zero out ocean around ship
    im[np.where(mask != [0, 0, 0])] = mask[np.where(mask != [0, 0, 0])]

    bbx = np.array([x0, y0, nW, nH])
    return im, bbx

# Example of generating a scenario.
if __name__ == "__main__":
    label_path = '/data/airbus/train_ship_segmentations_v2.csv'
    im_dir = '/data/airbus/images'
    outdir = '/workspaces/LostGANs/vis/'

    ab = data.airbus_loader.AirbusDataset(im_dir, label_path)

    labels = ab.labels
    im_id = labels.iloc[20]['ImageId'] # Choose 20th label for example
    bboxs = labels[labels['ImageId'] == im_id]
    im_path =  os.path.join(im_dir, im_id)

    with open(im_path, 'rb') as f:
        with PIL.Image.open(f) as image:
            image = image.convert('RGB')
    if not os.path.exists(outdir): os.makedirs(outdir)
    image = image.resize([768,768])

    # generate sequence of angles that start the ship moving at angle 290 then
    # slightly change heading as scenario progresses
    im = np.array(image.copy())
    angle = 290
    angles = []
    for i in range(120):
        if i < 80:
            if i % 8 == 0:
                angle -= 1
        else:
            if i % 8 == 0:
                angle += 1
        angles.append(angle)
    bbx = bboxs['BoundingBox'].iloc[1]
    label = np.append(bbx, 290)
    for step, angle in enumerate(angles):
        im, label = move_ship(im, label, angle)
        res_im = im.resize([128,128])
        res_im.save(os.path.join(outdir, f'{step}.png'))
    