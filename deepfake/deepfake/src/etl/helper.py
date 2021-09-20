import numpy as np
import cv2
import gc


def get_frames(vidfile, num_frames=-1, resize=None):
    cap = cv2.VideoCapture(vidfile)
    ret, frame = cap.read()
    # Longest side = resize
    if resize:
        if np.argmax(frame.shape) == 0:
            resize_hw = (resize, int(frame.shape[1] * resize / frame.shape[0]))
        elif np.argmax(frame.shape) == 1:
            resize_hw = (int(frame.shape[0] * resize / frame.shape[1]), resize)
        frame = cv2.resize(frame, resize_hw[::-1])
    frames = [frame]
    while ret and len(frames) < num_frames:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, resize_hw[::-1]) if resize else frame
            frames.append(frame)
    cap.release()
    gc.collect()
    return frames, list(range(num_frames))


def scale_bbox(boxes, scale_factor):
    # boxes.shape = [N, 4]
    # box format: [x1, y1, x2, y2]
    scaled = []
    for box in boxes:
        scaled.append([int(scale_factor * coord) for coord in box])
    return np.asarray(scaled)


def enlarge_box(boxes, img_shape, scale=1.1):
    # boxes.shape = [N, 4]
    # box format: [x1, y1, x2, y2]
    # w = max width
    # h = max height
    enlarged = []
    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        w *= scale
        h *= scale
        xc = (x2 + x1) / 2
        yc = (y2 + y1) / 2
        x1 = np.max((0, xc - w / 2))
        y1 = np.max((0, yc - h / 2))
        x2 = np.min((img_shape[1], xc + w / 2))
        y2 = np.min((img_shape[0], yc + h / 2))    
        enlarged.append([int(x1), int(y1), int(x2), int(y2)])
    return np.asarray(enlarged)


def convert_coords(boxes):
    # boxes.shape = [N, 4]
    # box format: [x1, y1, x2, y2]    
    # Convert from [x1, y1, x2, y2] to [xc, yc, w, h, class, conf] 
    # for ensemble box function
    converted = []
    for box in boxes:
        x1, y1, x2, y2 = box
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1
        converted.append([xc, yc, w, h, 1, 1])
    return np.asarray(converted)


def intify(box): return tuple([int(coord) for coord in box])