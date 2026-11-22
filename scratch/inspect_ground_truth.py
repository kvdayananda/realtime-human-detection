import cv2
from pathlib import Path
from ai.detection import HumanDetectorONNX
from ai.segmentation import HumanSegmenterONNX

det = HumanDetectorONNX()
seg = HumanSegmenterONNX()

items = [
    ('dataset/human detection dataset/1/0.png', 'front_facing_1'),
    ('dataset/human detection dataset/1/1.png', 'front_facing_2'),
    ('dataset/human detection dataset/1/2.png', 'casual_clothing'),
    ('dataset/human detection dataset/1/5.png', 'side_facing_1'),
    ('dataset/human detection dataset/1/10.png', 'outdoor_daylight'),
    ('dataset/human detection dataset/1/25.png', 'athletic_clothing'),
    ('dataset/human detection dataset/1/50.png', 'indoor_lighting'),
    ('dataset/human detection dataset/1/100.png', 'cluttered_background'),
    ('dataset/human detection dataset/0/0.png', 'negative_room'),
    ('dataset/human detection dataset/0/1.png', 'negative_furniture'),
    ('dataset/human detection dataset/0/2.png', 'negative_landscape'),
]

for img_path, name in items:
    im = cv2.imread(img_path)
    h, w = im.shape[:2]
    d = det.detect(im)
    s = seg.segment(im)
    print(f"=== {name} ({img_path}) {w}x{h} ===")
    for det_box in d['detections']:
        b = [round(v, 2) for v in det_box['bbox']]
        x1, y1, x2, y2 = det_box['bbox']
        norm_xc = round((x1 + x2) / (2 * w), 4)
        norm_yc = round((y1 + y2) / (2 * h), 4)
        norm_w = round((x2 - x1) / w, 4)
        norm_h = round((y2 - y1) / h, 4)
        print(f"  DET: bbox={b}, conf={det_box['confidence']:.3f}, norm_xywh=[0, {norm_xc}, {norm_yc}, {norm_w}, {norm_h}]")
    for s_box in s['segmentations']:
        print(f"  SEG: bbox={[round(v, 2) for v in s_box['bbox']]}, conf={s_box['confidence']:.3f}, mask_area={s_box['mask_area']}")
