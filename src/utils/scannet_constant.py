PANOPTIC_SEMANTIC2NAME = {
    0: "unlabeled",
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "desk",
    14: "curtain",
    15: "refrigerator",
    16: "shower curtain",
    17: "toilet",
    18: "sink",
    19: "bathtub",
    20: "otherfurniture",
}
STUFF_CLASSES = [
    0,
    1,
]  # wall, floor for output is begin from 0 and the last is unlabeled
THING_CLASSES = list(range(2, 20))  # 18 classes
PANOPTIC_NAME2SEMANTIC = {v: k for k, v in PANOPTIC_SEMANTIC2NAME.items()}
PANOPTIC_SEMANTIC2CONTINUOUS = dict(
    zip(PANOPTIC_SEMANTIC2NAME.keys(), range(len(PANOPTIC_SEMANTIC2NAME)))
)
PANOPTIC_CONTINUOUS2SEMANTIC = dict(
    zip(range(len(PANOPTIC_SEMANTIC2NAME)), PANOPTIC_SEMANTIC2NAME.keys())
)
PANOPTIC_COLOR_PALLETE = {
    0: [0, 0, 0],  # unlabeled
    1: [174, 199, 232],  # wall
    2: [152, 223, 138],  # floor
    3: [31, 119, 180],  # cabinet
    4: [255, 187, 120],  # bed
    5: [188, 189, 34],  # chair
    6: [140, 86, 75],  # sofa
    7: [255, 152, 150],  # table
    8: [214, 39, 40],  # door
    9: [197, 176, 213],  # window
    10: [148, 103, 189],  # bookshelf
    11: [196, 156, 148],  # picture
    12: [23, 190, 207],  # counter
    13: [247, 182, 210],  # desk
    14: [219, 219, 141],  # curtain
    15: [255, 127, 14],  # refrigerator
    16: [158, 218, 229],  # shower curtain
    17: [44, 160, 44],  # toilet
    18: [112, 128, 144],  # sink
    19: [227, 119, 194],  # bathtub
    20: [82, 84, 163],  # otherfurn
}
PANOPTIC_SEMANTIC2NAME.pop(0)

INSTANCE_SEMANTIC2NAME = {
    0: "unlabeled",
    1: "cabinet",
    2: "bed",
    3: "chair",
    4: "sofa",
    5: "table",
    6: "door",
    7: "window",
    8: "bookshelf",
    9: "picture",
    10: "counter",
    11: "desk",
    12: "curtain",
    13: "refrigerator",
    14: "shower curtain",
    15: "toilet",
    16: "sink",
    17: "bathtub",
    18: "otherfurniture",
}
INSTANCE_NAME2SEMANTIC = {v: k for k, v in INSTANCE_SEMANTIC2NAME.items()}
INSTANCE_SEMANTIC2CONTINUOUS = dict(
    zip(INSTANCE_SEMANTIC2NAME.keys(), range(len(INSTANCE_SEMANTIC2NAME)))
)
INSTANCE_CONTINUOUS2SEMANTIC = dict(
    zip(range(len(INSTANCE_SEMANTIC2NAME)), INSTANCE_SEMANTIC2NAME.keys())
)
INSTANCE_COLOR_PALLETE = {
    0: [0, 0, 0],  # unlabeled
    1: [31, 119, 180],  # cabinet
    2: [255, 187, 120],  # bed
    3: [188, 189, 34],  # chair
    4: [140, 86, 75],  # sofa
    5: [255, 152, 150],  # table
    6: [214, 39, 40],  # door
    7: [197, 176, 213],  # window
    8: [148, 103, 189],  # bookshelf
    9: [196, 156, 148],  # picture
    10: [23, 190, 207],  # counter
    11: [247, 182, 210],  # desk
    12: [219, 219, 141],  # curtain
    13: [255, 127, 14],  # refrigerator
    14: [158, 218, 229],  # shower curtain
    15: [44, 160, 44],  # toilet
    16: [112, 128, 144],  # sink
    17: [227, 119, 194],  # bathtub
    18: [82, 84, 163],  # otherfurn
}
INSTANCE_SEMANTIC2NAME.pop(0)
