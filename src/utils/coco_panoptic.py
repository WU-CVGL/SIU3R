import itertools
import os
from collections import defaultdict
from pycocotools.coco import COCO

try:
    import panopticapi
    from panopticapi.evaluation import VOID
    from panopticapi.utils import id2rgb
except ImportError:
    panopticapi = None
    id2rgb = None
    VOID = None


class COCOPanoptic(COCO):
    def __init__(self, annotation_file=None):
        if panopticapi is None:
            raise RuntimeError(
                "panopticapi is not installed, please install it by: "
                "pip install git+https://github.com/cocodataset/"
                "panopticapi.git."
            )

        super(COCOPanoptic, self).__init__(annotation_file)

    def createIndex(self):
        # create index
        print("creating index...")
        # anns stores 'segment_id -> annotation'
        anns, cats, imgs = {}, {}, {}
        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if "annotations" in self.dataset:
            for ann, img_info in zip(
                self.dataset["annotations"], self.dataset["images"]
            ):
                img_info["segm_file"] = ann["file_name"]
                for seg_ann in ann["segments_info"]:
                    # to match with instance.json
                    seg_ann["image_id"] = ann["image_id"]
                    seg_ann["height"] = img_info["height"]
                    seg_ann["width"] = img_info["width"]
                    img_to_anns[ann["image_id"]].append(seg_ann)
                    # segment_id is not unique in coco dataset orz...
                    if seg_ann["id"] in anns.keys():
                        anns[seg_ann["id"]].append(seg_ann)
                    else:
                        anns[seg_ann["id"]] = [seg_ann]

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                imgs[img["id"]] = img

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                for seg_ann in ann["segments_info"]:
                    cat_to_imgs[seg_ann["category_id"]].append(ann["image_id"])

        print("index created!")

        self.anns = anns
        self.imgToAnns = img_to_anns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats

    def load_anns(self, ids=[]):
        """Load anns with the specified ids.

        self.anns is a list of annotation lists instead of a
        list of annotations.

        Args:
            ids (int array): integer ids specifying anns

        Returns:
            anns (object array): loaded ann objects
        """
        anns = []

        if hasattr(ids, "__iter__") and hasattr(ids, "__len__"):
            # self.anns is a list of annotation lists instead of
            # a list of annotations
            for id in ids:
                anns += self.anns[id]
            return anns
        elif type(ids) is int:
            return self.anns[ids]
