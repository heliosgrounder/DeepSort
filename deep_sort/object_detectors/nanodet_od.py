import os
import torch
import cv2

from packages.nanodet.nanodet.data.batch_process import stack_batch_img
from packages.nanodet.nanodet.data.collate import naive_collate
from packages.nanodet.nanodet.data.transform import Pipeline
from packages.nanodet.nanodet.model.arch import build_model
from packages.nanodet.nanodet.util import Logger, cfg, load_config, load_model_weight
from packages.nanodet.nanodet.util.path import mkdir

from deep_sort.bbox import BBox
from deep_sort.types.nanodet_types import NanodetModelTypes

class NanodetOD:
    def __init__(self, model_type):
        match model_type:
            case NanodetModelTypes.PLUSM416:
                cfg_ = "models/detection/nanodet/nanodet-plus-m_416.yml"
                model_path = "models/detection/nanodet/nanodet-plus-m_416_checkpoint.ckpt"
            case _:
                cfg_ = None
                model_path = None
        
        self.__device = "cpu:0"
        load_config(cfg, cfg_)
        self.__cfg = cfg
        model = build_model(self.__cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, Logger(0, use_tensorboard=False))
        if self.__cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = self.__cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)

        self.__model = model.to(self.__device).eval()
        self.__pipeline = Pipeline(self.__cfg.data.val.pipeline, self.__cfg.data.val.keep_ratio)


    def get_detections(self, image_file, min_detection_height=0):
        img_info = {"id": 0}
        img = cv2.imread(image_file)
        img_info["file_name"] = os.path.basename(image_file)
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width

        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.__pipeline(None, meta, self.__cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.__device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.__model.inference(meta)

        detections_list = []
        results = results[0][self.__cfg.class_names.index("person")]
        # test = []
        for bboxes in results:
            x1, y1, x2, y2, confidence = bboxes
            bbox = [x1, y1, x2 - x1, y2 - y1]
            if bbox[3] < min_detection_height:
                continue
            detections_list.append(BBox(bbox, confidence))
            # test.append(confidence)
        # print(max(test))
        return detections_list