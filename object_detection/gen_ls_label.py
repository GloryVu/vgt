import argparse

import cv2

from ditod import add_vit_config
import re
import json
import torch

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from ditod.VGTTrainer import DefaultPredictor
import os
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Detectron2 inference script")
    parser.add_argument(
        "--image_root",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--grid_root",
        help="Path to input image",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_root",
        help="Name of the output visualization file.",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config-file",
        default="/home/kienpm3/vinhvq11-workspace/fine_tuning_vgt/vgt/object_detection/Configs/cascade/doclaynet_VGT_cascade_PTM.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # else:
    #     for image in os.listdir(args.image_root):
    #         image = image.replace('.jpg','')
    #         img_paths.append(args.image_root + image + ".jpg")
    #         grid_paths.append(args.grid_root + image + ".pdf.pkl")
    
    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    
    # Step 2: add model weights URL to config
    cfg.merge_from_list(args.opts)
    
    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # Step 4: define model
    predictor = DefaultPredictor(cfg)
    def extract_number(filename):
        name = os.path.basename(filename)
        idx = name.split('_')[-1]
        idx = int(idx.split('.')[0])
        return idx
    images = sorted(os.listdir(args.image_root),key=extract_number)
    ls_annots = []
    for pdf in tqdm(os.listdir('../data/tpbank/pdf')):
        basename = pdf.replace('.pdf','')
        img_paths =[]
        grid_paths =[]
        # if args.dataset in ('D4LA', 'doclaynet'):
        
        for image in images:
            if image.startswith(basename):
                image = image.replace(".png",'')
                img_paths.append(args.image_root + image + ".png")
                grid_paths.append(args.grid_root + image + ".pdf.pkl")
        
        annots = {
            'annotations':[{
                "result": []
            }],
            'data': {
                'file_name': pdf,
                'document':[{'page':f'/data/local-files/?d=documents/'+os.path.basename(img)} for img in img_paths],
                'domain': 'qms'
            }
        }

        for i, (image_path, grid_path) in enumerate(zip(img_paths,grid_paths)):
            # Step 5: run inference
            print(i,image_path,grid_path)
            img = cv2.imread(image_path)
            md = MetadataCatalog.get(cfg.DATASETS.TEST[0])

            if args.dataset == 'publaynet':
                md.set(thing_classes=["text","title","list","table","figure"])
            elif args.dataset == 'docbank':
                md.set(thing_classes=["abstract","author","caption","date","equation", "figure", "footer", "list", "paragraph", "reference", "section", "table", "title"])
            elif args.dataset == 'D4LA':
                md.set(thing_classes=["DocTitle","ParaTitle","ParaText","ListText","RegionTitle", "Date", "LetterHead", "LetterDear", "LetterSign", "Question", "OtherText", "RegionKV", "Regionlist", "Abstract", "Author", "TableName", "Table", "Figure", "FigureName", "Equation", "Reference", "Footnote", "PageHeader", "PageFooter", "Number", "Catalog", "PageNumber"])
            elif args.dataset == 'doclaynet':
                md.set(thing_classes=["text","title","list-item","table","figure", "form", "footnote", "useless"])
            categories=['text','title','list-item','table','figure','form','footnote','useless']
            {"id": 0, "name": "text"}, {"id": 1, "name": "title"}, {"id": 2, "name": "list-item"}, {"id": 3, "name": "table"}, {"id": 4, "name": "figure"}, {"id": 5, "name": "form"}, {"id": 6, "name": "footnote"}, {"id": 7, "name": "useless"}
            output = predictor(img, grid_path)["instances"]
            # import ipdb;ipdb.set_trace()
            v = Visualizer(img[:, :, ::-1],
                            md,
                            scale=1.0,
                            instance_mode=ColorMode.SEGMENTATION)
            # print('output',output.to("cpu"))
            # result = v.draw_instance_predictions(output.to("cpu"))
            # result_image = result.get_image()[:, :, ::-1]
            predictions=output.to("cpu")
            original_height, original_width = img.shape[0], img.shape[1]
            pred_boxes = predictions.pred_boxes.tensor.numpy()
            scores = predictions.scores.numpy()
            pred_classes = predictions.pred_classes.numpy()
            
            # Loop through each instance and print the details
            for j in range(pred_boxes.shape[0]):
                box = pred_boxes[j]
                score = scores[j]
                pred_class = pred_classes[j]
                annots['annotations'][0]["result"].append(
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": box[0]/original_width*100,
                            "y": box[1]/original_height*100,
                            "width": (box[2]-box[0])/original_width*100,
                            "height": (box[3]-box[1])/original_height*100,
                            "rotation": 0,
                            "rectanglelabels": [
                                categories[pred_class]
                            ]
                        },
                        "score": score.item(),
                        "to_name": f"page_{i}",
                        "from_name": f"labels_{i}",
                        "image_rotation": 0,
                        "original_width": original_width,
                        "original_height": original_height
                    }
                )
        ls_annots.append(annots)
            # step 6: save
            # cv2.imwrite(args.output_root+os.path.basename(image_path), result_image)
    with open(args.output_root+'ls_annots.json', 'w') as f:
        json.dump(ls_annots,f)
if __name__ == '__main__':
    main()

