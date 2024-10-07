from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile
import json


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box1, box2: Lists or tuples with coordinates [x1, y1, width, height]

    Returns:
    iou: IoU value
    """
    x1_1, y1_1, w1, h1 = box1
    x2_1, y2_1, w2, h2 = box2

    x1_2 = x1_1 + w1
    y1_2 = y1_1 + h1
    x2_2 = x2_1 + w2
    y2_2 = y2_1 + h2

    xi1 = max(x1_1, x2_1)
    yi1 = max(y1_1, y2_1)
    xi2 = min(x1_2, x2_2)
    yi2 = min(y1_2, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    return iou


def is_significant_overlap(box1, box2, threshold=0.9):
    """
    Check if the IoU between two bounding boxes is greater than the given threshold.

    Parameters:
    box1, box2: Lists or tuples with coordinates [x1, y1, width, height]
    threshold: float, IoU threshold

    Returns:
    bool: True if IoU > threshold, else False
    """
    iou = calculate_iou(box1, box2)
    return iou > threshold


def update_image_id(images, dataset_coco):
    img2id = {img['file_name']: img['id'] for img in images}
    dataset_coco['images'] = [img for img in dataset_coco['images'] if img['file_name'] in img2id]
    # assert len(images) == len(dataset_coco['images']),print(len(images),len(dataset_coco['images']))
    old_images = [image['file_name'] for image in dataset_coco['images']]
    for image in images:
        img_file_name = image['file_name']
        # assert image['file_name'] in old_images, f'{img_file_name}'

    id2id = {img['id']: img2id[img['file_name']] for img in dataset_coco['images']}
    dataset_coco['annotations'] = [annotation for annotation in dataset_coco['annotations'] if
                                   annotation['image_id'] in id2id.keys()]

    for image in dataset_coco['images']:
        image['id'] = id2id[image['id']]

    for annotation in dataset_coco['annotations']:
        annotation['image_id'] = id2id[annotation['image_id']]

    return dataset_coco


def re_index_images(gt_data, pred_data):
    pred_data = update_image_id(gt_data['images'], pred_data)
    return gt_data, pred_data


def eval(gt_data, pred_data, confidence_threshold=0.5, iou_threshold=0.9, ignore_image_ids=[], re_index_image_ids=True):
    """
    calculate confusion_matrix/precision/recall[@iou_threshold] from gt_data, pred_data
    params:
    gt_data: groundtuth dict format coco
    pred_data: predictions dict format coco
    conf_threshold: confidence threshold
    iou_threshold: iou threshold
    ignore_image_ids: list image ids to be ignore
    re_index_image_ids: bool True: id of image in gt and pred is not match but file_name is the same
    return:
    conf_mat, precision, recall: format sklearn
    """
    # Confidence threshold
    # Map annotations to images
    if re_index_image_ids:
        gt_data, pred_data = re_index_images(gt_data, pred_data)
    annotations = defaultdict(list)
    for ann in gt_data['annotations']:
        if ann['image_id'] not in ignore_image_ids:
            annotations[ann['image_id']].append(ann)

    # Map predictions to images
    predictions = defaultdict(list)
    for pred in pred_data['annotations']:
        if pred['image_id'] not in ignore_image_ids and ('score' not in pred or pred['score'] >= confidence_threshold):
            predictions[pred['image_id']].append(pred)

    # Initialize confusion matrix components
    y_true = []
    y_pred = []

    for image_id in annotations.keys():
        gt_labels = [ann['category_id'] for ann in annotations[image_id]]
        gt_boxes = [ann['bbox'] for ann in annotations[image_id]]

        pred_labels = [pred['category_id'] for pred in predictions.get(image_id, [])]
        pred_boxes = [pred['bbox'] for pred in predictions.get(image_id, [])]

        gt_matched = [False] * len(gt_labels)
        pred_matched = [False] * len(pred_labels)

        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                if is_significant_overlap(gt_box, pred_box, iou_threshold):
                    gt_matched[i] = True
                    pred_matched[j] = True
                    y_true.append(gt_labels[i])
                    y_pred.append(pred_labels[j])

        for i, matched in enumerate(gt_matched):
            if not matched:
                y_true.append(gt_labels[i])
                y_pred.append('background')

        for j, matched in enumerate(pred_matched):
            if not matched:
                y_true.append('background')
                y_pred.append(pred_labels[j])

    # Convert category IDs to category names if necessary
    category_id_to_name = {cat['id']: cat['name'] for cat in gt_data['categories']}
    category_id_to_name['background'] = 'background'

    y_true = [category_id_to_name.get(label, 'background') for label in y_true]
    y_pred = [category_id_to_name.get(label, 'background') for label in y_pred]

    labels = list(category_id_to_name.values())

    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return conf_mat, precision, recall, labels


def get_mAP(gt_data, pred_data, confidence_threshold=0, ignore_image_ids=[], re_index_image_ids=True):
    if re_index_image_ids:
        gt_data, pred_data = re_index_images(gt_data, pred_data)

    pred_data['images'] = [img for img in pred_data['images'] if img['id'] not in ignore_image_ids]
    pred_data['annotations'] = [annot for annot in pred_data['annotations']
                                if annot['image_id'] not in ignore_image_ids and (
                                            'score' not in annot or annot['score'] >= confidence_threshold)]
    # scanned_pages_id = [image['id'] for image in val_data['images'] if image['file_name'] in scanned_pages or image['file_name'].split('_page_')[0] not in identify_files]
    gt_data['annotations'] = [annot for annot in gt_data['annotations'] if annot['image_id'] not in ignore_image_ids]
    gt_data['images'] = [img for img in gt_data['images'] if img['id'] not in ignore_image_ids]

    # Create a temporary JSON file from the JSON string
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as temp_json_file:
        temp_json_file.write(json.dumps(gt_data))
        temp_json_file.seek(0)
        coco_gt = COCO(temp_json_file.name)
    predictions = []
    print(len(pred_data['images']))
    for ann in pred_data['annotations']:
        prediction = {
            'image_id': ann['image_id'],
            'category_id': ann['category_id'],
            'bbox': ann['bbox'],
            'score': ann['score'] if 'score' in ann else 1.0  # Use a default score if not provided
        }
        predictions.append(prediction)

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as temp_json_file:
        temp_json_file.write(json.dumps(pred_data))
        temp_json_file.seek(0)
        coco_dt = coco_gt.loadRes(predictions)
    # Use the COCO object
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')  # 'bbox', 'segm', 'keypoints'

    # Run the evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    overall_metrics = {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "APs": coco_eval.stats[3],
        "APm": coco_eval.stats[4],
        "APl": coco_eval.stats[5],
    }

    # Class-wise evaluation
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    category_names = [cat['name'] for cat in categories]
    category_ids = [cat['id'] for cat in categories]

    class_wise_metrics = {}
    for cat_id, cat_name in zip(category_ids, category_names):
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        class_wise_metrics[f"AP-{cat_name}"] = coco_eval.stats[0]

    # Combine overall and class-wise metrics
    eval_results = {
        "bbox": {**overall_metrics, **class_wise_metrics}
    }

    # Save evaluation results to a JSON file
    # eval_results_path = 'eval_rule/evaluation_results.json'
    # with open(eval_results_path, 'w') as f:
    #     json.dump(eval_results, f, indent=4)

    return eval_results