import mmcv
import mmengine
import os.path as osp
from tqdm import tqdm

def convert_to_coco(ann_file, out_file, image_prefix, train_ids_file):
    with open(train_ids_file, 'r') as f:
        train_ids_str = f.read()
        train_ids = train_ids_str.split('\n')
    data_infos = mmengine.load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for k, v in tqdm(data_infos.items(), total = len(list(data_infos))):
        if k not in train_ids:
            continue
        filename = k + '.png'
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=int(k),
            file_name=filename,
            height=height,
            width=width))
        for i, obj in enumerate(v['polygons']):
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (
                min(px), min(py), max(px), max(py))
            data_anno = dict(
                image_id=int(k),
                id=obj_count,
                category_id=0 if v['labels'][i] == 'top' else 1,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id':0, 'name': 'top'}, {'id':1, 'name': 'overlap'}])
    mmengine.dump(coco_format_json, out_file)

convert_to_coco(
    './data/annotation.json',
    './data/annotation_coco.json',
    './data/images/',
    './data/train_ids.txt')