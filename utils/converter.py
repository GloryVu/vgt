from standard_filename import rename_file


def reorder_element(elements):
    """
    reoder elements in a document to read order
    method: sort by page, by y then by x
    params:
    elements: List[{
                        "original_width": 612,
                        "original_height": 792,
                        "image_rotation": 0,
                        "value": {
                            "x": 11.781045502307368,
                            "y": 27.614477908972535,
                            "width": 20.7784253787371,
                            "height": 1.6779119318181819,
                            "rotation": 0,
                            "rectanglelabels": [
                                "title"
                            ]
                        },
                        "id": "f4647ae4b4",
                        "from_name": "labels_2",
                        "to_name": "page_2",
                        "type": "rectanglelabels",
                        "origin": "manual"
                    },]
    return ordered elements
    """

    def extract_page(text):
        return int(text.split('_')[-1])

    return sorted(elements, key=lambda x: (extract_page(x['to_name']), x['value']['y'], x['value']['x']))



def flatten(x):
    flattened = []
    stack = [x]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(current)
        else:
            flattened.append(current)
    flattened.reverse()
    return flattened


import fitz


def get_page_size(pdf_file, page):
    doc = fitz.open(pdf_file)
    rect = doc[page].rect
    return (rect.width, rect.height)


def convert_chunk2ls(chunks):
    """
    input:
    [
        {
            'file_name': '14_San_pham_bao_lanh_danh_cho_khach_hang_doanh_nghiep_va_khach_hang_ca_nhan.pdf',
            "topdown_chunk": {
                "content": {
                    "title": "QUYẾT ĐỊNH Về việc ủy quyền Cán bộ",
                    "context": "TỔNG GIÁM ĐỐC CÔNG TY TNHH FPT SMART CLOUD\n- Căn cứ Bộ Luật Dân sự số 91/2015/QH13;\n- Căn cứ Điều lệ Công ty TNHH FPT Smart Cloud;\n- Căn cứ Quy chế quản lý tài chính của Công ty TNHH FPT Smart Cloud;\nCăn cứ Quyết định số 52.1/2023/QĐ-FCI ký ngày 01/09/2023 về việc Bổ nhiệm Bà Đỗ Kim Hoa giữ vị trí Kế toán trưởng Công ty TNHH FPT Smart Cloud kể từ ngày 01/09/2023;\n- Căn cứ nhu cầu công việc,\nQUYẾT ĐỊNH:\nĐiều 1. Ủy quyền cho Bà Đỗ Kim Hoa hiện là Kế toán trưởng Công ty TNHH FPT Smart Cloud được thực hiện các quyền sau đây tại Công ty TNHH FPT Smart Cloud:\n- Ký các Biên bản điều chỉnh, thu hồi và hủy hóa đơn bán ra và mua vào của Công ty.\nĐiều 2. Quyết định có hiệu lực kể từ ngày 08/01/2024 đến hết ngày 31/08/2026.\nĐiều 3. Ban Tổng Giám đốc, Trưởng phòng Nhân sự, Trưởng phòng Hành chính, Bà Đỗ Kim Hoa và các cá nhân, đơn vị có liên quan chịu trách nhiệm thi hành Quyết định này./.\nTỔNG GIÁM ĐỐC (đã ký số)\nNơi nhận: - Như Điều 3; - Lưu VT, NS."
                },
                "location": [
                    {
                        "page": 1,
                        "bbox": [
                            67.37755584716797,
                            173.55895996093753,
                            542.1920166015625,
                            580.9492797851562
                        ]
                    }
                ]
            }
            "bottomup_chunks": [
                {
                    "content": {
                        "title": "QUYẾT ĐỊNH Về việc ủy quyền Cán bộ - TỔNG GIÁM ĐỐC CÔNG TY TNHH FPT SMART CLOUD",
                        "context": "- Căn cứ Bộ Luật Dân sự số 91/2015/QH13;\n- Căn cứ Điều lệ Công ty TNHH FPT Smart Cloud;\n- Căn cứ Quy chế quản lý tài chính của Công ty TNHH FPT Smart Cloud;\nCăn cứ Quyết định số 52.1/2023/QĐ-FCI ký ngày 01/09/2023 về việc Bổ nhiệm Bà Đỗ Kim Hoa giữ vị trí Kế toán trưởng Công ty TNHH FPT Smart Cloud kể từ ngày 01/09/2023;\n- Căn cứ nhu cầu công việc,"
                    },
                    "location": [
                        {
                            "page": 1,
                            "bbox": [
                                67.37755584716797,
                                231.4530792236328,
                                541.899658203125,
                                326.1887817382813
                            ]
                        }
                    ]
                }
            ]

        }
    ]
    output:
    [{
            'annotations':[{
                "result": [
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
                        "original_height": original_height,
                        ** input element
                    }
                ]
            }],
            'data': {
                'file_name': pdf,
                'document':[{'page':f'/data/local-files/?d=documents/'+os.path.basename(img)} for img in img_paths],
                'domain': 'qms'
            }
    }]
    """
    categories = set()
    ls_annots = []
    for chunk in chunks:
        # if len(doc_struct['element']) == 0:
        #     continue
        file_name = chunk['file_name']
        base_name = rename_file(file_name.replace('.pdf', ''))
        pages = [location['page'] for location in chunk['topdown_chunk']['location']]
        annots = {
            'annotations': [  # {
                {
                    "completed_by": 6,
                    'result': []
                }],

            'data': {
                'file_name': file_name,
                'document': [{'page': f'/data/local-files/?d=documents/{base_name}_page_{i}.png'} for i in pages]
            }
        }
        # idx2element = {i:e['idx'] for i,e in enumerate(doc_struct['elements'])}
        # print(idx2element)

        for j, location in enumerate(chunk['topdown_chunk']['location']):
            # try:
            original_width, original_height = get_page_size(
                f'/home/vinhvq11/Desktop/PDF_extraction/data_process/extract_elements_text/fci/pdf/{file_name}',
                location['page'] - 1)

            annots['annotations'][0]['result'].append(
                {
                    # "id": f'{element_idx}_{i}',
                    "type": "rectanglelabels",
                    "value": {
                        "x": location['bbox'][0] * 100 / original_width,
                        "y": location['bbox'][1] * 100 / original_height,
                        "width": (location['bbox'][2] - location['bbox'][0]) * 100 / original_width,
                        "height": (location['bbox'][3] - location['bbox'][1]) * 100 / original_height,
                        "rotation": 0,
                        "rectanglelabels": [
                            f'new'
                        ],

                    },
                    'meta': {
                        'text': [chunk['topdown_chunk']['content']['title'] + '\n' + chunk['topdown_chunk']['content'][
                            'context']] if j == 0 else ['similar previos chunk'],
                    },
                    "to_name": f"page_{pages.index(location['page'])}",
                    "from_name": f"labels_{pages.index(location['page'])}",
                    "image_rotation": 0,
                    "original_width": original_width,
                    "original_height": original_height,

                }
            )

        for i, c in enumerate(chunk['bottomup_chunks']):
            # if len(flatten(element['location'])) > 2:
            #     print(element)

            for j, location in enumerate(c['location']):
                # try:
                original_width, original_height = get_page_size(
                    f'/home/vinhvq11/Desktop/PDF_extraction/data_process/extract_elements_text/fci/pdf/{file_name}',
                    location['page'] - 1)

                annots['annotations'][0]['result'].append(
                    {
                        # "id": f'{element_idx}_{i}',
                        "type": "rectanglelabels",
                        "value": {
                            "x": location['bbox'][0] * 100 / original_width,
                            "y": location['bbox'][1] * 100 / original_height,
                            "width": (location['bbox'][2] - location['bbox'][0]) * 100 / original_width,
                            "height": (location['bbox'][3] - location['bbox'][1]) * 100 / original_height,
                            "rotation": 0,
                            "rectanglelabels": [
                                f'old'
                            ]
                        },
                        'meta': {
                            'text': [c['content']['title'] + '\n' + c['content']['context']] if j == 0 else [
                                'similar previos chunk']
                        },
                        "to_name": f"page_{pages.index(location['page'])}",
                        "from_name": f"labels_{pages.index(location['page'])}",
                        "image_rotation": 0,
                        "original_width": original_width,
                        "original_height": original_height,

                    }
                )

        ls_annots.append(annots)

    return ls_annots, categories


def convert_dla2ls(doc_structs):
    """
    input:
    [
        {
            'file_id': 474,
            'file_name': '14_San_pham_bao_lanh_danh_cho_khach_hang_doanh_nghiep_va_khach_hang_ca_nhan.pdf',
            'source': 'tp-bank',
            'elements': [
                {
                    "page": page_number+1,
                    'strategy': strategy,
                    "label": categories[int(box.cls.item())],
                    "set": None,
                    "confidence_score": box.conf.item(),
                    "original_width": original_width,
                    "original_height": original_height,
                    "bbox": {
                        "x": bbox[0].item(),
                        "y": bbox[1].item(),
                        "w": bbox[2].item() - bbox[0].item(),
                        "h": bbox[3].item() - bbox[1].item(),
                    }
            ]
        }
    ]
    output:
    [{
            'annotations':[{
                "result": [
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
                        "original_height": original_height,
                        ** input element
                    }
                ]
            }],
            'data': {
                'file_name': pdf,
                'document':[{'page':f'/data/local-files/?d=documents/'+os.path.basename(img)} for img in img_paths],
                'domain': 'qms'
            }
    }]
    """
    categories = set()
    ls_annots = []
    for doc_struct in doc_structs:
        # if len(doc_struct['element']) == 0:
        #     continue
        file_name = doc_struct['file_name']
        base_name = file_name.replace('.pdf', '')
        source = doc_struct['source']
        file_id = doc_struct['file_id']
        total_page = doc_struct['elements'][-1]['location'][-1]['page']
        annots = {
            'annotations': [  # {
                #     "completed_by": 6,
                #     'result':[]
                # },
                # {
                #     "completed_by": 2,
                #     'result':[]
                # },
                {
                    "completed_by": 6,
                    'result': []
                }],

            'data': {
                'file_name': file_name,
                'file_id': file_id,
                'source': source,
                'document': [{'page': f'/data/local-files/?d=documents/{base_name}_page_{i + 1}.png'} for i in
                             range(total_page)]
            }
        }
        # idx2element = {i:e['idx'] for i,e in enumerate(doc_struct['elements'])}
        # print(idx2element)
        for element in doc_struct['elements']:
            # if len(flatten(element['location'])) > 2:
            #     print(element)

            for i, location in enumerate(flatten(element['location'])):
                # try:
                category = element['label']

                annots['annotations'][0]['result'].append(
                    {
                        # "id": f'{element_idx}_{i}',
                        "type": "rectanglelabels",
                        "value": {
                            "x": element['bbox']['x'] * 100 / element['original_width'],
                            "y": element['bbox']['y'] * 100 / element['original_height'],
                            "width": element['bbox']['w'] * 100 / element['original_width'],
                            "height": element['bbox']['h'] * 100 / element['original_height'],
                            "rotation": 0,
                            "rectanglelabels": [
                                category
                            ]
                        },
                        "to_name": f"page_{element['page'] - 1}",
                        "from_name": f"labels_{element['page'] - 1}",
                        "image_rotation": 0,
                        "original_width": element['original_width'],
                        "original_height": element['original_height'],

                    }
                )

        ls_annots.append(annots)

    return ls_annots, categories


def convert_docstruct2ls(doc_structs):
    """
    input:
    [
        {
            'file_id': 474,
            'file_name': '14_San_pham_bao_lanh_danh_cho_khach_hang_doanh_nghiep_va_khach_hang_ca_nhan.pdf',
            'source': 'tp-bank',
            'elements': [{'idx': 0,
                'page': 0,
                'original_width': 612,
                'original_height': 792,
                'label': 'main-title',
                'set': 'benchmark',
                'bbox': [{'x': 17.349751790364586,
                'y': 10.208446810943911,
                'w': 7.447518242730035,
                'h': 1.393890380859375}],
                'content': 'MỤC LỤC   ',
                'metadata': {'start_token': 'MỤC',
                'start_with_lowercase': False,
                'style': 'unk',
                'heading-level': 0},
                'continued': False,
                'xpath': []}
            ]
        }
    ]
    output:
    [{
            'annotations':[{
                "result": [
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
                        "original_height": original_height,
                        ** input element
                    }
                ]
            }],
            'data': {
                'file_name': pdf,
                'document':[{'page':f'/data/local-files/?d=documents/'+os.path.basename(img)} for img in img_paths],
                'domain': 'qms'
            }
    }]
    """
    categories = set()
    ls_annots = []
    for doc_struct in doc_structs:
        # if len(doc_struct['element']) == 0:
        #     continue
        file_name = doc_struct['file_name']
        base_name = file_name.replace('.pdf', '')
        source = doc_struct['source']
        file_id = doc_struct['file_id']
        total_page = doc_struct['element'][-1]['location'][-1]['page']
        annots = {
            'annotations': [  # {
                #     "completed_by": 6,
                #     'result':[]
                # },
                # {
                #     "completed_by": 2,
                #     'result':[]
                # },
                {
                    "completed_by": 14,
                    'result': []
                }],

            'data': {
                'file_name': file_name,
                'file_id': file_id,
                'source': source,
                'document': [{'page': f'/data/local-files/?d=documents/{base_name}_page_{i}.png'} for i in
                             range(total_page)]
            }
        }
        # idx2element = {i:e['idx'] for i,e in enumerate(doc_struct['elements'])}
        # print(idx2element)
        for element in doc_struct['element']:
            # if len(flatten(element['location'])) > 2:
            #     print(element)

            for i, location in enumerate(flatten(element['location'])):
                # try:
                category = element['label']
                if 'metadata' in element:
                    if 'heading-level' in element['metadata']:
                        category += '-lv-' + str(element['metadata']['heading-level'])

                    elif 'nested-level' in element['metadata']:
                        category += '-lv-' + str(element['metadata']['nested-level'])
                    elif category == 'list-item':
                        category += '-lv-0'
                    if category not in categories:
                        categories.add(category)
                parentID = None

                if 'xpath' in element and element['xpath'] and len(element['xpath']) > 0:
                    parent_idx = element['xpath'][-1]
                    parentID = f'{parent_idx}_0'
                element_idx = element['idx']
                annots['annotations'][0]['result'].append(
                    {
                        "parentID": parentID,
                        "id": f'{element_idx}_{i}',
                        "type": "rectanglelabels",
                        "value": {
                            "x": location['bbox']['x'] * 100 / location['original_width'],
                            "y": location['bbox']['y'] * 100 / location['original_height'],
                            "width": location['bbox']['w'] * 100 / location['original_width'],
                            "height": location['bbox']['h'] * 100 / location['original_height'],
                            "rotation": 0,
                            "rectanglelabels": [
                                category
                            ]
                        },
                        "to_name": f"page_{location['page'] - 1}",
                        "from_name": f"labels_{location['page'] - 1}",
                        "image_rotation": 0,
                        "original_width": location['original_width'],
                        "original_height": location['original_height'],
                        'meta': {
                            'text': [element['content']],
                            "metadata": {
                                **element
                            }
                        }

                    }
                )

        ls_annots.append(annots)

    return ls_annots, categories


def convert_ls(ls_annot, training_set=None):
    """
    reoder elements in a document to read order

    params:
    ls_annot:
        [{
            "id": 81064,
            "annotations": [
                {
                "id": 29939,
                "completed_by": 6,
                "result": [
                    {
                        "original_width": 612,
                        "original_height": 792,
                        "image_rotation": 0,
                        "value": {
                            "x": 11.781045502307368,
                            "y": 27.614477908972535,
                            "width": 20.7784253787371,
                            "height": 1.6779119318181819,
                            "rotation": 0,
                            "rectanglelabels": [
                                "title"
                            ]
                        },
                        "id": "f4647ae4b4",
                        "from_name": "labels_2",
                        "to_name": "page_2",
                        "type": "rectanglelabels",
                        "origin": "manual"
                    },]
                    ,
            "data": {
                "file_name": "QMFE12__So_tay_chat_luong_FE_final.pdf",
                "document": [
                    {
                        "page": "\/data\/local-files\/?d=documents\/QMFE12__So_tay_chat_luong_FE_final_page_0.jpg"
                    },
                ],
                "domain": "qms"
            }
        }]
    return [{
        "file_id": null,
        "file_name": "07_2020_TTBTTTT_439926.pdf",
        "domain": "9k",
        "is_valid": null,
        "elements": [
            {
                "page": 0,
                "training_set":"train",
                "annotated_label": null,
                "predicted_label": "useless",
                "predicted_score": {
                    "useless": 0.9914798736572266
                },
                "original_width": 612,
                "original_height": 792,
                "bbox": {
                    "x": 14.5804748036503,
                    "y": 9.545940823025173,
                    "w": 68.31596224915748,
                    "h": 8.363588891848169
                }
            }]
    """

    def extract_page(text):
        return int(text.split('_')[-1])

    result = []
    for sample in ls_annot:
        file_name = sample['data']['file_name']
        domain = sample['data']['domain'] if 'domain' in sample['data'] else None
        source = sample['data']['source'] if 'source' in sample['data'] else None
        elements = []
        for annot in reorder_element(sample['annotations'][0]['result']):
            base_name = file_name.replace('.pdf', '')
            page = extract_page(annot['to_name'])
            img_name = f'{base_name}_page_{page}.png'
            imgset = None
            if 'rectanglelabels' in annot['value']:
                elements.append({
                    "page": page + 1,
                    "set": imgset if imgset else training_set,
                    "label": annot['value']['rectanglelabels'][0],
                    "confidence_score": None,
                    "original_width": annot['original_width'],
                    "original_height": annot['original_height'],
                    "bbox": {
                        "x": annot['value']['x'] / 100 * annot['original_width'],
                        "y": annot['value']['y'] / 100 * annot['original_height'],
                        "w": annot['value']['width'] / 100 * annot['original_width'],
                        "h": annot['value']['height'] / 100 * annot['original_height']
                    }

                })
        result.append({
            "file_id": None,
            "file_name": file_name,
            "domain": domain,
            'source': source,
            "elements": elements
        })

    return result