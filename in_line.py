from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import datetime

start = datetime.datetime.now()

def merge_boxes(boxes, margin=10):
    merged_boxes = []
    for box in boxes:
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            x, y, w, h = box
            lx, ly, lw, lh = merged_boxes[-1]
            if y <= ly + lh + margin:
                merged_boxes[-1] = (min(x, lx), min(y, ly), max(x + w, lx + lw) - min(x, lx), max(y + h, ly + lh) - min(y, ly))
            else:
                merged_boxes.append(box)
    return merged_boxes

def convert_pdf_to_tiff(path):
    pages = convert_from_path(path, dpi=300, fmt="png")
    img = np.array(pages[0])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    data = pytesseract.image_to_data(img_rgb, lang="rus", output_type=Output.DICT)
    
    boxes = []
    for i in range(len(data['level'])):
        if data['text'][i].strip():
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            boxes.append((x, y, w, h))
    
    merged_boxes = merge_boxes(boxes)
    
    for (x, y, w, h) in merged_boxes:
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('Text Detection', img_rgb)
    cv2.imwrite('text_detection.png', img_rgb)
    cv2.waitKey(0)

path = '/home/nzxt/rep/drawing_text_recognition/pdf/F127965467_0.prt.pdf'
convert_pdf_to_tiff(path)

end = datetime.datetime.now()
print(end-start)
