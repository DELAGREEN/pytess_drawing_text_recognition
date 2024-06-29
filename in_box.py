from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import datetime

start = datetime.datetime.now()

def convert_pdf_to_tiff(path):
    pages = convert_from_path(path, dpi=300)
    img = np.array(pages[0])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    # Используем Tesseract для получения данных о тексте
    data = pytesseract.image_to_data(img_rgb, lang="rus", output_type=Output.DICT)
    
    # Собираем координаты всех боксов текста
    boxes = []
    for i in range(len(data['level'])):
        if data['text'][i].strip():
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            boxes.append((x, y, w, h))
    
    # Создаем пустое изображение для маски
    mask = np.zeros_like(gray)
    
    # Заполняем маску прямоугольниками из боксов текста
    for (x, y, w, h) in boxes:
        mask[y:y+h, x:x+w] = 255
    
    # Применяем морфологические операции для объединения близко расположенных текстовых блоков
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 30))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    
    # Находим контуры объединенных блоков
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Рисуем объединенные блоки на изображении
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Фильтруем слишком большие области, чтобы уменьшить захват лишнего пространства
        if h < img_rgb.shape[0] * 0.5:  # например, игнорировать области, высота которых больше половины изображения
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('Text Detection', img_rgb)
    cv2.imwrite('text_detection_combined.png', img_rgb)
    cv2.waitKey(0)

path = '/home/nzxt/rep/drawing_text_recognition/pdf/F127965467_0.prt.pdf'
convert_pdf_to_tiff(path)

end = datetime.datetime.now()
print(end-start)
