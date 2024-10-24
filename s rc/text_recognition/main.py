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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 60))
    dilated = cv2.dilate(mask, kernel, iterations=1)

    # Находим контуры объединенных блоков
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Находим самую большую область текста
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Вырезаем самую большую область текста
    x, y, w, h = cv2.boundingRect(max_contour)
    largest_text_area = img_rgb[y:y+h, x:x+w]

    # Бинаризуем область текста
    gray = cv2.cvtColor(largest_text_area, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Распознаем текст
    text = pytesseract.image_to_string(thresh, lang="rus")

    print("Распознанный текст:")
    print(text)

    # Рисуем прямоугольник вокруг области текста на исходном изображении
    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Text Detection', img_rgb)
    #cv2.imshow('Largest Text Area', largest_text_area)
    #cv2.imshow('Binarized Text Area', thresh)
    #cv2.imwrite('text_detection_combined.png', img_rgb)
    #cv2.imwrite('largest_text_area.png', largest_text_area)
    #cv2.imwrite('binarized_text_area.png', thresh)



    # Используем Tesseract для получения данных о тексте
    data = pytesseract.image_to_data(thresh, lang="rus", output_type=Output.DICT)

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



    # Выводим текстовую область полностью и показываем, как именно обведен текст боксами
    cv2.imshow("Текстовая область", largest_text_area)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(largest_text_area, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Текстовая область с боксами", largest_text_area)
        


    cv2.waitKey(0)

path = '/home/nzxt/rep/pytess_drawing_text_recognition/pdf/КЛАБ.301329.369СБ_197011108.dwg.pdf'
convert_pdf_to_tiff(path)

end = datetime.datetime.now()
print(end-start)
