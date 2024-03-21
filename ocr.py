from PIL import Image
import easyocr

def ocr_from_image(image_path):
    # Create a reader to do OCR.
    reader = easyocr.Reader(["it"])  # need to run only once to load model into memory
    # Use EasyOCR to do OCR on the image
    result = reader.readtext(image_path)
    # Extract text from the OCR result
    text = ' '.join([item[1] for item in result])
    return text
