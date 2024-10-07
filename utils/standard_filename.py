# import pdf2image
import argparse
import os
import shutil
import re
from unidecode import unidecode


def remove_punctuation(text):
    # Sử dụng biểu thức chính quy để loại bỏ các ký tự dấu câu
    return re.sub(r'[^\w\s]', '', text)


def rename_file(filename):
    return remove_punctuation(unidecode(filename)).replace(' ', '_')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input",
                        required=True,
                        help="Path to the PDF file")
    parser.add_argument("--output",
                        required=False,
                        default="pdf",
                        help="Path to the output folder")
    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    for filename in os.listdir(args.input):
        # Convert the PDF to images
        standard_filename = rename_file(filename.replace('.pdf', ''))
        shutil.copy(os.path.join(args.input, filename), os.path.join(args.output, standard_filename + '.pdf'))