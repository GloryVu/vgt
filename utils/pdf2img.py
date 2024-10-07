# import pdf2image
import argparse
import logging
import os
import fitz  # PyMuPDF

def convert_pdf_to_images(pdf, output, format):
    """
    Convert pdf files in folder to images with format 
    params:
    pdf: input folder contain pdf file
    output: output image saved
    format: image file extention 'png', 'jpg'
    """
    # Open the PDF
    if not os.path.exists(output):
        os.makedirs(output)
    for file in os.listdir(pdf):
        # Convert the PDF to images
        pdf_document = fitz.open(os.path.join(pdf,file))
        file = file.replace('.pdf','')
        # Iterate through each page
        for page_number in range(len(pdf_document)):
            # Get the page
            page = pdf_document.load_page(page_number)

            # Convert the page to an image
            image = page.get_pixmap()

            # Save the image
            image.save(os.path.join(
                    output, f"{file}_page_{page_number}.{format}"))
    logging.info(f"PDF converted to images and saved at {output}")
    # Close the PDF
    pdf_document.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdf",
                        required=True,
                        help="Path to the PDF file")
    parser.add_argument("--output",
                        required=False,
                        default="images",
                        help="Path to the output folder")
    parser.add_argument("--format",
                        required=False,
                        default="jpg",
                        help="Output image format")

    args = parser.parse_args()
    convert_pdf_to_images(args.pdf, args.output, args.format)
    # # Create the output folder if it doesn't exist
    # if not os.path.exists(args.output):
    #     os.makedirs(args.output)
    # for file in os.listdir(args.pdf):
    #     # Convert the PDF to images
    #     pdf_document = fitz.open(os.path.join(args.pdf,file))
    #     file = file.replace('.pdf','')
    #     # Iterate through each page
    #     for page_number in range(len(pdf_document)):
    #         # Get the page
    #         page = pdf_document.load_page(page_number)

    #         # Convert the page to an image
    #         image = page.get_pixmap()

    #         # Save the image
    #         image.save(os.path.join(
    #                 args.output, f"{file}_page_{page_number}.{args.format}"))
    # logging.info(f"PDF converted to images and saved at {args.output}")