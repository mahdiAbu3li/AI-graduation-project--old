# Analyzes text in a document stored in an S3 bucket. Display polygon box around text and angled text
import boto3
import io
from io import BytesIO
import sys
import pandas as pd
import os
import cv2
import math
from PIL import Image, ImageDraw, ImageFont


def ShowBoundingBox(draw, box, width, height, boxColor):
    left = width * box['Left']
    top = height * box['Top']
    draw.rectangle([left, top, left + (width * box['Width']), top + (height * box['Height'])], outline=boxColor)


def ShowSelectedElement(draw, box, width, height, boxColor):
    left = width * box['Left']
    top = height * box['Top']
    draw.rectangle([left, top, left + (width * box['Width']), top + (height * box['Height'])], fill=boxColor)





# Displays information about a block returned by text detection and text analysis
def DisplayBlockInformation(block):
    print('Id: {}'.format(block['Id']))
    if 'Text' in block:
        print('    Detected: ' + block['Text'])
    print('    Type: ' + block['BlockType'])

    if 'Confidence' in block:
        print('    Confidence: ' + "{:.2f}".format(block['Confidence']) + "%")

    if block['BlockType'] == 'CELL':
        print("    Cell information")
        print("        Column:" + str(block['ColumnIndex']))
        print("        Row:" + str(block['RowIndex']))
        print("        Column Span:" + str(block['ColumnSpan']))
        print("        RowSpan:" + str(block['ColumnSpan']))

    if 'Relationships' in block:
        print('    Relationships: {}'.format(block['Relationships']))
    print('    Geometry: ')
    print('        Bounding Box: {}'.format(block['Geometry']['BoundingBox']))

    print('        Polygon: {}'.format(block['Geometry']['Polygon']))

    if block['BlockType'] == "KEY_VALUE_SET":
        print('    Entity Type: ' + block['EntityTypes'][0])

    if block['BlockType'] == 'SELECTION_ELEMENT':
        print('    Selection element detected: ', end='')

        if block['SelectionStatus'] == 'SELECTED':
            print('Selected')
        else:
            print('Not selected')

    if 'Page' in block:
        print('Page: ' + block['Page'])
    print()


def process_text_analysis(path):
    # Get the document from S3
    # s3_connection = boto3.resource('s3')
    #
    # s3_object = s3_connection.Object(bucket, document)
    # s3_response = s3_object.get()
    #
    # stream = io.BytesIO(s3_response['Body'].read())
    # image = Image.open(stream)

    # Analyze the document
    client = boto3.client('textract', region_name='us-west-2')

    im = cv2.imread(path)
    # im_resize = cv2.resize(im, (500, 500))

    is_success, im_buf_arr = cv2.imencode(".jpg", im)
    byte_im = im_buf_arr.tobytes()

    image = Image.open(path)

    # image_binary = stream.getvalue()
    response = client.analyze_document(Document={'Bytes': byte_im},
                                       FeatureTypes=["TABLES", "FORMS"])

    # Alternatively, process using S3 object
    # response = client.analyze_document(
    #    Document={'S3Object': {'Bucket': bucket, 'Name': document}},
    #    FeatureTypes=["TABLES", "FORMS"])

    # Get the text blocks
    blocks = response['Blocks']
    width, height = image.size
    draw = ImageDraw.Draw(image)
    print('Detected Document Text')
    r = ["", 0, 0, 0, 0, ""]
    df = pd.DataFrame([r], columns=["ID", "xmin", "ymin", "xmax", "ymax", "Text"])
    # Create image showing bounding box/polygon the detected lines/text
    for block in blocks:

        data = []

        if block['BlockType'] == 'WORD':
            data.append(block['Id'])
            data.append(block['Geometry']['Polygon'][0]['X'])
            data.append(block['Geometry']['Polygon'][0]['Y'])
            data.append(block['Geometry']['Polygon'][1]['X'])
            data.append(block['Geometry']['Polygon'][2]['Y'])
            if 'Text' in block:
                data.append(block['Text'])
            # print(data, "mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
            df.loc[len(df.index)] = data

        # DisplayBlockInformation(block)

        # draw = ImageDraw.Draw(image)
    #     if block['BlockType'] == "KEY_VALUE_SET":
    #         if block['EntityTypes'][0] == "KEY":
    #             ShowBoundingBox(draw, block['Geometry']['BoundingBox'], width, height, 'red')
    #         else:
    #             ShowBoundingBox(draw, block['Geometry']['BoundingBox'], width, height, 'green')
    #
    #     if block['BlockType'] == 'TABLE':
    #         ShowBoundingBox(draw, block['Geometry']['BoundingBox'], width, height, 'blue')
    #
    #     if block['BlockType'] == 'CELL':
    #         ShowBoundingBox(draw, block['Geometry']['BoundingBox'], width, height, 'yellow')
    #     if block['BlockType'] == 'SELECTION_ELEMENT':
    #         if block['SelectionStatus'] == 'SELECTED':
    #             ShowSelectedElement(draw, block['Geometry']['BoundingBox'], width, height, 'blue')
    #
    #             # uncomment to draw polygon for all Blocks
    #         points = []
    #         for polygon in block['Geometry']['Polygon']:
    #             points.append((width * polygon['X'], height * polygon['Y']))
    #         draw.polygon((points), outline='blue')
    #
    # # Display the image
    # image.show()
    return df
    # df.to_csv("data/n1.csv")
    # return len(blocks)


# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         print(filename)
#         img = cv2.imread(os.path.join(folder, filename))
#         if img is not None:
#             images.append(img)
#     return images


def main():
    bucket = ''
    document = ''
    # block_count = process_text_analysis()
    # print("Blocks detected: " + str(block_count))

    # images = load_images_from_folder("data/train_images")
    #
    # for i in range(len(images)):
    #     path = "data/train_images/c" + str(i) + ".jpg"
    #     data_frame = process_text_analysis(path)
    #     # data_frame.to_csv("data/train_csv" + str(i) + ".csv")



# if __name__ == '__main_':
#     main()
