import sys

sys.path.append('../')
from src.utilities.dicom_to_png import convert_decom2png, create_data_pkl
import argparse
from src.cropping.crop_mammogram import crop_mammogram


def preprocess(input_dir, output_dir):
    # convert the DCOM files to pngs.
    convert_decom2png(input_dir, output_dir)

    # need to create the pkl file before cropping the data
    metadata_file = create_data_pkl(output_dir)

    # crop the mammograms
    cropped_img_path = output_dir + r"\cropped_images"
    cropped_img_list = cropped_img_path + r'\cropped_img_list.pkl'
    crop_mammogram(input_data_folder=output_dir, output_data_folder=cropped_img_path, num_processes=10,
                   cropped_exam_list_path=cropped_img_list, exam_list_path=metadata_file, buffer_size=50,
                   num_iterations=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert DICOM images to png')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    preprocess(input_dir=args.input_dir, output_dir=args.output_dir)
