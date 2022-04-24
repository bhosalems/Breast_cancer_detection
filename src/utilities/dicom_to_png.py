import os
import pandas as pd
import pydicom as pyd
import png
import re
import numpy as np
import argparse
import pickle


def convert_decom2png(in_dir, out_dir):

    print("Converting the files from DCOM to PNG")
    files = os.listdir(in_dir)
    dicom = []
    for file in files:
        if file.endswith('dcm'):
            dicom.append(os.path.join(file))

    print("Total number of DCOM files {0}".format(len(files)))
    filevals=[]
    for filename in dicom:
        filevals.append(re.findall( r"^([^.]*).*" , filename)[0])

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for file in filevals:
        dcm=pyd.dcmread(in_dir+'/%s.dcm'%file)
        shape=dcm.pixel_array.shape
        image_2d=dcm.pixel_array.astype(float)
        image_2d_scaled=(np.maximum(image_2d,0)/image_2d.max())*256
        image_2d_scaled=np.uint8(image_2d_scaled)

        with open(out_dir+'/%s.png'%file,'wb') as png_file:
            w=png.Writer(shape[1],shape[0],greyscale=True)
            w.write(png_file, image_2d_scaled)


def create_data_pkl(output_dir):
    '''
    Create metadafile name called as all_png_metadata.pkl in output dir, it contains the metadata of each image such as
    view etc.
    :param output_dir:
    :return:
    '''
    print("creating the metadata file")
    files = os.listdir(output_dir)
    df = pd.DataFrame([file.split("_") for file in files])
    df.columns = ['file_no', 'patient_id', 'mg', 'side', 'view', 'end']
    grouped_df = df.groupby(by='patient_id')

    # iterate over each group and create the pickle file needed to crop the images
    l = []
    for group_name, group_data in grouped_df:
        if (group_data.shape[0]) % 4 != 0:
            continue
        cnt = 0
        for row_index, row in group_data.iterrows():
            # print(row['file_no']+","+row['side']+","+row['view'])
            if cnt % 4 == 0:
                element = {'horizontal_flip': 'NO'}
            if row['view'] == "CC":
                fname = row['file_no'] + "_" + group_name + "_" + "MG_" + row['side'] + "_" + row['view'] + "_" + "ANON"
                element[row['side'] + '-' + row['view']] = [fname]
            else:
                fname = row['file_no'] + "_" + group_name + "_" + "MG_" + row['side'] + "_" + row['view'] + "_ANON"
                element[row['side'] + '-' + "MLO"] = [fname]
            # print(cnt, len(element))
            if cnt % 4 == 3 and len(element) == 5:
                l.append(element)
            cnt += 1

    output_file = output_dir + r"\all_png_metadata.pkl"
    print("output_file "+output_file)
    with open(output_file, 'wb') as handle:
        pickle.dump(l, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_file

