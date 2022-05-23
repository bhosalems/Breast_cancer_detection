# Breast_cancer_detection

Preprocess the images
python preprocess.py --input-dir "path_to_AllDICOMs" --output-dir "path_to_output_dir"

To train the traditional classifiers for DDSM and INbreast, 
head over to pretrained_classifier.ipynb

To train the multi-stream view wise model
python training.py --data-path "path_to_output_dir/data.pkl" --image-path "path_to_output_dir/cropped_images" --annotation "path_to_INbreast.xls" --num-rows 410 --device-type "cuda/cpu"
