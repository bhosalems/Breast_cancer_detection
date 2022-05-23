# Breast_cancer_detection

Preprocess the images
```
python preprocess.py --input-dir "path_to_AllDICOMs" --output-dir "path_to_output_dir"
```

To train the traditional classifiers head over to pretrained_classifier.ipynb

INBreast dataset Can be downloaded from below link -
https://drive.google.com/file/d/19n-p9p9C0eCQA1ybm6wkMo-bbeccT_62/view

To train the multi-stream view wise model
```
python training.py --data-path "path_to_output_dir/data.pkl" --image-path "path_to_output_dir/cropped_images" --annotation "path_to_INbreast.xls" --num-rows 410 --device-type "cuda/cpu"
```
