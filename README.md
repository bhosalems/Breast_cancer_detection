# Breast_cancer_detection

First, download the INBreast dataset from below link -
https://drive.google.com/file/d/19n-p9p9C0eCQA1ybm6wkMo-bbeccT_62/view

Preprocess the INBreast dataset images,
```
python preprocess.py --input-dir "path_to_AllDICOMs" --output-dir "path_to_output_dir"
```

After preprocessing, to train the multi-stream view wise model
```
python training.py --data-path "path_to_output_dir/data.pkl" --image-path "path_to_output_dir/cropped_images" --annotation "path_to_INbreast.xls" --num-rows 410 --device-type "cuda/cpu"
```

To train the traditional classifiers on DDSM/INBreast dataset head over to [pretrained_classifier.ipynb](https://github.com/bhosalems/Breast_cancer_detection/blob/main/pretrained_classifier.ipynb)

Checkout the [report](src/ybadugu_mbhosale_yulinchi_cchen248.pdf) -
