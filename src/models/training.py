import argparse
import collections as col
import numpy as np
import os
import pandas as pd
import torch
import tqdm

import src.utilities.pickling as pickling
from src.constant import VIEWS, VIEWANGLES, LABELS, MODELMODES
import src.utilities.preprocess as preprocess
from src.models.models import SplitBreastModel
import torch.nn as nn


def compute_batch_predictions(y_hat, mode):
    """
    Format predictions from different heads
    """

    if mode == MODELMODES.VIEW_SPLIT:
        assert y_hat[VIEWANGLES.CC].shape[1:] == (4, 2)
        assert y_hat[VIEWANGLES.MLO].shape[1:] == (4, 2)
        batch_prediction_tensor_dict = col.OrderedDict()
        batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 0]
        batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 0]
        batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 1]
        batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 1]
        batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 2]
        batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 2]
        batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 3]
        batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 3]
        batch_prediction_dict = col.OrderedDict([
            (k, torch.exp(v))
            for k, v in batch_prediction_tensor_dict.items()
        ])
    elif mode == MODELMODES.IMAGE:
        assert y_hat[VIEWS.L_CC].shape[1:] == (2, 2)
        assert y_hat[VIEWS.R_CC].shape[1:] == (2, 2)
        assert y_hat[VIEWS.L_MLO].shape[1:] == (2, 2)
        assert y_hat[VIEWS.R_MLO].shape[1:] == (2, 2)
        batch_prediction_tensor_dict = col.OrderedDict()
        batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWS.L_CC] = y_hat[VIEWS.L_CC][:, 0]
        batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWS.L_MLO] = y_hat[VIEWS.L_MLO][:, 0]
        batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWS.R_CC] = y_hat[VIEWS.R_CC][:, 0]
        batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWS.R_MLO] = y_hat[VIEWS.R_MLO][:, 0]
        batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWS.L_CC] = y_hat[VIEWS.L_CC][:, 1]
        batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWS.L_MLO] = y_hat[VIEWS.L_MLO][:, 1]
        batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWS.R_CC] = y_hat[VIEWS.R_CC][:, 1]
        batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWS.R_MLO] = y_hat[VIEWS.R_MLO][:, 1]

        batch_prediction_dict = col.OrderedDict([
            (k, np.exp(v.cpu().detach().numpy()))
            for k, v in batch_prediction_tensor_dict.items()
        ])
    else:
        raise KeyError(mode)
    return batch_prediction_dict


def get_label2classmap(df, label_col):
    """
    Takes in whole annotations dataframe and finds mapping of BIRADs label to unique number used later to find out class
    label in function class2label().
    df: Original data frame of annotations
    label_col : column name containing the annotations/labels.
    """
    labels = []
    tmp = list(set(df[label_col]))
    for l in tmp:
        labels.append(str(l))
    labels.sort()
    label2class = {}
    c = 0
    for l in labels:
        label2class[l] = c
        c += 1
    return label2class


def label2class(df, label2classmap):
    """
    Convert those to class labels based on their BIRADS value to Benign(0) and Malignant(1) class labels, and returns
    converted annotations df. Note it will drop the column in the original form of annotation.
    """
    new_class_list = []
    for l in df['label']:
        new_class_list.append(label2classmap[str(l)])
    classes = [0 if i < 3 else 1 for i in new_class_list]
    df.drop(columns=['label'])
    df['label'] = pd.Series(classes)
    return df


def train(parameters):
    print("Implemented nothing here yet")


def train_splits_breast(parameters):
    """
    Training the classifier on the png data converted from DICOM.
    :return: Nothing
    """
    exam_list = pickling.unpickle_from_file(parameters['data_path'])
    random_number_generator =np.random.RandomState(parameters['seed'])
    n_epochs = parameters['num_epochs']
    annotations_file = parameters['a_file']
    annotations = pd.read_excel(annotations_file, nrows=parameters['num_rows'])
    label2classmap = get_label2classmap(df=annotations, label_col=parameters['label_col'])
    model = SplitBreastModel(1)
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    use_gpu = (parameters["device_type"] == "cuda")

    if use_gpu and torch.has_cudnn:
        device = torch.device("cuda:{}".format(parameters["gpu_number"]))
    else:
        device = torch.device("cpu")
    model = model.to(device)
    for epoch in range(0, n_epochs):
        correct = 0
        cnt = 0
        total_loss = 0
        for datum in tqdm.tqdm(exam_list):

            # Read the images from the dataset.
            predictions_for_datum = []
            loaded_image_dict = {view: [] for view in VIEWS.LIST}
            loaded_heatmaps_dict = {view: [] for view in VIEWS.LIST}
            birads_labels = pd.DataFrame(columns=['image', 'label'])
            for view in VIEWS.LIST:
                for short_file_path in datum[view]:
                    image_path = os.path.join(parameters["image_path"], short_file_path + ".png")
                    loaded_image = preprocess.load_image(
                        image_path=image_path,
                        view=view,
                        horizontal_flip=datum["horizontal_flip"],
                    )

                    if parameters["use_heatmaps"]:
                        loaded_heatmaps = preprocess.load_heatmaps(
                            benign_heatmap_path=os.path.join(parameters["heatmaps_path"], "heatmap_benign",
                                                             short_file_path + ".hdf5"),
                            malignant_heatmap_path=os.path.join(parameters["heatmaps_path"], "heatmap_malignant",
                                                                short_file_path + ".hdf5"),
                            view=view,
                            horizontal_flip=datum["horizontal_flip"],
                        )
                    else:
                        loaded_heatmaps = None
                    loaded_heatmaps_dict[view].append(loaded_heatmaps)
                    loaded_image_dict[view].append(loaded_image)
                    birads_val = \
                    annotations[annotations['File Name'] == int(short_file_path.split('_')[0])]['Bi-Rads'].reset_index(
                        drop=True)[0]
                    birads_labels = birads_labels.append({'image': image_path, 'label': birads_val}, ignore_index=True)
            birads_labels = label2class(df=birads_labels, label2classmap=label2classmap)
            for data_batch in preprocess.partition_batch(range(parameters["num_epochs"]), parameters["batch_size"]):
                batch_dict = {view: [] for view in VIEWS.LIST}
                for _ in data_batch:
                    for view in VIEWS.LIST:
                        image_index = 0
                        if parameters["augmentation"]:
                            image_index = random_number_generator.randint(low=0, high=len(datum[view]))
                        cropped_image, cropped_heatmaps = preprocess.augment_and_normalize_image(
                            image=loaded_image_dict[view][image_index],
                            auxiliary_image=loaded_heatmaps_dict[view][image_index],
                            view=view,
                            best_center=datum["best_center"][view][image_index],
                            random_number_generator=random_number_generator,
                            augmentation=parameters["augmentation"],
                            max_crop_noise=parameters["max_crop_noise"],
                            max_crop_size_noise=parameters["max_crop_size_noise"],
                        )
                        if loaded_heatmaps_dict[view][image_index] is None:
                            batch_dict[view].append(cropped_image[:, :, np.newaxis])
                        else:
                            batch_dict[view].append(np.concatenate([
                                cropped_image[:, :, np.newaxis],
                                cropped_heatmaps,
                            ], axis=2))

                tensor_batch = {
                    view: torch.tensor(np.stack(batch_dict[view])).permute(0, 3, 1, 2).to(device)
                    for view in VIEWS.LIST
                }
                output = model(tensor_batch)
                batch_predictions = compute_batch_predictions(output, mode=parameters["model_mode"])
                pred_df = {}
                for k, v in batch_predictions.items():
                    pred_df[k] = (v[:, 1])
                final_pred = []
                for label in LABELS.LIST:
                    final_pred.append((pred_df[(label, 'CC')] + pred_df[(label, 'MLO')])/2)
                # pred_df = pd.DataFrame({k: v[:, 1] for k, v in batch_predictions_t.items()})
                # pred_df.columns.names = ["label", "view_angle"]
                # predictions = pred_df.T.reset_index().groupby("label").mean().T[LABELS.LIST].values
                # pred_df.columns.names = ["label", "view_angle"]
                # print(pred_df)
                # print(pred_df.T.reset_index())
                # print(pred_df.T.reset_index().groupby("label").mean())
                # print(pred_df.T.reset_index().groupby("label").mean().T[LABELS.LIST].values)
                gt = np.transpose(birads_labels['label'].values.reshape(len(final_pred), 1))
                gt = torch.tensor(gt, dtype=torch.float, requires_grad=True).to(device)
                # predictions = torch.tensor(predictions, requires_grad=True)
                final_pred = torch.tensor(final_pred)
                final_pred = final_pred.reshape(gt.shape)
                l = loss(final_pred, gt)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                print("loss:" + str(l.item()))
                total_loss = total_loss + l.item()
                final_pred = final_pred.clone().detach().numpy() < 0.5
                f_final_pred = np.array(np.zeros(shape=final_pred.shape))
                for i in range(0, final_pred.shape[0]):
                    if (final_pred[0][i]):
                        f_final_pred[i] = 1.0
                    else:
                        f_final_pred[i] = 0.0
                correct += (gt.clone().detach().numpy() == f_final_pred).sum()
                cnt = cnt+1
        print("Epoch:{} Loss:{} Train Accuracy:{}".format(epoch, str(total_loss/cnt), str(correct/cnt)))


def main():
    parser = argparse.ArgumentParser(description='Run image-only model or image+heatmap model')
    # parser.add_argument('--model-mode', default=MODELMODES.VIEW_SPLIT, type=str)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--image-path', required=True)
    # parser.add_argument('--output-path', required=True)
    parser.add_argument('--annotation', required=True)
    parser.add_argument('--num-rows', required=True, type=int)
    parser.add_argument('--label-col', default="Bi-Rads")
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--use-augmentation', action="store_true")
    parser.add_argument('--num-epochs', default=1, type=int)
    parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument("--use_heatmaps", type=bool, default=False)
    parser.add_argument("--model_mode", type=str, default=MODELMODES.VIEW_SPLIT)
    args = parser.parse_args()
    parameters = {
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": args.image_path,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "augmentation": args.use_augmentation,
        "num_epochs": args.num_epochs,
        "data_path": args.data_path,
        "a_file": args.annotation,
        "num_rows": args.num_rows,
        "label_col": args.label_col,
        "use_heatmaps": args.use_heatmaps,
        "model_mode": args.model_mode
    }
    train_splits_breast(parameters = parameters)


if __name__ == "__main__":
    print("HERE")
    main()