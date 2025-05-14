from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, jaccard_score, f1_score
from ultralytics import SAM

sam_model = SAM("sam2_b.pt")


def world_to_voxel(world_coords, origin, spacing):
    """
    Convert world coordinates to voxel coordinates.
    """
    voxel_coords = [(world_coords[i] - origin[i]) / spacing[i] for i in range(3)]
    return voxel_coords


def voxel_to_mask(voxel_coords, diameter, image_shape):
    """
    Create a binary mask for a nodule at given voxel coordinates and diameter.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    x, y = voxel_coords[0], voxel_coords[1]
    radius = int(round(diameter / 2))

    x_min = max(0, x - radius)
    x_max = min(image_shape[1], x + radius)
    y_min = max(0, y - radius)
    y_max = min(image_shape[0], y + radius)

    mask[y_min:y_max, x_min:x_max] = 1
    return mask


def calculate_metrics(ground_truth, prediction):
    """
    Calculate evaluation metrics for segmentation.
    """
    gt_flat = ground_truth.flatten()
    pred_flat = prediction.flatten()

    cm = confusion_matrix(gt_flat, pred_flat)
    accuracy = accuracy_score(gt_flat, pred_flat)
    precision = precision_score(gt_flat, pred_flat, zero_division=1)
    recall = recall_score(gt_flat, pred_flat, zero_division=1)
    jaccard = jaccard_score(gt_flat, pred_flat, zero_division=1)
    dice = f1_score(gt_flat, pred_flat, zero_division=1)

    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "jaccard_index": jaccard,
        "dice_coefficient": dice
    }


def process_annotations_with_sam(annotations_path, dataset_folder, output_folder, sam_device='cpu'):
    """
    Process annotations and save images with bounding boxes and SAM-based segmentation.
    """
    annotations = pd.read_csv(annotations_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    metrics_list = []

    for subset in os.listdir(dataset_folder):
        subset_path = os.path.join(dataset_folder, subset)
        subset_output_folder = os.path.join(output_folder, subset)

        if not os.path.isdir(subset_path):
            continue

        if not os.path.exists(subset_output_folder):
            os.makedirs(subset_output_folder)

        print(f"Processing subset: {subset}")

        for mhd_file in os.listdir(subset_path):
            if not mhd_file.endswith('.mhd'):
                continue

            seriesuid = os.path.splitext(mhd_file)[0]
            mhd_path = os.path.join(subset_path, mhd_file)

            print(f"Processing file: {mhd_path}")

            relevant_annotations = annotations[annotations['seriesuid'] == seriesuid]

            image = sitk.ReadImage(mhd_path)
            array = sitk.GetArrayFromImage(image)
            origin = image.GetOrigin()
            spacing = image.GetSpacing()

            series_output_folder = os.path.join(subset_output_folder, seriesuid)
            if not os.path.exists(series_output_folder):
                os.makedirs(series_output_folder)

            for z in range(array.shape[0]):
                slice_array = array[z]
                slice_array = (slice_array - np.min(slice_array)) / (np.max(slice_array) - np.min(slice_array)) * 255
                slice_array = slice_array.astype(np.uint8)

                img = Image.fromarray(slice_array).convert("RGB")
                slice_ground_truth_mask = np.zeros(slice_array.shape, dtype=np.uint8)

                for idx, row in relevant_annotations.iterrows():
                    coordX, coordY, coordZ, diameter_mm = row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm']
                    voxel_coords = world_to_voxel([coordX, coordY, coordZ], origin, spacing)
                    voxel_coords = [int(round(c)) for c in voxel_coords]

                    if abs(voxel_coords[2] - z) <= int(round(diameter_mm / 2 / spacing[2])):
                        mask = voxel_to_mask(voxel_coords, diameter_mm / spacing[0], slice_array.shape)
                        slice_ground_truth_mask = np.logical_or(slice_ground_truth_mask, mask).astype(np.uint8)

                        x_min = max(0, voxel_coords[0] - int(round(diameter_mm / 2 / spacing[0])))
                        x_max = min(slice_array.shape[1] - 1, voxel_coords[0] + int(round(diameter_mm / 2 / spacing[0])))
                        y_min = max(0, voxel_coords[1] - int(round(diameter_mm / 2 / spacing[0])))
                        y_max = min(slice_array.shape[0] - 1, voxel_coords[1] + int(round(diameter_mm / 2 / spacing[0])))

                        if x_min >= x_max or y_min >= y_max:
                            print(f"Skipping invalid bounding box: {x_min}, {y_min}, {x_max}, {y_max}")
                            continue

                        draw = ImageDraw.Draw(img)
                        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

                        # Run SAM for the bounding box
                        bboxes = [[x_min, y_min, x_max, y_max]]
                        sam_results = sam_model.predict(
                            source=img,
                            bboxes=bboxes,
                            verbose=False,
                            save=False,
                            device=sam_device
                        )

                        for nodule_index, sam_result in enumerate(sam_results):
                            segmented_mask = sam_result.masks.data[0].numpy()

                            metrics = calculate_metrics(slice_ground_truth_mask, segmented_mask)
                            metrics['slice_index'] = z
                            metrics['seriesuid'] = seriesuid
                            metrics['nodule_index'] = nodule_index
                            metrics_list.append(metrics)

                            segmentation_output_path = os.path.join(
                                series_output_folder, f"slice_{z}_nodule_{nodule_index}_segmented.jpg"
                            )
                            sam_result.save(segmentation_output_path)

                slice_output_path = os.path.join(series_output_folder, f"slice_{z}.jpg")
                img.save(slice_output_path)

            print(f"Saved all slices for: {seriesuid}")

    # Save metrics to a CSV file
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(output_folder, 'segmentation_metrics.csv'), index=False)
    print("Saved evaluation metrics.")


# Example usage
annotations_path = 'annotations.csv'
dataset_folder = 'LUNA'
output_folder = 'Segmented_LUNA2'
process_annotations_with_sam(annotations_path, dataset_folder, output_folder, sam_device='cpu')
