from PIL import Image
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
from ultralytics import SAM
from ultralytics.engine.results import Results

sam_model = SAM("sam2_b.pt")

def world_to_voxel(world_coords, origin, spacing):
    """
    Convert world coordinates to voxel coordinates.
    """
    return [(world_coords[i] - origin[i]) / spacing[i] for i in range(3)]

def process_annotations_with_sam(annotations_path, mhd_folder, output_folder, sam_device='cpu'):
    """
    Process annotations and save images with bounding boxes and SAM-based segmentation.
    """
    # Load annotations
    annotations = pd.read_csv(annotations_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for _, row in annotations.iterrows():
        seriesuid = row['seriesuid']
        coordX, coordY, coordZ, diameter_mm = row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm']

        # Find the corresponding .mhd file
        mhd_path = os.path.join(mhd_folder, f"{seriesuid}.mhd")
        if not os.path.exists(mhd_path):
            print(f"File not found: {mhd_path}")
            continue

        print(f"Processing: {mhd_path}")

        # Read the .mhd file
        image = sitk.ReadImage(mhd_path)
        array = sitk.GetArrayFromImage(image)
        origin = image.GetOrigin()
        spacing = image.GetSpacing()

        print(f"Image origin: {origin}")
        print(f"Image spacing: {spacing}")

        # Convert world coordinates to voxel coordinates
        voxel_coords = world_to_voxel([coordX, coordY, coordZ], origin, spacing)
        voxel_coords = [int(round(c)) for c in voxel_coords]
        print(f"World coordinates: ({coordX}, {coordY}, {coordZ})")
        print(f"Voxel coordinates: {voxel_coords}")

        # Extract Z-dimension range affected by the nodule
        z_nodule = voxel_coords[2]
        z_radius = int(round(diameter_mm / 2 / spacing[2]))
        z_start = max(0, z_nodule - z_radius)
        z_end = min(array.shape[0], z_nodule + z_radius + 1)
        print(f"Nodule affects slices: {z_start} to {z_end - 1}")

        radius = int(round(diameter_mm / 2 / spacing[0]))  # Assuming isotropic spacing for X and Y
        print(f"Nodule radius in pixels: {radius}")

        # Create output folder for the seriesuid
        series_output_folder = os.path.join(output_folder, seriesuid)
        if not os.path.exists(series_output_folder):
            os.makedirs(series_output_folder)

        # Process each slice
        for z in range(array.shape[0]):
            slice_array = array[z]

            # Normalize slice for visualization
            slice_array = (slice_array - np.min(slice_array)) / (np.max(slice_array) - np.min(slice_array)) * 255
            slice_array = slice_array.astype(np.uint8)

            # Convert to RGB image
            img = Image.fromarray(slice_array).convert("RGB")
            draw = ImageDraw.Draw(img)

            # Draw bounding box if this slice is within the affected range
            if z_start <= z < z_end:
                # Calculate bounding box coordinates
                x_min = max(0, voxel_coords[0] - radius)
                x_max = min(slice_array.shape[1], voxel_coords[0] + radius)
                y_min = max(0, voxel_coords[1] - radius)
                y_max = min(slice_array.shape[0], voxel_coords[1] + radius)

                # Ensure bounding box coordinates are valid
                if x_min < x_max and y_min < y_max:
                    print(f"Drawing bounding box on slice {z}: ({x_min}, {y_min}, {x_max}, {y_max})")
                    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
                    
                    # Prepare bounding boxes for SAM
                    bboxes = [[x_min, y_min, x_max, y_max]]

                    # Run SAM for segmentation with bounding boxes
                    slice_path = os.path.join(series_output_folder, f"slice_{z}.jpg")
                    img.save(slice_path)  # Save original image with bounding box for reference

                    # Directly pass the image (not the path) to SAM for prediction
                    sam_results = sam_model.predict(
                        source=img,  # Pass the image directly here
                        bboxes=bboxes,
                        verbose=False,
                        save=False,
                        device=sam_device
                    )

                    # Save the SAM segmentation results
                    segmentation_output_path = os.path.join(series_output_folder, f"slice_{z}_segmented.jpg")
                    sam_results[0].save(segmentation_output_path)  # Assuming sam_results returns images
                    print(f"Saved segmentation for slice {z}")

                else:
                    print(f"Invalid bounding box for slice {z}: ({x_min}, {y_min}, {x_max}, {y_max})")

            # Save the image with bounding box
            output_path = os.path.join(series_output_folder, f"slice_{z}.jpg")
            img.save(output_path)

        print(f"Saved slices and segmentations for: {seriesuid}")


# Example usage
annotations_path = 'annotations.csv'
mhd_folder = 'LUNA-short/images'
output_folder = 'output_sam3d_segment'
process_annotations_with_sam(annotations_path, mhd_folder, output_folder, sam_device='cpu')
