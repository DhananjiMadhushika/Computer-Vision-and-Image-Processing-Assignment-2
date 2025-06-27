#----------------------------------------------------------
# 2. . Implement a region-growing technique for image segmentation. The basic idea is to 
#      start from a set of points inside the object of interest (foreground), denoted as seeds, 
#      and recursively add neighboring pixels as long as they are in a pre-defined range of 
#      the pixel values of the seeds. 
#----------------------------------------------------------

import cv2
import numpy as np
import os

def display_segmentation(segmentation_mask, scale_factor=0.5):

    # Resize for zoom-out view
    height, width = segmentation_mask.shape
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    resized_mask = cv2.resize(segmentation_mask, (new_width, new_height))
    cv2.imshow('Segmentation Process', resized_mask)
    cv2.waitKey(1)

def perform_region_growing_segmentation(input_image, initial_seeds, similarity_threshold):
  
    # Initialize segmentation mask
    segmentation_mask = np.zeros_like(input_image, dtype=np.uint8)
    # Initialize processing queue with seed points
    processing_queue = []
    for seed_coord in initial_seeds:
        processing_queue.append(seed_coord)
    iteration_counter = 0
    
    # Main region growing loop
    while processing_queue:
        iteration_counter += 1
        # Get next pixel to process
        current_pixel = processing_queue.pop(0)
        # Extract pixel intensity at current location
        current_intensity = input_image[current_pixel[1], current_pixel[0]]
        # Mark current pixel as part of segmented region
        segmentation_mask[current_pixel[1], current_pixel[0]] = 255
        # Update display every 20 iterations for better performance
        if iteration_counter % 20 == 0:
            display_segmentation(segmentation_mask)
        
        # Examine 8-connected neighborhood
        for row_offset in range(-1, 2):
            for col_offset in range(-1, 2):
                # Skip center pixel
                if row_offset == 0 and col_offset == 0:
                    continue
                
                # Calculate neighbor coordinates
                neighbor_x = current_pixel[0] + col_offset
                neighbor_y = current_pixel[1] + row_offset
                
                # Validate boundary conditions
                if (0 <= neighbor_x < input_image.shape[1] and 
                    0 <= neighbor_y < input_image.shape[0]):
                    
                    # Get neighbor pixel intensity
                    neighbor_intensity = input_image[neighbor_y, neighbor_x]
                    
                    # Check similarity criterion
                    intensity_difference = np.abs(neighbor_intensity - current_intensity)
                    
                    if (intensity_difference <= similarity_threshold and 
                        segmentation_mask[neighbor_y, neighbor_x] == 0):
                        
                        # Add similar unvisited neighbor to queue
                        processing_queue.append((neighbor_x, neighbor_y))
                        # Mark as visited to avoid reprocessing
                        segmentation_mask[neighbor_y, neighbor_x] = 255
    
    return segmentation_mask

def save_segmentation_results(original_img, segmented_mask, output_directory):

    os.makedirs(output_directory, exist_ok=True)
    
    # Save original image
    cv2.imwrite(os.path.join(output_directory, 'original_image.png'), original_img)
    
    # Save segmentation mask
    cv2.imwrite(os.path.join(output_directory, 'segmented_mask.png'), segmented_mask)
    
    # Create overlay visualization
    overlay_result = cv2.addWeighted(original_img, 0.7, segmented_mask, 0.3, 0)
    cv2.imwrite(os.path.join(output_directory, 'overlay_result.png'), overlay_result)
    
    print(f"Results saved to: {output_directory}")

def display_results_with_zoom(original_img, segmented_mask, zoom_factor=0.6):

    height, width = original_img.shape
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    
    # Resize images for display
    resized_original = cv2.resize(original_img, (new_width, new_height))
    resized_mask = cv2.resize(segmented_mask, (new_width, new_height))
    
    # Create side-by-side comparison
    comparison_view = np.hstack((resized_original, resized_mask))
    
    cv2.imshow('Results: Original (Left) | Segmented (Right)', comparison_view)
    cv2.imshow('Original Image', resized_original)
    cv2.imshow('Segmentation Result', resized_mask)
def main():

    # Load input image in grayscale
    input_image_path = '../input/input.png'
    source_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    if source_image is None:
        print(f"Error: Could not load image from {input_image_path}")
        return

    # Define seed points for region growing
    seed_coordinates = [(385, 395), (450, 780), (620, 190)]

    # Set similarity threshold for pixel grouping
    pixel_threshold = 10

    # Perform segmentation
    final_segmentation = perform_region_growing_segmentation(
        source_image, seed_coordinates, pixel_threshold
    )

    # Prepare output directory
    output_directory = '../result/Q2'
    os.makedirs(output_directory, exist_ok=True)

    # Save original image
    cv2.imwrite(os.path.join(output_directory, 'original_image.png'), source_image)

    # Save final segmented mask
    cv2.imwrite(os.path.join(output_directory, 'segmented_mask.png'), final_segmentation)

    # Save image with seed points marked
    image_with_seeds = cv2.cvtColor(source_image.copy(), cv2.COLOR_GRAY2BGR)
    for x, y in seed_coordinates:
        cv2.circle(image_with_seeds, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # red dot
    cv2.imwrite(os.path.join(output_directory, 'image_with_seeds.png'), image_with_seeds)

    print("Saved: original image, segmented mask, and image with seed points.")


if __name__ == "__main__":
    main()