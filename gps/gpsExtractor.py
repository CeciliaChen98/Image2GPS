import argparse
import os
import csv
import shutil
import random
import exifread


def get_exif_data(image_path):
    """Extract EXIF data from an image file."""
    with open(image_path, 'rb') as image_file:
        tags = exifread.process_file(image_file)
    return tags


def convert_to_decimal_degrees(value):
    """Convert GPS coordinates in degrees, minutes, and seconds to decimal degrees."""
    d, m, s = value.values
    return d.num / d.den + (m.num / m.den) / 60 + (s.num / s.den) / 3600


def extract_gps_from_image(filepath):
    """Extract GPS coordinates from a single image file."""
    exif_data = get_exif_data(filepath)
    if exif_data:
        gps_latitude = exif_data.get('GPS GPSLatitude', None)
        gps_latitude_ref = exif_data.get('GPS GPSLatitudeRef', None)
        gps_longitude = exif_data.get('GPS GPSLongitude', None)
        gps_longitude_ref = exif_data.get('GPS GPSLongitudeRef', None)

        if gps_latitude and gps_longitude:
            latitude = convert_to_decimal_degrees(gps_latitude)
            longitude = convert_to_decimal_degrees(gps_longitude)

            if gps_latitude_ref.values[0] == 'S':
                latitude = -latitude
            if gps_longitude_ref.values[0] == 'W':
                longitude = -longitude

            return latitude, longitude
    return None


def write_metadata_csv(output_csv, data):
    """Write GPS metadata to a CSV file."""
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['file_name', 'Latitude', 'Longitude']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            writer.writerow(entry)


def process_images(image_folder, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Process images, split into train/val/test, and extract GPS coordinates."""
    random.seed(seed)

    # Collect all valid images with GPS data
    valid_images = []
    for filename in os.listdir(image_folder):
        filepath = os.path.join(image_folder, filename)
        if os.path.isfile(filepath):
            try:
                gps_coords = extract_gps_from_image(filepath)
                if gps_coords:
                    valid_images.append({
                        'filename': filename,
                        'filepath': filepath,
                        'latitude': gps_coords[0],
                        'longitude': gps_coords[1]
                    })
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Found {len(valid_images)} images with GPS data")

    # Shuffle the images
    random.shuffle(valid_images)

    # Calculate split indices
    total = len(valid_images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': valid_images[:train_end],
        'validation': valid_images[train_end:val_end],
        'test': valid_images[val_end:]
    }

    # Create output directories and process each split
    for split_name, split_data in splits.items():
        split_dir = os.path.join(output_folder, split_name)
        os.makedirs(split_dir, exist_ok=True)

        metadata = []
        for img in split_data:
            # Copy image to split directory
            dest_path = os.path.join(split_dir, img['filename'])
            shutil.copy2(img['filepath'], dest_path)

            metadata.append({
                'file_name': img['filename'],
                'Latitude': img['latitude'],
                'Longitude': img['longitude']
            })

        # Write metadata CSV for this split
        csv_path = os.path.join(split_dir, 'metadata.csv')
        write_metadata_csv(csv_path, metadata)

        print(f"{split_name}: {len(split_data)} images -> {split_dir}")

    print(f"\nData split complete. Output saved to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract GPS coordinates from images and split into train/val/test")
    parser.add_argument("image_folder", help="Path to folder containing images")
    parser.add_argument("-o", "--output", default="dataset", help="Output folder for split data (default: dataset)")
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--val", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    args = parser.parse_args()

    # Validate ratios
    if abs(args.train + args.val + args.test - 1.0) > 0.001:
        print("Error: Train, validation, and test ratios must sum to 1.0")
        exit(1)

    process_images(args.image_folder, args.output, args.train, args.val, args.test, args.seed)