from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def get_gps_from_image(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()

    if exif_data is None:
        return None

    gps_info = {}
    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        if tag_name == "GPSInfo":
            for gps_tag in value:
                gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                gps_info[gps_tag_name] = value[gps_tag]

    if not gps_info:
        return None

    # Convert to decimal degrees
    lat = _convert_to_degrees(gps_info.get("GPSLatitude"))
    lon = _convert_to_degrees(gps_info.get("GPSLongitude"))

    if lat is None or lon is None:
        return None

    # Apply hemisphere references
    if gps_info.get("GPSLatitudeRef") == "S":
        lat = -lat
    if gps_info.get("GPSLongitudeRef") == "W":
        lon = -lon

    return {"latitude": lat, "longitude": lon, "raw": gps_info}


def _convert_to_degrees(value):
    if value is None:
        return None

    d, m, s = value
    return float(d) + float(m) / 60 + float(s) / 3600


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gps_extractor.py <image_path>")
        sys.exit(1)

    gps = get_gps_from_image(sys.argv[1])
    if gps:
        print(f"Latitude: {gps['latitude']}, Longitude: {gps['longitude']}")
    else:
        print("No GPS data found")
