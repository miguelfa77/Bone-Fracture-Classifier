import os
def list_png_files(root_dir):
    png_files = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(subdir, file)
                png_files.append(full_path)
    return png_files