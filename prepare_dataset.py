import argparse
import os
import shutil


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parase_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_photo_root', type=str, default='~/train_data/rendered_256x256/256x256/photo/tx_000000000000',
                        help='Source photo root directory')
    parser.add_argument('--source_sketch_root', type=str, default='~/train_data/rendered_256x256/256x256/sketch/tx_000000000000',
                        help='Source sketch root directory')
    parser.add_argument('--target', type=str, default='dataset/',
                        help='Training dataset root directory')
    parser.add_argument('--clean_target', type=str2bool, nargs='?', default=False,
                        help='Remove everything in target root directory')
    parser.add_argument('--test_image_files', type=str, default='test_img.txt',
                        help='File list used as test photo')
    parser.add_argument('--test_sketch_files', type=str, default='test_sketch.txt',
                        help='File list used as test sketch')

    return check_args(parser.parse_args())


"""
Dataset
  ├── photo-train               # the training set of photos
  ├── sketch-triplet-train      # the training set of sketches
  ├── photo-test                # the testing set of photos
  ├── sketch-triplet-test       # the testing set of sketches
"""


def get_target_dirs(root_dir):
    return {
        "photo-train": os.path.join(root_dir, 'photo-train'),
        "sketch-train": os.path.join(root_dir, "sketch-triplet-train"),
        "photo-test": os.path.join(root_dir, "photo-test"),
        "sketch-test": os.path.join(root_dir, "sketch-triplet-test"),
    }


def check_args(args):
    # clean up and create target directories
    if args.clean_target:
        shutil.rmtree(args.target)
    if not os.path.exists(args.target):
        os.mkdir(args.target)
    for child_dir in get_target_dirs(args.target):
        if not os.path.exists(child_dir):
            os.mkdir(child_dir)

    try:
        assert os.path.exists(args.test_image_files)
    except:
        print('test photo/image file list needs to be set')

    try:
        assert os.path.exists(args.test_sketch_files)
    except:
        print('test sketch file list needs to be set')

    return args


def adjust_files(photo_root, sketch_root, other_photo_root):
    # Walk through all files in the source directory
    for root, _, files in os.walk(photo_root):
        for file_name in files:
            # Get the relative path of the file from the source_dir
            relative_path = os.path.relpath(os.path.join(root, file_name), photo_root)

            # Determine destination based on whether the file is in file_set
            paths = relative_path.split('/')
            fname = paths[-1].split('.')[0]
            cname = paths[-2]
            sketchs = sorted(os.listdir(os.path.join(sketch_root, cname)))

            sketch_rel = []
            for sketch_name in sketchs:
                if sketch_name.split('-')[0] == fname:
                    sketch_rel.append(sketch_name)

            if len(sketch_rel) < 1:
                # If there is no sketch files for the photo, remove the photo or move back to test
                # Create the destination directory if it doesn't exist
                dest_path = os.path.join(other_photo_root, relative_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # Move the file
                shutil.move(os.path.join(root, file_name), dest_path)
                print(f"moving file: {os.path.join(root, file_name)} -> {dest_path}")


def copy_files(source_dir, train_dir, test_dir, file_list_path):
    """
    Split files in source_dir to train_dir and test_dir, copying files listed in file_list_path to test_dir
    and the rest to train_dir, preserving the sub-directory structure.

    Parameters:
    source_dir (str): The root directory containing all files and sub-directories.
    train_dir (str): The directory where training files will be copied.
    test_dir (str): The directory where test files (specified in file_list_path) will be copied.
    file_list_path (str): Path to a file containing a list of file paths (relative to source_dir)
                          that should be copied to test_dir.
    """
    # Read file paths from file_list_path and convert them to a set for efficient lookup
    with open(file_list_path, 'r') as f:
        file_set = set(line.strip() for line in f if line.strip())

    # Walk through all files in the source directory
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            # Get the relative path of the file from the source_dir
            relative_path = os.path.relpath(os.path.join(root, file_name), source_dir)

            # Determine destination based on whether the file is in file_set
            if relative_path in file_set:
                dest_dir = test_dir
            else:
                dest_dir = train_dir

            # Create the destination directory if it doesn't exist
            dest_path = os.path.join(dest_dir, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Copy the file
            shutil.copy2(os.path.join(root, file_name), dest_path)

if __name__ == "__main__":
    print("An example command in multiple lines")
    print("""
        python prepare_dataset.py \\
            --source_photo_root /Users/minglirui/train_data/rendered_256x256/256x256/photo/tx_000000000000 \\
            --source_sketch_root /Users/minglirui/train_data/rendered_256x256/256x256/sketch/tx_000000000000 \\
            --target /tmp/dataset \\
            --clean_target false \\
            --test_image_files test_img.txt \\
            --test_sketch_files test_sketch.txt
    """)

    print("An example command in one line")
    print("python prepare_dataset.py  --source_photo_root /Users/minglirui/train_data/rendered_256x256/256x256/photo/tx_000000000000 --source_sketch_root /Users/minglirui/train_data/rendered_256x256/256x256/sketch/tx_000000000000 --target /tmp/dataset --clean_target false --test_image_files test_img.txt --test_sketch_files test_sketch.txt")

    args = parase_args()
    if args is None:
        exit()
    target_dirs = get_target_dirs(args.target)
    copy_files(args.source_photo_root, target_dirs["photo-train"], target_dirs["photo-test"], args.test_image_files)
    copy_files(args.source_sketch_root, target_dirs["sketch-train"], target_dirs["sketch-test"], args.test_sketch_files)
    # The test_photo file and
    adjust_files(target_dirs["photo-train"], target_dirs["sketch-train"], target_dirs["photo-test"])
    adjust_files(target_dirs["photo-test"], target_dirs["sketch-test"], target_dirs["photo-train"])

    print("Arguments for train.py are:")
    print(f"""
    --photo_root {target_dirs["photo-train"]} \\
    --sketch_root {target_dirs["sketch-train"]} \\
    --photo_test  {target_dirs["photo-test"]} \\
    --sketch_test {target_dirs["sketch-test"]} \\
    """)

