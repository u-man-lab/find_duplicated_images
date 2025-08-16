# find_duplicated_images

* NOTE: 日本語版のREADMEは[`README-ja.md`](./README-ja.md)を参照。

## Overview

This repository contains two main Python scripts designed to help deduplicate image collections.  
1. The first script, [`group_file_paths_list_by_its_name.py`](#1-group_file_paths_list_by_its_namepy), groups image file paths based on similarities in their file names.
1. The second script, [`find_duplicated_images.py`](#2-find_duplicated_imagespy), analyzes images within predefined groups and detects visually duplicated files, even if they are rotated or resized.

### Disclaimer
The developer assumes no responsibility for any issues that may arise if you delete files based on the duplicate image lists produced by these scripts.  
It is strongly recommended that you manually review the generated CSV outputs and visually confirm that several of the detected duplicates are indeed copies of the original images before taking any irreversible actions such as file deletion.

---

## License & Developer

- **License**: See [`LICENSE`](./LICENSE) in this repository.
- **Developer**: U-MAN Lab. ([https://u-man-lab.com/](https://u-man-lab.com/))

---

## 1. `group_file_paths_list_by_its_name.py`

### 1.1. Description

[`group_file_paths_list_by_its_name.py`](./group_file_paths_list_by_its_name.py) reads a CSV file containing file paths, extracts file names, and groups files whose names share a common substring.

The script appends the following columns to the CSV:

- Assigned group name
- Extracted file name

---

### 1.2. Installation & Usage

#### (1) Install Python

Install Python from the [official Python website](https://www.python.org/downloads/).  
The scripts may not work properly if the version is lower than the verified one. Refer to the [`.python-version`](./.python-version).

#### (2) Clone the repository

```bash
git clone https://github.com/u-man-lab/find_duplicated_images.git
# If you don't have "git", copy the scripts and YAMLs manually to your environment.
cd ./find_duplicated_images
```

#### (3) Install Python dependencies

The scripts may not work properly if the versions are lower than the verified ones.
```bash
pip install --upgrade pip
pip install -r ./requirements.txt
```

#### (4) Prepare a CSV file for input
Prepare a CSV file containing the paths of photo or video files on the PC.  
If you do not have a CSV file, create one by the following method.
```bash
TARGET_FOLDER='<folder where target files are stored>'
find "$TARGET_FOLDER" -type f > ./data/file_paths_list.csv
sed -i '1s/^/file_paths\n/' ./data/file_paths_list.csv  # Add column header
```

#### (5) Edit the configuration file

Open the configuration file [`configs/group_file_paths_list_by_its_name.yaml`](./configs/group_file_paths_list_by_its_name.yaml) and edit the values according to the comments in the file.

#### (6) Run the script

```bash
python ./group_file_paths_list_by_its_name.py ./configs/group_file_paths_list_by_its_name.yaml
```

---

### 1.3. Expected Output

On success, stderr will include logs similar to:

```
2025-08-14 13:56:47,965 [INFO] __main__: "group_file_paths_list_by_its_name.py" start!
2025-08-14 13:56:48,007 [INFO] __main__: Reading CSV file "data/file_paths_list.csv"...
2025-08-14 13:56:48,635 [INFO] __main__: Now grouping...
2025-08-14 14:00:22,192 [INFO] __main__: Filtering to only grouped files...
2025-08-14 14:00:22,504 [INFO] __main__: Writing CSV file "results/file_paths_list_grouped_by_its_name.csv"...
2025-08-14 14:00:23,134 [INFO] __main__: "group_file_paths_list_by_its_name.py" done!
```
For reference, it took about 3m40s to process 33,753 files. Run on Synology DS218 (Realtek RTD1296, 4-core 1.4 GHz, 2GB DDR4) with 2×4TB WD Red (SMR) in RAID1.

The resulting CSV will be like:

```
file_paths,group,file_name
/path1/animal928-img600x380-1320574493ftqrpo80714.jpg,animal928-img600x380-1320574493ftqrpo80714,animal928-img600x380-1320574493ftqrpo80714.jpg
/path2/animal928-img600x380-1320574493ftqrpo80714.jpg,animal928-img600x380-1320574493ftqrpo80714,animal928-img600x380-1320574493ftqrpo80714.jpg
:
```

---

### 1.4. Common Errors

For full details, see the script source. Common errors include:

- **Missing config path argument**
  ```
  2025-08-13 09:46:05,471 [ERROR] __main__: This script needs a config file path as an arg.
  ```
- **Invalid or missing config field**
  ```
  2025-08-13 09:47:40,930 [ERROR] __main__: Failed to parse the config file.: "configs\group_file_paths_list_by_its_name.yaml"
  Traceback (most recent call last):
  :
  ```

---

## 2. `find_duplicated_images.py`

### 2.1. Description

[`find_duplicated_images.py`](./find_duplicated_images.py) reads a CSV file containing grouped image file paths and detects visually duplicated images within each group.  
It supports detection of duplicates even when images are rotated or proportionally resized, using perceptual hashing.

The output includes:
- **A CSV file**: The following columns are appended to the input CSV.
   - Duplicated id: Id of duplicates group in each group.
   - Not-oldest mark: Whether the file is not oldest in each duplicates. (The oldest file can be considered an original image in duplicates.) If there are multiple oldest files within a duplicate group, the file at the top of the input CSV file will be considered the oldest.
- **A TXT file**: List of images which are not considered an original. It is exactly the same as the list of file paths marked "not-oldest" in the CSV file.

For not-oldest marking, serial datetime information for the image files is necessary.  
If you do not have, you can obtain it using `extract_image_taken_datetime.py` in the repository below before running this script.  
https://github.com/u-man-lab/extract_image_taken_datetime/tree/main

---

### 2.2. Usage

Before running, ensure you have already:

- Installed Python
- Cloned the repository
- Installed dependencies

(See [Chapter 1](#1-group_file_paths_list_by_its_namepy) for setup details.)

#### (1) Prepare a CSV file for input
Prepare a CSV file containing the paths of photo or video files on the PC.  
The group and serial datetime of the files is also necessary in other columns.

#### (2) Edit the configuration file

Open the configuration file [`configs/find_duplicated_images.yaml`](./configs/find_duplicated_images.yaml) and edit the values according to the comments in the file.

#### (3) Run the script

```bash
python ./find_duplicated_images.py ./configs/find_duplicated_images.yaml
```

---

### 2.3. Expected Output

On success, stderr will include logs similar to:

```
2025-08-13 23:45:57,760 [INFO] __main__: "find_duplicated_images.py" start!
2025-08-13 23:45:57,773 [INFO] __main__: Reading file "results/file_paths_list_grouped_by_its_name.csv"...
2025-08-13 23:46:02,195 [INFO] __main__: 6910 groups found.
2025-08-13 23:46:02,201 [INFO] __main__: Processing [1/6910]: group "20100823071358"...
2025-08-13 23:46:05,715 [INFO] __main__: Processing [2/6910]: group "20100823071411"...
:
2025-08-14 02:56:25,575 [INFO] __main__: Processing [6910/6910]: group "_DSC0257"...
2025-08-14 02:57:11,720 [INFO] __main__: Writing CSV file "results/file_paths_list_grouped_by_its_name_with_duplicated_id.csv"...
2025-08-14 02:57:13,670 [INFO] __main__: Writing TXT file "results/not_oldest_in_duplicated_file_paths_list.txt"...
2025-08-14 02:57:13,676 [INFO] __main__: "find_duplicated_images.py" done!
```
For reference, it took about 3h11m to process 15,956 files grouped in 6,910 groups. Run on Synology DS218 (Realtek RTD1296, 4-core 1.4 GHz, 2GB DDR4) with 2×4TB WD Red (SMR) in RAID1.

The resulting CSV will be like:

```
file_paths,datetime_tag_by_exiftool,datetime_local_unix,group,file_name,duplicated_id,not_oldest
/path1/animal928-img600x380-1320574493ftqrpo80714.jpg,1330865697.000000,animal928-img600x380-1320574493ftqrpo80714,animal928-img600x380-1320574493ftqrpo80714.jpg,1,X
/path2/animal928-img600x380-1320574493ftqrpo80714.jpg,1330865688.000000,animal928-img600x380-1320574493ftqrpo80714,animal928-img600x380-1320574493ftqrpo80714.jpg,1,
:
```

The resulting TXT will be like:

```
/path1/animal928-img600x380-1320574493ftqrpo80714.jpg
/path1/penticton_river_channel_sc0292.jpg
:
```

---

### 2.4. Common Errors

For full details, see the script source. Common errors are the same as [`group_file_paths_list_by_its_name.py`](#1-group_file_paths_list_by_its_namepy).
