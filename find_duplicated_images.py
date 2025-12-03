import codecs
import os
import random
import sys
from logging import DEBUG, INFO, basicConfig, getLogger
from pathlib import Path
from typing import Any, Final

import imagehash
import pandas as pd
import yaml
from PIL import Image, ImageChops, UnidentifiedImageError
from pillow_heif import register_heif_opener
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    NewPath,
    StrictBool,
    StrictStr,
    field_validator,
)

# Make HEIF files openable by PIL.Image.open()
register_heif_opener()


class EncodingStr:
    """Represents a validated string that must be a valid text encoding name.

    Validates whether the provided string is a supported encoding.
    """

    def __init__(self, value: Any):
        self.__validate_value(value)
        self.__value: str = value

    def __str__(self) -> str:
        return self.__value

    @staticmethod
    def __validate_value(arg: Any) -> str:
        if not isinstance(arg, str):
            raise TypeError(f'The argument must be a string, got "{arg}" [{type(arg)}].')

        try:
            codecs.lookup(arg)
        except LookupError as err:
            raise ValueError(f'"{arg}" is not supported as an encoding string.') from err
        return arg


class PathEncodingConverterMixin:
    """Pydantic mixin for automatic validation and conversion of PATH and ENCODING fields.

    Provides field validators for converting string paths to 'Path' and encoding strings
    to 'EncodingStr' during model initialization.
    """

    @field_validator('PATH', mode='before')
    @classmethod
    def __convert_str_to_file_path_and_validate(cls, arg: Any) -> Path:
        if not isinstance(arg, str):
            raise TypeError(f'The argument must be a string, got "{arg}" [{type(arg)}].')
        return Path(arg.strip())

    @field_validator('ENCODING', mode='before')
    @classmethod
    def __convert_str_to_encoding_str_and_validate(cls, arg: Any) -> EncodingStr:
        if not isinstance(arg, str):
            raise TypeError(f'The argument must be a string, got "{arg}" [{type(arg)}].')
        return EncodingStr(arg.strip())


class GroupedFilePathsListCsvConfig(PathEncodingConverterMixin, BaseModel):
    """Configuration for a CSV file containing a list of file paths and its group.
    'INPUT' > 'GROUPED_FILE_PATHS_LIST_CSV' in YAML

    Attributes:
        PATH: Path to an existing CSV file.
        ENCODING: Encoding used to read the CSV.
        FILE_PATHS_LIST_COLUMN: Column name containing file paths.
        GROUP_COLUMN: Column name containing groups.
        DATETIME_SERIAL_COLUMN: Column name containing serial datetimes.
    """

    PATH: FilePath  # Must be existing file
    ENCODING: EncodingStr
    FILE_PATHS_LIST_COLUMN: StrictStr
    GROUP_COLUMN: StrictStr
    DATETIME_SERIAL_COLUMN: StrictStr

    model_config = ConfigDict(
        frozen=True, extra='forbid', strict=True, arbitrary_types_allowed=True
    )

    def __get_missing_columns(self, df: pd.DataFrame) -> tuple[str, ...]:
        """Returns a tuple of necessary columns that does not exist in the given DataFrame.

        Args:
            df: DataFrame to check columns missing.

        Returns:
            tuple[str, ...]: A tuple of necessary columns that does not exist in the df.
        """

        NECESSARY_COLUMNS: Final[tuple[str, ...]] = (
            self.FILE_PATHS_LIST_COLUMN,
            self.GROUP_COLUMN,
            self.DATETIME_SERIAL_COLUMN,
        )

        return tuple(col for col in NECESSARY_COLUMNS if col not in df.columns)

    def read_csv(self, allow_empty: bool = True) -> pd.DataFrame:
        """Reads the configured CSV file into a pandas DataFrame.

        Args:
            allow_empty: Whether to allow empty rows below header row.

        Returns:
            pd.DataFrame: DataFrame containing the contents of the CSV file.
        """

        getLogger(__name__).info(f'Reading CSV file "{self.PATH}"...')

        df = pd.read_csv(self.PATH, encoding=str(self.ENCODING), dtype=str, keep_default_na=False)

        missing_columns = self.__get_missing_columns(df)
        if missing_columns:
            missing_columns_str = '", "'.join(missing_columns)
            raise ValueError(f'Necessary columns are missing in the CSV.: "{missing_columns_str}"')

        if not allow_empty and df.shape[0] == 0:
            raise ValueError('Empty rows in the CSV.')

        return df


class InputConfig(BaseModel):
    """Input section of the configuration.
    'INPUT' in YAML.

    Attributes:
        GROUPED_FILE_PATHS_LIST_CSV: Configuration for the input file list CSV.
    """

    GROUPED_FILE_PATHS_LIST_CSV: GroupedFilePathsListCsvConfig

    model_config = ConfigDict(frozen=True, extra='forbid', strict=True)


class GroupedFilePathsListWithDuplicatedIdCsvConfig(PathEncodingConverterMixin, BaseModel):
    """Configuration for output CSV containing file paths and its group and duplicated id.
    'OUTPUT' > 'GROUPED_FILE_PATHS_LIST_WITH_DUPLICATED_ID_CSV' in YAML.

    Attributes:
        PATH: Path to a new output CSV file.
        ENCODING: Encoding to use when writing the CSV.
        NOT_OLDEST_MARK_STRING (optional):
            Marking string for "NOT_OLDEST_COLUMN" below (1 length at least). Default to "X".
        ONLY_DUPLICATED: If True, eliminate unduplicated file paths from list.
        DUPLICATED_ID_COLUMN: Column name for duplicated id in group.
        NOT_OLDEST_COLUMN: Column name for whether file is not oldest in duplicates.
            (The oldest file can be considered an original image in duplicates.)
    """

    PATH: NewPath  # Must not exist & parent must exist
    ENCODING: EncodingStr
    NOT_OLDEST_MARK_STRING: StrictStr = Field('X', min_length=1)
    ONLY_DUPLICATED: StrictBool
    DUPLICATED_ID_COLUMN: StrictStr
    NOT_OLDEST_COLUMN: StrictStr

    model_config = ConfigDict(
        frozen=True, extra='forbid', strict=True, arbitrary_types_allowed=True
    )

    def get_already_existing_new_columns(self, df: pd.DataFrame) -> tuple[str, ...]:
        """Returns a tuple of columns that already exist in the given DataFrame
        and would conflict with newly added datetime columns.

        Args:
            df: DataFrame to check for column name conflicts.

        Returns:
            tuple[str, ...]: Existing column names in the df.
        """

        NEW_COLUMNS: Final[tuple[str, ...]] = (
            self.DUPLICATED_ID_COLUMN,
            self.NOT_OLDEST_COLUMN,
        )

        return tuple(col for col in NEW_COLUMNS if col in df.columns)

    def write_csv_from_dataframe(
        self, df: pd.DataFrame, columns: list | None = None, index: bool = True
    ):
        """Writes the given DataFrame to the configured output CSV file.

        Args:
            df: DataFrame to write.
            columns (optional): Specific columns to include. Writes all if None.
            index (optional): Whether to include the DataFrame index. Defaults to True.
        """

        getLogger(__name__).info(f'Writing CSV file "{self.PATH}"...')
        df.to_csv(self.PATH, encoding=str(self.ENCODING), columns=columns, index=index)


class NotOldestInDuplicatedFilePathsListTxtConfig(PathEncodingConverterMixin, BaseModel):
    """Configuration for output TXT containing file paths which is not oldest in duplicates.
    'OUTPUT' > 'NOT_OLDEST_IN_DUPLICATED_FILE_PATHS_LIST_TXT' in YAML.

    Attributes:
        PATH: Path to a new output TXT file.
        ENCODING: Encoding to use when writing the TXT.
    """

    PATH: NewPath  # Must not exist & parent must exist
    ENCODING: EncodingStr

    model_config = ConfigDict(
        frozen=True, extra='forbid', strict=True, arbitrary_types_allowed=True
    )

    def write(self, _str: str):
        """Writes the given string to the TXT file.

        Args:
            _str: String to write.
        """

        getLogger(__name__).info(f'Writing TXT file "{self.PATH}"...')
        with open(self.PATH, 'w', encoding=str(self.ENCODING)) as fw:
            fw.write(_str)


class OutputConfig(BaseModel):
    """Output section of the configuration.
    'OUTPUT' in YAML.

    Attributes:
        GROUPED_FILE_PATHS_LIST_WITH_DUPLICATED_ID_CSV: Configuration for the output CSV file.
        NOT_OLDEST_IN_DUPLICATED_FILE_PATHS_LIST_TXT: Configuration for the output TXT file.
    """

    GROUPED_FILE_PATHS_LIST_WITH_DUPLICATED_ID_CSV: GroupedFilePathsListWithDuplicatedIdCsvConfig
    NOT_OLDEST_IN_DUPLICATED_FILE_PATHS_LIST_TXT: NotOldestInDuplicatedFilePathsListTxtConfig

    model_config = ConfigDict(frozen=True, extra='forbid', strict=True)


class Config(BaseModel):
    """Main configuration object loaded from YAML.

    Attributes:
        INPUT: Input file configuration.
        OUTPUT: Output file configuration.
    """

    INPUT: InputConfig
    OUTPUT: OutputConfig

    model_config = ConfigDict(frozen=True, extra='forbid', strict=True)

    @classmethod
    def from_yaml(cls, path: Path) -> 'Config':
        """Loads the configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Config: Parsed configuration object.
        """

        with open(path, 'r', encoding='utf-8') as fr:
            content = yaml.safe_load(fr)
        return cls(**content)


def __read_arg_config_path():
    """Parses the configuration file path from command-line arguments and loads the config.

    Returns:
        Config: Loaded configuration object.

    Raises:
        SystemExit: If the config path is not provided or cannot be parsed.
    """

    logger = getLogger(__name__)

    if len(sys.argv) != 2:
        logger.error('This script needs a config file path as an arg.')
        sys.exit(1)
    config_path = Path(sys.argv[1])

    try:
        CONFIG: Final[Config] = Config.from_yaml(config_path)
    except Exception:
        logger.exception(f'Failed to parse the config file.: "{config_path}"')
        sys.exit(1)

    return CONFIG


class UnionFind:
    """Disjoint-set (Union-Find) data structure for efficient group merging and lookup.

    Attributes:
        __parent (list[int]): Internal list storing parent references for each element.
    """

    def __init__(self, size: int):
        self.__parent = list(range(size))

    def find(self, x: int) -> int:
        """Finds the representative element (root) of the set containing x.
        Uses path compression to flatten the structure for future lookups.

        Args:
            x (int): Element index.

        Returns:
            int: Representative element index.
        """

        if self.__parent[x] != x:
            self.__parent[x] = self.find(self.__parent[x])
        return self.__parent[x]

    def union(self, x: int, y: int):
        """Unites the sets containing x and y.

        Args:
            x (int): First element index.
            y (int): Second element index.
        """

        px, py = self.find(x), self.find(y)
        if px != py:
            self.__parent[py] = px

    def groups(self) -> list[int]:
        """Gets the representative (root) for each element.

        Returns:
            list[int]: A list of representative indices for each element.
        """

        return [self.find(i) for i in range(len(self.__parent))]


class RotatedAndResizedImage:
    """Wrapper for image comparison that handles rotation and resizing.
    Enables equality checks between images considering possible rotation and
    proportional resizing, using perceptual hashing (pHash) for similarity detection.

    Attributes:
        __PHASH_DIFF_THRESH: Threshold for pHash difference to consider images similar.
        __img (Image.Image): Loaded RGB image.
    """

    # __PHASH_DIFF_THRESH: Final[int] = 8
    __PHASH_DIFF_THRESH: Final[int] = 10
    # set ~8: 9% resized image is judged as a different image.
    # set 20~: totally different images are judged as the same image.

    def __init__(self, file_path: str | Path):
        try:
            self.__img = Image.open(file_path).convert('RGB')
        except UnidentifiedImageError as err:
            raise ValueError(f'This is an unsupported image file.: "{file_path}"') from err

    @staticmethod
    def __is_possibly_resized(
        img1_size: tuple[int, int], img2_size: tuple[int, int], epsilon: float = 0.05
    ):
        """Checks if two image sizes have approximately the same aspect ratio.

        Args:
            img1_size: Width and height of the first image.
            img2_size: Width and height of the second image.
            epsilon (optional): Allowed difference in aspect ratios. Defaults to 0.05.

        Returns:
            bool: True if aspect ratios are within epsilon, False otherwise.

        Raises:
            ValueError: If any dimension is zero or negative.
        """

        w1, h1 = img1_size
        w2, h2 = img2_size
        if w1 <= 0 or w2 <= 0 or h1 <= 0 or h2 <= 0:
            raise ValueError('Invalid image sizes.')
        r1 = w1 / h1
        r2 = w2 / h2
        return abs(r1 - r2) < epsilon

    @staticmethod
    def __get_angles_to_unify_sizes_of_images(
        img1: Image.Image, img2: Image.Image, allow_resize: bool
    ) -> tuple[int, ...]:
        """Determines rotation angles to make two images the same size for comparison.

        Args:
            img1: First image.
            img2: Second image.
            allow_resize: Whether resizing is allowed.

        Returns:
            tuple[int, ...]: Rotation angles (0, 90, 180, 270) that unify the sizes.
        """

        img1_width, img1_height = img1.size
        img2_width, img2_height = img2.size

        # Same aspect and square
        if img1_width == img1_height and img2_width == img2_height:
            if not allow_resize and img1_width != img2_width:
                return tuple()
            return (0, 90, 180, 270)

        # Same aspect
        if RotatedAndResizedImage.__is_possibly_resized(img1.size, img2.size):
            if not allow_resize and img1.size != img2.size:
                return tuple()
            return (0, 180)

        # Same aspect if 90 degrees rotated
        rotated_90degrees_img2_size = (img2_height, img2_width)
        if RotatedAndResizedImage.__is_possibly_resized(img1.size, rotated_90degrees_img2_size):
            if not allow_resize and img1.size != rotated_90degrees_img2_size:
                return tuple()
            return (90, 270)

        # Others
        return tuple()

    @staticmethod
    def __images_are_same_if_rotated(img1: Image.Image, img2: Image.Image, angle: float) -> bool:
        """Checks if two images are identical when one is rotated by a given angle.

        Args:
            img1: First image.
            img2: Second image.
            angle: Rotation angle in degrees.

        Returns:
            bool: True if images are exactly the same after rotation, False otherwise.

        Raises:
            ValueError: If image sizes differ after rotation.
        """

        rotated_img2 = img2.rotate(angle, expand=True)
        if img1.size != rotated_img2.size:
            raise ValueError('Images size must be the same after rotated.')
        diff_img = ImageChops.difference(img1, rotated_img2)
        return diff_img.getbbox() is None

    @staticmethod
    def __images_are_similar_if_rotated_and_resized(
        img1: Image.Image, img2: Image.Image, angle: float, phash_diff_thresh: int
    ) -> bool:
        """Checks if two images are similar when rotated and resized.

        Args:
            img1: First image.
            img2: Second image.
            angle: Rotation angle in degrees.
            phash_diff_thresh: Max pHash difference to consider images similar.

        Returns:
            bool: True if images are similar, False otherwise.
        """

        rotated_img2 = img2.rotate(angle, expand=True)
        smaller_img, larger_img = (
            (img1, rotated_img2) if img1.size[0] < rotated_img2.size[0] else (rotated_img2, img1)
        )
        resized_larger_img = larger_img.resize(smaller_img.size)
        smaller_img_phash = imagehash.phash(smaller_img)
        larger_img_phash = imagehash.phash(resized_larger_img)
        return (larger_img_phash - smaller_img_phash) <= phash_diff_thresh

    def __eq__(self, other: Any) -> bool:
        """Determines if two RotatedAndResizedImage objects are equal.
        Equality is based on whether images match under allowed rotation and resizing.

        Args:
            other: Object to compare with.

        Returns:
            bool: True if equal, False otherwise.
        """

        if not isinstance(other, RotatedAndResizedImage):
            return NotImplemented

        angles_tuple = self.__get_angles_to_unify_sizes_of_images(
            self.__img, other.__img, allow_resize=False
        )
        if angles_tuple:
            for angle in angles_tuple:
                if self.__images_are_same_if_rotated(self.__img, other.__img, angle):
                    return True
            return False

        angles_tuple = self.__get_angles_to_unify_sizes_of_images(
            self.__img, other.__img, allow_resize=True
        )
        if angles_tuple:
            for angle in angles_tuple:
                if self.__images_are_similar_if_rotated_and_resized(
                    self.__img, other.__img, angle, self.__PHASH_DIFF_THRESH
                ):
                    return True
            return False

        return False

    @classmethod
    def assign_duplicated_ids_to_images(
        cls, image_paths_tuple: tuple[str | Path, ...]
    ) -> tuple[str, ...]:
        """Assigns duplicate IDs to a sequence of image paths based on visual similarity.

        Args:
            image_paths_tuple: Paths to image files.

        Returns:
            tuple[str, ...]: Duplicate IDs corresponding to each image path.
        """

        logger = getLogger(__name__)

        images_num = len(image_paths_tuple)
        if images_num == 0:
            return tuple()
        elif images_num == 1:
            return ('',)

        images_list: list[RotatedAndResizedImage | float] = []
        for path in image_paths_tuple:
            try:
                image = cls(path)
            except FileNotFoundError:
                logger.warning(f'FileNotFoundError: "{path}"')
                images_list.append(float('nan'))  # NaN == NaN is always False.
            except ValueError:
                logger.warning(f'Skipped this because an unsupported image file.: "{path}"')
                images_list.append(float('nan'))
            else:
                images_list.append(image)

        uf = UnionFind(images_num)

        for i in range(images_num):
            for j in range(i + 1, images_num):
                if uf.find(i) == uf.find(j):
                    continue
                if images_list[i] == images_list[j]:
                    uf.union(i, j)

        root_ids = uf.groups()
        duplicated_ids = [''] * images_num
        root_id_vs_duplicated_id_dict = {}
        duplicated_id_counter = 1

        for i, root_id in enumerate(root_ids):
            if root_ids.count(root_id) <= 1:
                continue
            if root_id not in root_id_vs_duplicated_id_dict:
                root_id_vs_duplicated_id_dict[root_id] = str(duplicated_id_counter)
                duplicated_id_counter += 1
            duplicated_ids[i] = root_id_vs_duplicated_id_dict[root_id]

        return tuple(duplicated_ids)

    # NOTE: The followings are experimental methods for determining self.__PHASH_DIFF_THRESH.

    # @staticmethod
    # def __get_phash_diff_with_resized_self(
    #     img: Image.Image, resize_ratio_a: float, resize_ratio_b: float
    # ) -> int:
    #     """Calculates the pHash difference after resizing an image within a ratio range.

    #     Args:
    #         img: Image to resize and compare.
    #         resize_ratio_a: Minimum resize ratio.
    #         resize_ratio_b: Maximum resize ratio.

    #     Returns:
    #         int: pHash difference between original and resized image.

    #     Raises:
    #         ValueError: If ratios are not positive.
    #         RuntimeError: If resize results in zero-pixel dimensions.
    #     """

    #     if resize_ratio_a <= 0 or resize_ratio_b <= 0:
    #         raise ValueError('resize_ratio_a/b must be float larger than zero.')
    #     resize_ratio = random.uniform(resize_ratio_a, resize_ratio_b)
    #     resize_size = (int(img.size[0] * resize_ratio), int(img.size[1] * resize_ratio))
    #     if resize_size[0] == 0 or resize_size[1] == 0:
    #         raise RuntimeError('Image cannot be resized to zero pixel.')

    #     once_resized_img = img.resize(resize_size).resize(img.size)
    #     return imagehash.phash(img) - imagehash.phash(once_resized_img)

    # @staticmethod
    # def show_info_to_adjust_phash_diff_thresh(dir_path: str | Path):
    #     """Displays pHash difference statistics to help adjust the similarity threshold.
    #     Iterates over images in a directory, computing differences between resized
    #     versions of the same image and between different images.

    #     Args:
    #         dir_path: Path to a directory containing images.

    #     Raises:
    #         TypeError: If dir_path is not a str or Path.
    #         FileNotFoundError: If the path does not exist.
    #         ValueError: If the path is not a directory or contains no valid images.
    #     """

    #     MIN_RESIZE_RATIO: Final[float] = 0.3
    #     MAX_RESIZE_RATIO: Final[float] = 0.8
    #     SELF_RESIZE_TIMES: Final[int] = 10

    #     if isinstance(dir_path, str):
    #         dir_path_obj = Path(dir_path)
    #     elif isinstance(dir_path, Path):
    #         dir_path_obj = dir_path
    #     else:
    #         raise TypeError(f'Arg must str or Path, but "{type(dir_path)}".')

    #     if not dir_path_obj.exists():
    #         raise FileNotFoundError('Arg path does not exist.')
    #     elif not dir_path_obj.is_dir():
    #         raise ValueError('Arg must be a directory path.')

    #     img_list: list[Image.Image] = []
    #     for sub_path in dir_path_obj.iterdir():
    #         if not sub_path.is_file():
    #             continue
    #         try:
    #             img_list.append(Image.open(sub_path).convert('RGB'))
    #         except UnidentifiedImageError:
    #             print(f'[WARN] "{sub_path}" is not an image file.', flush=True)

    #     img_list_len = len(img_list)
    #     if img_list_len == 0:
    #         raise ValueError('No valid image files in the directory.')
    #     print(f'{img_list_len} image files are loaded.', flush=True)

    #     print('--- Minimum phash diff range ---', flush=True)
    #     diff_list: list[int] = []
    #     for img in img_list:
    #         for i in range(SELF_RESIZE_TIMES):
    #             diff = RotatedAndResizedImage.__get_phash_diff_with_resized_self(
    #                 img, MIN_RESIZE_RATIO, MAX_RESIZE_RATIO
    #             )
    #             diff_list.append(diff)
    #     diff_list_desc = pd.Series(diff_list).describe()
    #     for stat_name in diff_list_desc.index:
    #         print(f'  {stat_name}: {diff_list_desc[stat_name]:.1f}', flush=True)

    #     print('--- Maximun phash diff range ---', flush=True)
    #     diff_list: list[int] = []
    #     for i in range(img_list_len):
    #         for j in range(i + 1, img_list_len):
    #             resized_img_j = img_list[j].resize(img_list[i].size)
    #             diff = imagehash.phash(resized_img_j) - imagehash.phash(img_list[i])
    #             diff_list.append(diff)
    #     diff_list_desc = pd.Series(diff_list).describe()
    #     for stat_name in diff_list_desc.index:
    #         print(f'  {stat_name}: {diff_list_desc[stat_name]:.1f}', flush=True)


def __mark_not_oldest(
    cluster_ids_tuple: tuple[str, ...], datetime_serials_tuple: tuple[float, ...]
) -> tuple[bool, ...]:
    """Marks items that are not the oldest within their cluster.

    Args:
        cluster_ids_tuple: Cluster identifiers for each item.
        datetime_serials_tuple: Corresponding serialized datetime values.

    Returns:
        tuple[bool, ...]: True if item is not the oldest in its cluster, False otherwise.

    Raises:
        ValueError: If input tuples have different lengths.
    """

    if len(cluster_ids_tuple) != len(datetime_serials_tuple):
        raise ValueError('Length of args must be the same.')

    not_oldest = [True] * len(cluster_ids_tuple)

    cluster_id_to_indices: dict[str, list[int]] = {}
    for i, cluster_id in enumerate(cluster_ids_tuple):
        # Blank('') cluster_id is always the oldest
        if cluster_id == '':
            not_oldest[i] = False
            continue

        cluster_id_to_indices.setdefault(cluster_id, []).append(i)

    for cluster_id, indices in cluster_id_to_indices.items():
        if len(indices) == 1:
            not_oldest[indices[0]] = False
            continue

        datetime_min_i = min(indices, key=lambda i: datetime_serials_tuple[i])
        not_oldest[datetime_min_i] = False

    return tuple(not_oldest)


def __get_image_duplication_info_list(
    file_paths_list: list[str], group_list: list[str], datetime_serial_list: list[float]
) -> list[dict[str, str | bool]]:
    """Generates duplicated id and not-oldest bool for images in groups by comparing them.

    Args:
        file_paths_list: File paths of images.
        group_list: Groups of images.
        datetime_serial_list: Serialized datetimes of images.

    Returns:
        list[dict[str, str | bool]]: Duplicated id and not-oldest bool for each images.
    """

    logger = getLogger(__name__)

    lengths = tuple(map(len, (file_paths_list, group_list, datetime_serial_list)))
    if not (lengths[0] == lengths[1] == lengths[2]):
        raise ValueError(f'Length of arg lists are different: {lengths}')

    group_to_index_dict: dict[str, list[int]] = {}
    for i, group in enumerate(group_list):
        group_to_index_dict.setdefault(group, []).append(i)

    total_groups = len(group_to_index_dict)
    logger.info(f'{total_groups} groups in total.')

    image_duplication_info_list: list[dict[str, str | bool]] = [None] * lengths[0]
    for counter, (group, indices_in_group) in enumerate(group_to_index_dict.items()):

        logger.info(f'Processing [{counter + 1}/{total_groups}]: group "{group}"...')

        if group == '':
            for i in indices_in_group:
                image_duplication_info_list[i] = {
                    'duplicated_id': '',
                    'not_oldest': False,
                }
            logger.warning(
                'Skipped group "". This script always assumes that photos in the group "" are unique.'
            )
            continue

        file_paths_in_group = tuple(file_paths_list[i] for i in indices_in_group)
        datetime_serial_in_group = tuple(datetime_serial_list[i] for i in indices_in_group)

        duplicated_ids_in_group = RotatedAndResizedImage.assign_duplicated_ids_to_images(
            file_paths_in_group
        )
        not_oldest_in_group = __mark_not_oldest(duplicated_ids_in_group, datetime_serial_in_group)

        for i, duplicated_id, not_oldest in zip(
            indices_in_group, duplicated_ids_in_group, not_oldest_in_group
        ):
            image_duplication_info_list[i] = {
                'duplicated_id': duplicated_id,
                'not_oldest': not_oldest,
            }

    return image_duplication_info_list


def __find_duplicated_images():

    basicConfig(level=INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    logger = getLogger(__name__)

    logger.info(f'"{os.path.basename(__file__)}" start!')

    CONFIG: Final[Config] = __read_arg_config_path()

    source_csv_config = CONFIG.INPUT.GROUPED_FILE_PATHS_LIST_CSV
    try:
        source_csv_df = source_csv_config.read_csv(allow_empty=False)
    except Exception:
        logger.exception(f'Failed to read the CSV "{source_csv_config.PATH}".')
        sys.exit(1)

    output_csv_config = CONFIG.OUTPUT.GROUPED_FILE_PATHS_LIST_WITH_DUPLICATED_ID_CSV
    already_existing_new_columns = output_csv_config.get_already_existing_new_columns(
        source_csv_df
    )
    if already_existing_new_columns:
        already_existing_new_columns_str = '", "'.join(already_existing_new_columns)
        logger.error(
            f'Tried to create new columns, but already exist in "{CONFIG.INPUT.GROUPED_FILE_PATHS_LIST_CSV.PATH}".'
            + f': "{already_existing_new_columns_str}"'
        )
        sys.exit(1)

    processing_df = source_csv_df.copy()

    group_list = processing_df[source_csv_config.GROUP_COLUMN].to_list()
    file_paths_list = processing_df[source_csv_config.FILE_PATHS_LIST_COLUMN].to_list()
    try:
        datetime_serial_list = (
            processing_df[source_csv_config.DATETIME_SERIAL_COLUMN].apply(float).to_list()
        )
    except ValueError:
        logger.exception('Failed to convert datetime serial into float type.')
        sys.exit(1)

    logger.info('Analyzing images duplication...')

    try:

        image_duplication_info_list = __get_image_duplication_info_list(
            file_paths_list, group_list, datetime_serial_list
        )

        processing_df[output_csv_config.DUPLICATED_ID_COLUMN] = pd.Series(
            [info['duplicated_id'] for info in image_duplication_info_list],
            index=processing_df.index,
        )
        processing_df[output_csv_config.NOT_OLDEST_COLUMN] = pd.Series(
            [
                output_csv_config.NOT_OLDEST_MARK_STRING if info['not_oldest'] else ''
                for info in image_duplication_info_list
            ],
            index=processing_df.index,
        )

    except Exception:
        logger.exception('Failed to analyze images duplication.')
        sys.exit(1)

    if output_csv_config.ONLY_DUPLICATED:

        logger.info('Filtering to only duplicated files...')

        try:

            is_target_row_series = processing_df[output_csv_config.DUPLICATED_ID_COLUMN] != ''
            processing_df = processing_df.loc[is_target_row_series].copy()

            processing_df = processing_df.sort_values(
                [output_csv_config.DUPLICATED_ID_COLUMN, source_csv_config.DATETIME_SERIAL_COLUMN],
                key=lambda col_series: col_series.apply(float),
                kind='stable',
            )
            processing_df = processing_df.sort_values(
                source_csv_config.GROUP_COLUMN, kind='stable'
            )
            processing_df = processing_df.sort_values(
                source_csv_config.GROUP_COLUMN,
                key=lambda col_series: col_series.str.len(),
                ascending=False,
                kind='stable',
            )

        except Exception:
            logger.exception('Failed to filter to only duplicated files.')
            sys.exit(1)

    try:
        output_csv_config.write_csv_from_dataframe(processing_df, index=False)
    except Exception:
        logger.exception(f'Failed to write the CSV "{output_csv_config.PATH}".')
        sys.exit(1)

    is_not_oldest_series = (
        processing_df[output_csv_config.NOT_OLDEST_COLUMN]
        == output_csv_config.NOT_OLDEST_MARK_STRING
    )
    lines_to_write = processing_df.loc[
        is_not_oldest_series, source_csv_config.FILE_PATHS_LIST_COLUMN
    ].tolist()

    output_txt_config = CONFIG.OUTPUT.NOT_OLDEST_IN_DUPLICATED_FILE_PATHS_LIST_TXT
    try:
        output_txt_config.write('\n'.join(lines_to_write))
    except Exception:
        logger.exception(f'Failed to write the TXT "{output_txt_config.PATH}".')
        sys.exit(1)

    logger.info(f'"{os.path.basename(__file__)}" done!')


if __name__ == '__main__':
    __find_duplicated_images()
