import codecs
import os
import sys
from logging import DEBUG, INFO, basicConfig, getLogger
from pathlib import Path
from typing import Any, Final

import pandas as pd
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    FilePath,
    NewPath,
    PositiveInt,
    StrictBool,
    StrictStr,
    field_validator,
)


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


class FilePathsListCsvConfig(PathEncodingConverterMixin, BaseModel):
    """Configuration for a CSV file containing a list of file paths.
    'INPUT' > 'FILE_PATHS_LIST_CSV' in YAML

    Attributes:
        PATH: Path to an existing CSV file.
        ENCODING: Encoding used to read the CSV.
        FILE_PATHS_LIST_COLUMN: Column name containing file paths.
    """

    PATH: FilePath  # Must be existing file
    ENCODING: EncodingStr
    FILE_PATHS_LIST_COLUMN: StrictStr

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

        NECESSARY_COLUMNS: Final[tuple[str, ...]] = (self.FILE_PATHS_LIST_COLUMN,)

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
        FILE_PATHS_LIST_CSV: Configuration for the input file list CSV.
    """

    FILE_PATHS_LIST_CSV: FilePathsListCsvConfig

    model_config = ConfigDict(frozen=True, extra='forbid', strict=True)


class ProcessConfig(BaseModel):
    """Processing section of the configuration.
    'PROCESS' in YAML.

    Attributes:
        MIN_GROUP_NAME_LENGTH: Minimum number of characters for group name.
    """

    MIN_GROUP_NAME_LENGTH: PositiveInt

    model_config = ConfigDict(frozen=True, extra='forbid', strict=True)


class FilePathsListGroupedByItsNameCsvConfig(PathEncodingConverterMixin, BaseModel):
    """Configuration for output CSV containing file paths and its group.
    'OUTPUT' > 'FILE_PATHS_LIST_GROUPED_BY_ITS_NAME_CSV' in YAML.

    Attributes:
        PATH: Path to a new output CSV file.
        ENCODING: Encoding to use when writing the CSV.
        ONLY_GROUPED: If True, sort by group and eliminate ungrouped file paths from list.
        GROUP_COLUMN: Column name for group.
        FILE_NAME_COLUMN: Column name for file name.
    """

    PATH: NewPath  # Must not exist & parent must exist
    ENCODING: EncodingStr
    ONLY_GROUPED: StrictBool
    GROUP_COLUMN: StrictStr
    FILE_NAME_COLUMN: StrictStr

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
            self.GROUP_COLUMN,
            self.FILE_NAME_COLUMN,
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


class OutputConfig(BaseModel):
    """Output section of the configuration.
    'OUTPUT' in YAML.

    Attributes:
        FILE_PATHS_LIST_GROUPED_BY_ITS_NAME_CSV: Configuration for the output CSV file.
    """

    FILE_PATHS_LIST_GROUPED_BY_ITS_NAME_CSV: FilePathsListGroupedByItsNameCsvConfig

    model_config = ConfigDict(frozen=True, extra='forbid', strict=True)


class Config(BaseModel):
    """Main configuration object loaded from YAML.

    Attributes:
        INPUT: Input file configuration.
        PROCESS: Processing parameters configuration.
        OUTPUT: Output file configuration.
    """

    INPUT: InputConfig
    PROCESS: ProcessConfig
    OUTPUT: OutputConfig

    model_config = ConfigDict(frozen=True, extra='forbid', strict=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'Config':
        """Loads the configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Config: Parsed configuration object.
        """

        with open(path, 'r', encoding='utf-8') as fr:
            content = yaml.safe_load(fr)
        return cls(**content)


def __read_arg_config_path() -> Config:
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


def __group_file_names_list(file_names_list: list[str], min_group_name_length: int) -> list[str]:
    """Groups file names by whether a name include another name.

    Args:
        file_names_list: List of file names.
        min_group_name_length: Minimum number of characters for group name.

    Returns:
        list[str]: List of groups of input file names.
    """

    file_names_without_ext_list = [os.path.splitext(file_name)[0] for file_name in file_names_list]

    shorter_unique_file_names_without_ext_list = sorted(
        set(file_names_without_ext_list),
        key=lambda file_name_without_ext: [len(file_name_without_ext), file_name_without_ext],
    )

    file_name_without_ext_to_group_dict: dict[str, str] = {}
    for group_name in shorter_unique_file_names_without_ext_list:

        if len(group_name) < min_group_name_length:
            continue

        members_list = [
            file_name_without_ext
            for file_name_without_ext in file_names_without_ext_list
            if group_name in file_name_without_ext
        ]

        if len(members_list) >= 2:  # At least 1 other than me
            members_to_group_dict = dict.fromkeys(members_list, group_name)
            file_name_without_ext_to_group_dict.update(members_to_group_dict)

    return [
        file_name_without_ext_to_group_dict.get(file_name_without_ext, '')
        for file_name_without_ext in file_names_without_ext_list
    ]


def __group_file_paths_list_by_its_name():

    basicConfig(level=INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    logger = getLogger(__name__)

    logger.info(f'"{os.path.basename(__file__)}" start!')

    CONFIG: Final[Config] = __read_arg_config_path()

    source_csv_config = CONFIG.INPUT.FILE_PATHS_LIST_CSV
    try:
        source_csv_df = source_csv_config.read_csv(allow_empty=False)
    except Exception:
        logger.exception(f'Failed to read the CSV "{source_csv_config.PATH}".')
        sys.exit(1)

    output_csv_config = CONFIG.OUTPUT.FILE_PATHS_LIST_GROUPED_BY_ITS_NAME_CSV
    already_existing_new_columns = output_csv_config.get_already_existing_new_columns(
        source_csv_df
    )
    if already_existing_new_columns:
        already_existing_new_columns_str = '", "'.join(already_existing_new_columns)
        logger.error(
            f'Tried to create new columns, but already exist in "{CONFIG.INPUT.FILE_PATHS_LIST_CSV.PATH}".'
            + f': "{already_existing_new_columns_str}"'
        )
        sys.exit(1)

    processing_df = source_csv_df.copy()

    logger.info('Now grouping...')
    try:

        file_name_series = processing_df[source_csv_config.FILE_PATHS_LIST_COLUMN].apply(
            lambda path_str: Path(path_str).name
        )

        group_list = __group_file_names_list(
            file_name_series.to_list(), CONFIG.PROCESS.MIN_GROUP_NAME_LENGTH
        )
        group_series = pd.Series(group_list, index=processing_df.index)

        processing_df[output_csv_config.GROUP_COLUMN] = group_series
        processing_df[output_csv_config.FILE_NAME_COLUMN] = file_name_series

    except Exception:
        logger.exception('Failed to group files.')
        sys.exit(1)

    if output_csv_config.ONLY_GROUPED:

        logger.info('Filtering to only grouped files...')

        try:

            processing_df = processing_df.loc[
                group_series.duplicated(keep=False) & (group_series != '')
            ].copy()

            processing_df = processing_df.sort_values(
                [output_csv_config.GROUP_COLUMN, output_csv_config.FILE_NAME_COLUMN], kind='stable'
            )
            processing_df = processing_df.sort_values(
                output_csv_config.GROUP_COLUMN,
                key=lambda col_series: col_series.str.len(),
                ascending=False,
                kind='stable',
            )

        except Exception:
            logger.exception('Failed to filter to only grouped files.')
            sys.exit(1)

    try:
        output_csv_config.write_csv_from_dataframe(processing_df, index=False)
    except Exception:
        logger.exception(f'Failed to write the CSV "{output_csv_config.PATH}".')
        sys.exit(1)

    logger.info(f'"{os.path.basename(__file__)}" done!')


if __name__ == '__main__':
    __group_file_paths_list_by_its_name()
