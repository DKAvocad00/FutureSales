import os
import platform
from functools import wraps
from typing import Callable


def validate_directory(expected_directory: str) -> Callable:
    """
    **Decorator to ensure the existence of a directory.**

    This decorator checks if the specified directory exists and creates it if it doesn't.

    :param expected_directory: The path to the directory that should exist or be created.
    :return: The decorator function.
    """

    def decorator(func: Callable) -> Callable:
        """
        **Inner decorator function.**

        :param func: The function to be decorated.
        :return: The wrapper function.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            """
            **Wrapper function to validate directory existence and create if necessary.**

            This wrapper function ensures that the specified directory exists. If it doesn't, it creates it.
            Additionally, it prints informative messages about the directory status.

            :param self: The class instance.
            :param args: Positional arguments passed to the method.
            :param kwargs: Keyword arguments passed to the method.
            :return: The result of the decorated function.
            """
            try:
                if not os.path.exists(expected_directory):
                    os.makedirs(expected_directory)
                    print(f"[INFO]: A directory has been created: {expected_directory}")
                else:
                    print(f"[INFO]: The {expected_directory} directory already exists.")
            except Exception as e:
                print(f"[ERROR]: An error occurred while creating the directory: {e}")
                raise

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class DriveUtils:
    """

    This class provides methods to download data and models from Google Drive using their respective folder IDs.

    """

    def __init__(self) -> None:
        """
        **Initialize DriveUtils object.**

        Sets up folder IDs and output folder names for data and models.

        :return: None
        """

        self.data_folder_id = '1zHvR-nds_IVJV3ZT9XERoUdgtq_UIVkF?usp=drive_link'
        self.data_output_folder = "data"

        self.models_folder_id = '1XjnZ927RlbuBpxwjUab0mKJe63hvaeKq?usp=drive_link'
        self.models_output_folder = "models"

    @staticmethod
    def _download_from_google_drive(folder_id: str, output_folder: str) -> None:
        """
        **Download files from Google Drive folder.**

        This method downloads files from the specified Google Drive folder using its folder ID.

        :param folder_id: The ID of the Google Drive folder.
        :param output_folder: The local output folder where files will be downloaded.

        :return: None
        """
        current_dir = os.getcwd()
        output_dir = os.path.join(current_dir, output_folder)
        output_dir = output_dir if platform.system() != "Windows" else current_dir.replace("\\", "/")
        command = f"gdown --folder https://drive.google.com/drive/folders/{folder_id} -O {output_dir}"
        os.system(command)

    @validate_directory("data")
    def data_download_from_google_drive(self) -> None:
        """
        **Download data from Google Drive.**

        This method downloads data from Google Drive using the provided folder ID and saves it to the specified output folder.

        :return: None
        """
        self._download_from_google_drive(self.data_folder_id, self.data_output_folder)

    @validate_directory("models")
    def models_download_from_google_drive(self) -> None:
        """
        **Download models from Google Drive.**

        This method downloads models from Google Drive using the provided folder ID and saves them to the specified output folder.

        :return: None
        """
        self._download_from_google_drive(self.models_folder_id, self.models_output_folder)
