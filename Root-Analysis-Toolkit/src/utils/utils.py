import os
import logging
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from typing import Literal, Optional

class LoggerManager:
    """
    LoggerManager class for creating and managing log files with dynamic naming and directory structure.
    Handles automatic enumeration of log files/directories to avoid overwriting and enforces naming conventions.
    
    Args:
        name (str): Base name for the logger and log files.
        context (Literal["app", "dev", "roalt"]): Contextual identifier determining the base log directory.
        enum_log (bool, optional): Whether to enumerate log files (e.g., name_0.log, name_1.log). Defaults to True.
        log_subdir (Optional[str], optional): Subdirectory name within the context base path. Defaults to None.
        enum_subdir (bool, optional): Whether to enumerate subdirectories (e.g., log_subdir_0, log_subdir_1). 
            Only effective if log_subdir is provided. Defaults to False.
    
    Raises:
        ValueError: If the provided `name` already exists as a subdirectory in the base path.
    
    Attributes:
        name (str): Logger name.
        base_path (str): Full path to the base directory for logs (e.g., "./logs/app/subdir_0").
        log_file (str): Full path to the log file (e.g., "./logs/dev/name_1.log").
        logger (logging.Logger): Configured logger instance with file and console handlers.
    
    Notes:
        - Uses `RichHandler` for console output (requires `rich` library).
        - Disables logger propagation to prevent duplicate logging.
        - Automatically creates directories if they don't exist.
    """
    def __init__(
            self,
            name:str,
            context:Literal["app", "dev", "roalt"],
            enum_log:bool=True,
            log_subdir:Optional[str]=None,
            enum_subdir:bool=False
        ):
        self.name = name
        self.base_path = f"./logs/{context}"
        os.makedirs(self.base_path, exist_ok=True)

        if (os.path.isdir(os.path.join(self.base_path, name)) or os.path.isdir(os.path.join(self.base_path, name, "_0"))):
            raise ValueError("Provided name is already being used as subdirectory name, please use another name.")

        if log_subdir is None and enum_log is False:
            self.log_file = f"./logs/{context}/{name}.log"
        
        if log_subdir is None and enum_log is True:
            existing_num = self._get_existing_num(name)
            self.log_file = f"{self.base_path}/{name}_{existing_num + 1}.log"


        if enum_subdir:
            existing_num_dir = self._get_existing_num(log_subdir)
            self.base_path = f"./logs/{context}/{log_subdir}_{existing_num_dir + 1}"
            os.makedirs(self.base_path, exist_ok=True)
        elif not(enum_subdir):
            self.base_path = f"./logs/{context}/{log_subdir}"
            os.makedirs(self.base_path, exist_ok=True)


        if enum_log:
            existing_num_log = self._get_existing_num(name)
            self.log_file = f"{self.base_path}/{name}_{existing_num_log + 1}.log"
        else:
            self.log_file = f"{self.base_path}/{name}.log"
            
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self._setup_handler()

    def _get_existing_num(self, name):
        existing_num = -1
        for entry in os.listdir(self.base_path):
            if entry.startswith(f"{name}_"):
                try:
                    log_num = int(entry.split('_')[1])
                    existing_num = log_num
                except (ValueError, IndexError):
                    continue
        return existing_num

    def _setup_handler(self):
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)

        console_handler = RichHandler(console=Console())
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False