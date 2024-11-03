import json
import logging
import os


def GetParamEnv(env_name: str) -> str:
    return os.getenv("runtime.component.parameter." + env_name)


def get_io_filename_from_env(input: bool) -> str:
    host_url = os.getenv("system.stortage")
    if not host_url or not host_url.startswith("file://"):
        host_url = os.getenv("system.storage.host.url")
        if not host_url or not host_url.startswith("file://"):
            return None

    root_path = host_url[6:]
    json_str = (
        os.getenv("runtime.component.input.train_data")
        if input
        else os.getenv("runtime.component.output.train_data")
    )
    if not json_str:
        return None
    json_object = json.loads(json_str)
    relative_path = json_object["namespace"]
    file_name = json_object["name"]

    absolute_path = os.path.join(root_path, relative_path)
    if not input:
        os.makedirs(absolute_path, exist_ok=True)
    logging.info(f"input_filename: {absolute_path}")
    return os.path.join(absolute_path, file_name)


def get_input_filename_from_env() -> str:
    return get_io_filename_from_env(True)


def get_output_filename_from_env() -> str:
    return get_io_filename_from_env(False)


def get_input_filename(defult_file) -> str:
    input_filename = defult_file
    input_optional = get_input_filename_from_env()
    if input_optional is not None:
        input_filename = input_optional
    return input_filename


def get_output_filename(defult_file) -> str:
    output_filename = defult_file
    output_optional = get_output_filename_from_env()
    if output_optional is not None:
        output_filename = output_optional
    return output_filename
