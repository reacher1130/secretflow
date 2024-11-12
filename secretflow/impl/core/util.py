import json
import logging
import os


def str_to_bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError('Cannot convert string to boolean: ' + s)


def GetParamEnv(env_name: str) -> str:
    env_value = os.getenv("runtime.component.parameter." + env_name)
    if env_value is None:
        raise ValueError(
            f"Environment variable 'runtime.component.parameter.{env_name}' not found."
        )
    return env_value


def get_io_filename_from_env(input: bool) -> str:
    host_url = os.getenv("system.storage")
    logging.info("host_url(system.storage): %s", host_url)
    if not host_url or not host_url.startswith("file://"):
        host_url = os.getenv("system.storage.host.url")
        logging.info("host_url(system.storage.host.url): %s", host_url)
        if not host_url or not host_url.startswith("file://"):
            raise ValueError("Unable to determine storage URL")

    root_path = host_url[7:]
    json_str = (
        os.getenv("runtime.component.input.train_data")
        if input
        else os.getenv("runtime.component.output.train_data")
    )
    if not json_str:
        raise ValueError("Unable to determine JSON string for IO data")

    json_object = json.loads(json_str)
    relative_path = json_object["namespace"]
    file_name = json_object["name"]

    absolute_path = os.path.join(root_path, relative_path)
    if not input:
        os.makedirs(absolute_path, exist_ok=True)
    return os.path.join(absolute_path, file_name)


def get_input_filename_from_env() -> str:
    return get_io_filename_from_env(True)


def get_output_filename_from_env() -> str:
    return get_io_filename_from_env(False)


def get_input_filename(default_file) -> str:
    input_filename = default_file
    input_optional = get_input_filename_from_env()
    if input_optional is not None:

        input_filename = input_optional
    return input_filename


def get_output_filename(default_file) -> str:
    output_filename = default_file
    output_optional = get_output_filename_from_env()
    if output_optional is not None:
        output_filename = output_optional
    return output_filename
