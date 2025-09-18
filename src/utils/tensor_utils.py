import numpy as np
import torch


from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def inspect_shape(data, key_name=None, indent=0) -> None:
    """
    Dynamically inspect the shape, type, and structure of various data types.

    Args:
        data: The data to inspect. It can be a dictionary, list, tuple, tensor, numpy array, etc.
        indent: Current indentation level (used for pretty log.infoing).
        key_name: Optional key name if the data is part of a dictionary.
    """
    prefix = " " * (indent * 4)  # 4 spaces per indent level

    if key_name is not None:
        log.info(f"{prefix}{key_name}:")
        prefix += " " * 4

    # Handle dictionary
    if isinstance(data, dict):
        log.info(f"{prefix}Dictionary with {len(data)} keys:")
        for key, value in data.items():
            inspect_shape(value, indent=indent + 2, key_name=key)

    # Handle numpy array
    elif isinstance(data, np.ndarray):
        log.info(f"{prefix}Numpy Array: shape={data.shape}, dtype={data.dtype}")

    # Handle PyTorch tensor
    elif isinstance(data, torch.Tensor):
        device = data.device if data.is_cuda else "cpu"
        log.info(
            f"{prefix}Torch Tensor: shape={data.shape}, dtype={data.dtype}, device={device}"
        )

    # Handle list or tuple
    elif isinstance(data, (list, tuple)):
        data_type = "List" if isinstance(data, list) else "Tuple"
        log.info(f"{prefix}{data_type} with length={len(data)}:")
        if len(data) > 0:
            log.info(f"{prefix}First element:")
            inspect_shape(data[0], indent=indent + 2)

    # Handle string
    elif isinstance(data, str):
        log.info(f"{prefix}String: {repr(data)}")

    # Handle other types
    else:
        log.info(f"{prefix}Type: {type(data).__name__}, Value: {repr(data)}")


def itemize(data):
    if isinstance(data, dict):
        return {key: itemize(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(itemize(item) for item in data)
    elif isinstance(data, torch.Tensor):
        return data.tolist()
    else:
        return data
