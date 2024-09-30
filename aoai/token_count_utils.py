import os
import re
import json
import tiktoken
import numpy as np
from typing import Dict, List, Union
from collections import defaultdict
#logger = logging.getLogger(__name__)
from logger import logger


def validate_json(parsed_data) -> Union[bool, str]:
    """
    Validate the parsed JSON data to ensure it has the required keys and values.
    
    Args:
        parsed_data: (dict): Parsed JSON data.

    Returns:
        bool: True if the JSON data is valid, False otherwise.
        message: str: Error message if the JSON data is invalid.    
    """
    try:
        # Check if 'messages' key exists and is a list
        if "messages" not in parsed_data or not isinstance(parsed_data["messages"], list):
            logger.warning("Invalid JSON: 'messages' key is missing or not a list.")
            return False, "missing_messages_list"

        # Check if each message has the required keys according to its 'role'
        for message in parsed_data["messages"]:
            if not isinstance(message, dict):
                logger.warning(f"Invalid JSON: Each message should be a dictionary. Found: {type(message)}")
                return False, "missing_message_dict"

            # Check if 'role' key exists and is of type string
            if "role" not in message or not isinstance(message["role"], str):
                logger.warning(f"Invalid JSON: Each message should contain a 'role' key of type string.")
                return False, "missing_role_key"

            # Check required keys based on role
            role = message["role"]
            
            if role == "system":
                # 'system' role must have 'content'
                if "content" not in message or not isinstance(message["content"], str):
                    logger.warning(f"Invalid JSON: 'system' role must have a 'content' key of type string.")
                    return False, "content_key_missing"
                if not message["content"].strip():
                    logger.warning("Invalid JSON: 'system' role 'content' cannot be empty or only whitespace.")
                    return False, "content_empty"
            
            elif role == "user":
                # 'user' role must have 'content'
                if "content" not in message or not isinstance(message["content"], str):
                    logger.warning(f"Invalid JSON: 'user' role must have a 'content' key of type string.")
                    return False, "content_key_missing"
                if not message["content"].strip():
                    logger.warning("Invalid JSON: 'user' role 'content' cannot be empty or only whitespace.")
                    return False, "content_empty"
            
            elif role == "assistant":
                # The 'assistant' role must have at least one of 'content' or 'tool_calls'
                if ("content" not in message or not isinstance(message["content"], str)) and ("tool_calls" not in message):
                    logger.warning(f"Invalid JSON: 'assistant' role must have either 'content' or 'tool_calls'.")
                    return False, "content_or_tool_calls_missing"
                if "content" in message:
                    if not message["content"].strip():
                        logger.warning("Invalid JSON: 'assistant' role 'content' cannot be empty or only whitespace.")
                        return False, "content_empty"
                if "tool_calls" in message and not isinstance(message["tool_calls"], list):
                    logger.warning(f"Invalid JSON: 'tool_calls' must be a list if provided.")
                    return False, "tool_calls_not_list"

            elif role == "tool":
                # 'tool' role must have a 'tool_call_id'
                if "tool_call_id" not in message or not isinstance(message["tool_call_id"], str):
                    logger.warning(f"Invalid JSON: 'tool' role must have a 'tool_call_id' key of type string.")
                    return False, "tool_call_id_missing"
            
            else:
                logger.warning(f"Invalid JSON: Unknown role '{role}'.")
                return False, "unknown_role"

        # Validate 'tools' key (if necessary)
        if "tools" in parsed_data and not isinstance(parsed_data["tools"], list):
            logger.warning(f"Invalid JSON: 'tools' key must be a list if present.")
            return False, "tools_not_list"
        
        return True, "passed"

    except Exception as e:
        logger.warning(f"Error during validation: {e}")
        return False, "error_during_validation"


def validate_jsonl(jsonl_files):

    for jsonl_path in jsonl_files:
            
        # Format error checks
        format_errors = defaultdict(int)
        dataset = []
        logger.info('*' * 50)
        logger.info(f"### [JSONL_VALIDATION] Processing file: {jsonl_path}")        

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, start=1):
                try:
                    parsed_data = json.loads(line)
                    dataset.append(parsed_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {idx}: Invalid JSON format - {e}")
                except Exception as e:
                    logger.warning(f"Line {idx}: Unexpected error - {e}")

        for idx, data in enumerate(dataset):
            is_valid, error_key = validate_json(data)
            if not is_valid:
                logger.warning(f"Validation failed for line {idx + 1}")
                format_errors[error_key] += 1

        if format_errors:
            for k, v in format_errors.items():
                logger.info(f"{k}: {v}")
        else:
            logger.info(f"{jsonl_path}: All examples are valid")
        logger.info('*' * 50)


def get_max_token_limit(model: str = "gpt-3.5-turbo-0613") -> int:
    # Handle common azure model names/aliases
    model = re.sub(r"^gpt\-?35", "gpt-3.5", model)
    model = re.sub(r"^gpt4", "gpt-4", model)

    max_token_limit = {
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-0125": 16385,
        "gpt-3.5-turbo-0301": 4096,
        "gpt-3.5-turbo-0613": 4096,
        "gpt-3.5-turbo-instruct": 4096,
        "gpt-3.5-turbo-16k": 16385,
        "gpt-3.5-turbo-16k-0613": 16385,
        "gpt-3.5-turbo-1106": 16385,
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-2024-04-09": 128000,
        "gpt-4-32k": 32768,
        "gpt-4-32k-0314": 32768,  # deprecate in Sep
        "gpt-4-0314": 8192,  # deprecate in Sep
        "gpt-4-0613": 8192,
        "gpt-4-32k-0613": 32768,
        "gpt-4-1106-preview": 128000,
        "gpt-4-0125-preview": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-4-vision-preview": 128000,
        "gpt-4o": 128000,
        "gpt-4o-2024-05-13": 128000,
        "gpt-4o-2024-08-06": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4o-mini-2024-07-18": 128000,
    }
    return max_token_limit[model]


def percentile_used(input, model="gpt-3.5-turbo-0613"):
    return count_token(input) / get_max_token_limit(model)


def token_left(input: Union[str, List, Dict], model="gpt-3.5-turbo-0613") -> int:
    """Count number of tokens left for an OpenAI model.

    Args:
        input: (str, list, dict): Input to the model.
        model: (str): Model name.

    Returns:
        int: Number of tokens left that the model can use for completion.
    """
    return get_max_token_limit(model) - count_token(input, model=model)


def count_token(input: Union[str, List, Dict], model: str = "gpt-3.5-turbo-0613") -> int:
    """Count number of tokens used by an OpenAI model.
    Args:
        input: (str, list, dict): Input to the model.
        model: (str): Model name.

    Returns:
        int: Number of tokens from the input.
    """
    if isinstance(input, str):
        return _num_token_from_text(input, model=model)
    elif isinstance(input, list) or isinstance(input, dict):
        return _num_token_from_messages(input, model=model)
    else:
        raise ValueError(f"input must be str, list or dict, but we got {type(input)}")


def _num_token_from_text(text: str, model: str = "gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(f"Model {model} not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _num_token_from_messages(messages: Union[List, Dict], model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages.

    retrieved from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb/
    """
    if isinstance(messages, dict):
        messages = [messages]

    if "gpt-4o" in model:
        encoding = tiktoken.get_encoding("o200k_base")      
    else:
        encoding = tiktoken.get_encoding("cl100k_base")

    if model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    else:
        tokens_per_message = 3
        tokens_per_name = 1
        
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if value is None:
                continue

            # function calls
            if not isinstance(value, str):
                try:
                    value = json.dumps(value)
                except TypeError:
                    logger.warning(
                        f"Value {value} is not a string and cannot be converted to json. It is a type: {type(value)} Skipping."
                    )
                    continue

            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def num_assistant_tokens_from_messages(messages, model="gpt-3.5-turbo-0613") -> int:
    
    if "gpt-4o" in model:
        encoding = tiktoken.get_encoding("o200k_base")      
    else:
        encoding = tiktoken.get_encoding("cl100k_base")
            
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            if "content" in message:
                num_tokens += len(encoding.encode(message["content"]))
    return num_tokens


def num_tokens_from_functions(functions, model="gpt-3.5-turbo-0613") -> int:
    """Return the number of tokens used by a list of functions.

    Args:
        functions: (list): List of function descriptions that will be passed in model.
        model: (str): Model name.

    Returns:
        int: Number of tokens from the function descriptions.
    """
    if "gpt-4o" in model:
        encoding = tiktoken.get_encoding("o200k_base")      
    else:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for f in functions:
        if 'function' in f:
            function = f["function"]
        else:
            function = f
        function_tokens = len(encoding.encode(function["name"]))
        function_tokens += len(encoding.encode(function["description"]))
        function_tokens -= 2
        if "parameters" in function:
            parameters = function["parameters"]
            if "properties" in parameters:
                for propertiesKey in parameters["properties"]:
                    function_tokens += len(encoding.encode(propertiesKey))
                    v = parameters["properties"][propertiesKey]
                    for field in v:
                        if field == "type":
                            function_tokens += 2
                            function_tokens += len(encoding.encode(v["type"]))
                        elif field == "description":
                            function_tokens += 2
                            function_tokens += len(encoding.encode(v["description"]))
                        elif field == "enum":
                            function_tokens -= 3
                            for o in v["enum"]:
                                function_tokens += 3
                                function_tokens += len(encoding.encode(o))
                        else:
                            logger.warning(f"Not supported field {field}")
                function_tokens += 11
                if len(parameters["properties"]) == 0:
                    function_tokens -= 2

        num_tokens += function_tokens

    num_tokens += 12
    return num_tokens


def print_distribution(values, name):
    if (len(values) > 0):
        logger.info(f"### Distribution of {name}:")
        logger.info(f"min / max: {min(values)}, {max(values)}")
        logger.info(f"mean / median: {np.mean(values)}, {np.median(values)}")
        logger.info(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")


def print_stats_tokens(jsonl_files, model="gpt-4o-2024-05-13"):
    
    for jsonl_path in jsonl_files:
        logger.info('*' * 50)        
        logger.info(f"### [TOKEN_STATS] Processing file: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]

        total_tokens = []
        assistant_tokens = []
        function_tokens = []

        for idx, ex in enumerate(dataset):
            messages = ex.get("messages", {})
            functions = ex.get("tools", {""})
            total_tokens.append(count_token(messages, model))
            assistant_tokens.append(num_assistant_tokens_from_messages(messages, model))
            if len(functions) > 1 and functions != {''}:
                function_tokens.append(num_tokens_from_functions(functions, model))

        print_distribution(total_tokens, "total tokens")
        print_distribution(function_tokens, "function tokens")
        print_distribution(assistant_tokens, "assistant tokens")    
        logger.info('*' * 50)