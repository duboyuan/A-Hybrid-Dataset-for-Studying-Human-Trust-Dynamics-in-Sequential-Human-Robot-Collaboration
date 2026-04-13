#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/1/18 18:16
# @Author  : dby
# @File    : utils.py
# @Software: PyCharm
import random
import numpy as np
import json
import re
import time
import importlib.util
import inspect

import time
from functools import wraps

def set_seed(seed=42):
    random.seed(seed)

def auto_parse_text(text, tags: list, keep_tag=False):
    result_dict = dict()

    for tag in tags:
        pattern = re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL)
        result = pattern.search(text)
        if not result:
            result_text = ""
        elif keep_tag:
            result_text = result.group(0).strip()
        else:
            result_text = result.group(1).strip()

        try:
            _result_text = json.loads(result_text)
        except Exception:
            _result_text = result_text
        result_dict[tag] = _result_text

    return result_dict


def extract_dict_from_string(input_string,logger):
    try:
        return json.loads(input_string)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', input_string, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            logger.error("Can not decode json, string is {}".format(input_string))
            return None
    else:
        logger.error("No valid dictionary content found, string is {}".format(input_string))
        return None


def get_function_names_from_file(file_path):
    spec = importlib.util.spec_from_file_location("module_name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function_names = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            function_names.append(name)

    return function_names

def retry(max_retries=3, delay=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying... ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
            raise Exception(f"Failed after {max_retries} retries.")
        return wrapper
    return decorator



def test_parse():
    raw_text = '''
This is a test text.

<total ingredients> 
{
    "a": "b"
}
</total ingredients> 

<overall preference> 
{
    "c": "d"
}
</overall preference> 

<message> 
Hello Happy World!
From Kokoro Tsurumaki
</message> 

'''
    result = auto_parse_text(raw_text, ["total ingredients", "overall preference", "message"])
    print(result)
    ingre = json.loads(result["total ingredients"])
    pref = json.loads(result["overall preference"])
    msg = result["message"]
    print(ingre, pref, msg)

if __name__ == '__main__':
    test_parse()
