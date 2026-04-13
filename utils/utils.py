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
    # 设置 Python 内置 random 库的种子
    random.seed(seed)

    # 设置 numpy 的随机种子
    np.random.seed(seed)

    # 设置 PyTorch 的随机种子
    # torch.manual_seed(seed)
    #
    # # 如果你在使用 GPU，也需要设置 CUDA 随机种子
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    #
    # # 如果使用了 cudnn 加速，设置它以确保结果的一致性
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

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
            # 必须用捕获组，不能用 strip("<Tag>")：strip 把参数当「字符集合」，
            # 会从内容里误删如 Decision 中的 D、e、i 等，例如 "Directly" -> "rectly"。
            result_text = result.group(1).strip()

        try:
            _result_text = json.loads(result_text)
        except Exception:
            _result_text = result_text
        result_dict[tag] = _result_text

    return result_dict


def extract_dict_from_string(input_string,logger):
    # input_string = fix_json_string(input_string)
    # 如果输入已经是有效的 JSON 字符串，直接解析并返回
    try:
        return json.loads(input_string)
    except json.JSONDecodeError:
        pass
    # 使用正则表达式匹配花括号中的内容
    match = re.search(r'\{.*\}', input_string, re.DOTALL)
    if match:
        # 提取出内容并转换为字典
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            logger.error("Can not decode json, string is {}".format(input_string))
            return None
    else:
        logger.error("No valid dictionary content found, string is {}".format(input_string))
        return None


def get_function_names_from_file(file_path):
    # 加载模块
    spec = importlib.util.spec_from_file_location("module_name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取所有函数名
    function_names = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            function_names.append(name)

    return function_names

def retry(max_retries=3, delay=3):
    """
    重试装饰器，用于在函数失败时自动重试。
    :param max_retries: 最大重试次数
    :param delay: 每次重试的间隔时间（秒）
    """
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
    # 期望的结果：
    # {
    #     "total ingredients": {"a": "b"},
    #     "overall preference": {"c": "d"},
    #     "message": "Hello Happy World!\nFrom Kokoro Tsurumaki"
    # }
    # 注意：message 部分包含了换行符，需要处理掉。


if __name__ == '__main__':
    test_parse()
