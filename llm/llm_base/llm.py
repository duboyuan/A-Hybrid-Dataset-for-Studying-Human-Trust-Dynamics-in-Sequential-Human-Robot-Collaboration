#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ACN 
@File    ：llm.py
@IDE     ：PyCharm 
@Author  ：dby
@Date    ：2025/2/8 16:54 
'''
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from utils.constants import (OPENAI_API_KEY, DEEPSEEK_API_KEY, BAICAI_API_KEY_wcx, SILICONFLOW_API_KEY,
                             SILICONFLOW_API_KEY_WCX, SILICONFLOW_API_KEY_LSW, OPENAI_API_KEY_LSW_HK)

def model(LLM_model_name,temperature):
    if LLM_model_name == "gpt-4o-mini":
        return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY,temperature=temperature)
    if LLM_model_name == "gpt-4o":
        return ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY,temperature=temperature)
    if LLM_model_name == "gpt-4o-mini-hk":
        return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY_LSW_HK,temperature=temperature, base_url="https://api.openai-hk.com/v1/")
    if LLM_model_name == "gpt-4o-hk":
        return ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY_LSW_HK,temperature=temperature, base_url="https://api.openai-hk.com/v1/")
    if LLM_model_name == "deepseek-v3":
        return ChatOpenAI(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com",temperature=temperature)
    if LLM_model_name == "deepseek-r1":
        return ChatOpenAI(model="deepseek-reasoner", api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com",temperature=temperature)
    if LLM_model_name == "baicai-gpt-4o-mini":
        return ChatOpenAI(model="gpt-4o-mini", api_key=BAICAI_API_KEY_wcx, base_url="https://api.baicaigpt.cn",temperature=temperature)
    if LLM_model_name == "baicai-gpt-4o":
        return ChatOpenAI(model="gpt-4o", api_key=BAICAI_API_KEY_wcx, base_url="https://api.baicaigpt.cn",temperature=temperature)
    if LLM_model_name == "sf-DeepSeek-R1-Distill-Llama-8B":
        return ChatOpenAI(model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", api_key=SILICONFLOW_API_KEY_WCX, base_url="https://api.siliconflow.cn/v1",temperature=temperature)
    if LLM_model_name == "sf-DeepSeek-R1-Distill-Qwen-7B":
        return ChatOpenAI(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", api_key=SILICONFLOW_API_KEY_WCX, base_url="https://api.siliconflow.cn/v1",temperature=temperature)
    if LLM_model_name == "sf-Qwen2.5-7B-Instruct":
        return ChatOpenAI(model="Qwen/Qwen2.5-7B-Instruct", api_key=SILICONFLOW_API_KEY_WCX, base_url="https://api.siliconflow.cn/v1",temperature=temperature)
    if LLM_model_name == "sf-Meta-Llama-3.1-8B-Instruct":
        return ChatOpenAI(model="meta-llama/Meta-Llama-3.1-8B-Instruct", api_key=SILICONFLOW_API_KEY_WCX, base_url="https://api.siliconflow.cn/v1",temperature=temperature)
    if LLM_model_name == "sf-DeepSeek-V3":
        return ChatOpenAI(model="deepseek-ai/DeepSeek-V3", api_key=SILICONFLOW_API_KEY_WCX, base_url="https://api.siliconflow.cn/v1", temperature=temperature)
    if LLM_model_name == "sf-DeepSeek-V3-alter1":
        return ChatOpenAI(model="Pro/deepseek-ai/DeepSeek-V3", api_key=SILICONFLOW_API_KEY_WCX, base_url="https://api.siliconflow.cn/v1", temperature=temperature)
    if LLM_model_name == "sf-DeepSeek-V3-alter2":
        return ChatOpenAI(model="deepseek-ai/DeepSeek-V3", api_key=SILICONFLOW_API_KEY, base_url="https://api.siliconflow.cn/v1", temperature=temperature)
    if LLM_model_name == "sf-DeepSeek-V3-alter3":
        return ChatOpenAI(model="Pro/deepseek-ai/DeepSeek-V3", api_key=SILICONFLOW_API_KEY, base_url="https://api.siliconflow.cn/v1", temperature=temperature)
    if LLM_model_name == "sf-Llama-3.3-70B":
        return ChatOpenAI(model="meta-llama/Llama-3.3-70B-Instruct", api_key=SILICONFLOW_API_KEY_WCX, base_url="https://api.siliconflow.cn/v1", temperature=temperature)
    if LLM_model_name == "sf-Llama-3.3-70B-alter1":
        return ChatOpenAI(model="meta-llama/Llama-3.3-70B-Instruct", api_key=SILICONFLOW_API_KEY,
                          base_url="https://api.siliconflow.cn/v1", temperature=temperature)
    if LLM_model_name == "sf-Llama-3.3-70B-alter2":
        return ChatOpenAI(model="meta-llama/Llama-3.3-70B-Instruct", api_key=SILICONFLOW_API_KEY_LSW,
                          base_url="https://api.siliconflow.cn/v1", temperature=temperature)
    if LLM_model_name == "sf-Qwen2.5-72B":
        return ChatOpenAI(model="Qwen/Qwen2.5-72B-Instruct", api_key=SILICONFLOW_API_KEY_WCX, base_url="https://api.siliconflow.cn/v1", temperature=temperature)
    if LLM_model_name == "sf-Qwen2.5-72B-alter1":
        return ChatOpenAI(model="Qwen/Qwen2.5-72B-Instruct", api_key=SILICONFLOW_API_KEY, base_url="https://api.siliconflow.cn/v1", temperature=temperature)
    if LLM_model_name == "sf-Qwen2.5-72B-alter2":
        return ChatOpenAI(model="Qwen/Qwen2.5-72B-Instruct", api_key=SILICONFLOW_API_KEY_LSW, base_url="https://api.siliconflow.cn/v1", temperature=temperature)

if __name__ == '__main__':
    # model = model("gpt-4o-mini",0.7)
    # model = model("sf-DeepSeek-V3",0.7)
    model = model("sf-Qwen2.5-7B-Instruct",0.7)
    # model = model("sf-Llama-3.3-70B-alter2",0.7)
    messages = [
        HumanMessage(content="who are you?")
    ]
    response = model.invoke(messages)
    a=1