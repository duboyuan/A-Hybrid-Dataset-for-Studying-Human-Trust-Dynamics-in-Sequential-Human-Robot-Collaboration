#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/3/3 21:10
# @Author  : dby
# @File    : agent.py
# @Software: PyCharm
import json
import logging

from langgraph.prebuilt import create_react_agent

from llm.llm_base.llm import model
from utils.utils import get_function_names_from_file, retry, auto_parse_text

from langchain_core.prompts import PromptTemplate


class Agent:
    def __init__(self, agent_name, tool_path, agent_type, logger, response_type="json",response_format=None, llm_model_name='gpt-4o-mini',
                 model_temperature=0, is_record_token=True, max_retries=3, retry_delay=1):
        self.llm_model_name = llm_model_name
        self.model_temperature = model_temperature
        self.tool_path = None
        self.agent_type = agent_type
        if logger is None:
            self.logger = logging.getLogger(f"{agent_name}.agent")
            self.logger.addHandler(logging.NullHandler())
        else:
            self.logger = logger
        self.response_type = response_type
        self.is_record_token = is_record_token
        self.agent_name = agent_name
        self.response_format = response_format
        if self.response_format is not None:
            self.use_response_format = True
        else:
            self.use_response_format = False
        self.model = model(self.llm_model_name, self.model_temperature)
        if self.tool_path is not None:
            self.tools = get_function_names_from_file(self.tool_path)
        else:
            self.tools = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def creat_agent(self):
        if self.agent_type == "react":
            if self.use_response_format:
                agent = create_react_agent(self.model, tools=self.tools, response_format=self.response_format)
            else:
                agent = create_react_agent(self.model, tools=self.tools)
        elif self.agent_type == "chat":
            if self.use_response_format:
                agent = self.model.with_structured_output(self.response_format)
            else:
                agent = self.model
        return agent

    def get_agent_response(self, prompt_format, agent):
        @retry(max_retries=self.max_retries, delay=self.retry_delay)
        def get_agent_response_():
            response = agent.invoke({"messages": prompt_format})
            input_tokens = response["messages"][-1].usage_metadata["input_tokens"]
            output_tokens = response["messages"][-1].usage_metadata["output_tokens"]
            total_tokens = response["messages"][-1].usage_metadata["total_tokens"]
            input_tokens_details = response["messages"][-1].usage_metadata["input_token_details"]
            output_tokens_details = response["messages"][-1].usage_metadata["output_token_details"]
            token_dict = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "input_tokens_details": input_tokens_details,
                "output_tokens_details": output_tokens_details,
            }
            self.single_step_log(token_dict, response, prompt_format)
            return response, token_dict
        return get_agent_response_()

    def single_step_log(self, token_dict, response, prompt_format):
        self.logger.info('{} is called'.format(self.agent_name))
        self.logger.info('{} input is {}'.format(self.agent_name, prompt_format))
        self.logger.info('{} output is {}'.format(self.agent_name, response["messages"][-1].content))
        self.logger.debug("{} input_tokens: {}".format(self.agent_name, token_dict["input_tokens"]))
        self.logger.debug("{} output_tokens: {}".format(self.agent_name, token_dict["output_tokens"]))
        self.logger.debug("{} input_tokens_details: {}".format(self.agent_name, token_dict["input_tokens_details"]))
        self.logger.debug("{} output_tokens_details: {}".format(self.agent_name, token_dict["output_tokens_details"]))

    def parse_response(self, response,tags):
        if self.response_type == "<>":
            parsed_response = auto_parse_text(response["messages"][-1].content, tags)
        elif self.response_type == "json":
            parsed_response = json.loads(response["messages"][-1].content.strip().replace("\n", "").replace("None", "\"none\""))
        # todo 如果需要再支持其他算法
        return parsed_response

    def prompt_format(self,prompt_template, **kwargs):
        template_variables = PromptTemplate(input_variables=[], template=prompt_template).input_variables
        valid_kwargs = {key: kwargs[key] for key in template_variables if key in kwargs}
        formatted_prompt = PromptTemplate(input_variables=template_variables, template=prompt_template).format_prompt(
            **valid_kwargs)
        return formatted_prompt.text

