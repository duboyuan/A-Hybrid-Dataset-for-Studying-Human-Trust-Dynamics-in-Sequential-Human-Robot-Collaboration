#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/7 18:16
# @Author  : dby
# @File    : llm_reflection.py
# @Software: PyCharm
from llm.llm_base.agent import Agent
from llm.llm_reflection.reflection_prompt import REFLECTION_PROMPT
from utils.utils import retry


def _pct_rank_for_prompt(value):
    if value is None:
        return "50"
    return str(value).strip().replace("%", "")


class LlmReflection(Agent):
    def __init__(self, tool_path=None, agent_type="react", logger=None, response_type="<>", response_format=None,
                 llm_model_name='gpt-4o-mini', model_temperature=0.3, is_record_token=True, max_retries=3,
                 retry_delay=1):
        super(LlmReflection, self).__init__("LlmReflection", tool_path, agent_type, logger, response_type,
                                              response_format,
                                              llm_model_name, model_temperature, is_record_token, max_retries,
                                              retry_delay)
        self.agent = self.creat_agent()
        self.response_type = response_type
        if self.response_type == "<>":
            self.tags = ["Judgment"]
        else:
            self.tags = []
        self.process_end = False
        self.prompt_template = REFLECTION_PROMPT
        self.tool_path = tool_path

    def get_llm_reflection_agent_response(self, prompt_format):
        @retry(max_retries=self.max_retries, delay=self.retry_delay)
        def _call():
            resp, token_dict = self.get_agent_response(prompt_format, self.agent)
            raw_text = resp["messages"][-1].content
            parsed = self.parse_response(resp, self.tags)
            return parsed, token_dict, raw_text

        return _call()

    def build_reflection_prompt_text(
            self,
            source_prompt,
            llm_response,
            task_id=1,
            room_name="living room",
            room_size_rank=1,
            age=28,
            gender="male",
            education="undergraduate degree",
            industry="a related industry",
            extraversion="50",
            agreeableness="50",
            conscientiousness="50",
            neuroticism="50",
            openness="50",
    ):
        args = {
            "task_id": task_id,
            "room_name": room_name,
            "room_size_rank": room_size_rank,
            "source_prompt": source_prompt,
            "llm_response": llm_response,
            "age": age,
            "gender": gender,
            "education": education,
            "industry": industry,
            "openness": _pct_rank_for_prompt(openness),
            "conscientiousness": _pct_rank_for_prompt(conscientiousness),
            "extraversion": _pct_rank_for_prompt(extraversion),
            "agreeableness": _pct_rank_for_prompt(agreeableness),
            "neuroticism": _pct_rank_for_prompt(neuroticism),
        }
        return self.prompt_format(self.prompt_template, **args)

    def reflect(
            self,
            source_prompt,
            llm_response,
            task_id=1,
            room_name="living room",
            room_size_rank=1,
            age=28,
            gender="male",
            education="undergraduate degree",
            industry="a related industry",
            extraversion="50",
            agreeableness="50",
            conscientiousness="50",
            neuroticism="50",
            openness="50",
    ):
        prompt_format = self.build_reflection_prompt_text(
            source_prompt, llm_response,
            task_id=task_id, room_name=room_name, room_size_rank=room_size_rank,
            age=age, gender=gender, education=education, industry=industry,
            extraversion=extraversion, agreeableness=agreeableness, conscientiousness=conscientiousness,
            neuroticism=neuroticism, openness=openness,
        )
        parsed, _, raw_full = self.get_llm_reflection_agent_response(prompt_format)
        raw = str(parsed.get("Judgment", "")).strip()
        lower = raw.lower()
        if lower == "true":
            return True, raw_full
        if lower == "false":
            return False, raw_full
        raise ValueError(f"reflection: unexpected Judgment: {raw!r}")
