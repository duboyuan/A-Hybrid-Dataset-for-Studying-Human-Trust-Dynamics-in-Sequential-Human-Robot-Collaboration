#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/7 17:28
# @Author  : dby
# @File    : llm_human_action.py
# @Software: PyCharm
from llm.llm_base.agent import Agent
from llm.llm_human_action.human_action_prompt import DECISION_PROMPT
from utils.utils import retry


def _pct_rank_for_prompt(value):
    if value is None:
        return "50"
    s = str(value).strip().replace("%", "")
    return s


class LlmHumanAction(Agent):
    def __init__(self, tool_path=None, agent_type="react", logger=None, response_type="<>", response_format=None,
                 llm_model_name='gpt-4o-mini', model_temperature=0.8, is_record_token=True, max_retries=3,
                 retry_delay=1):
        super(LlmHumanAction, self).__init__("LlmHumanAction", tool_path, agent_type, logger, response_type,
                                             response_format,
                                             llm_model_name, model_temperature, is_record_token, max_retries,
                                             retry_delay)
        self.agent = self.creat_agent()
        self.response_type = response_type
        if self.response_type == "<>":
            self.tags = ["Decision"]
        else:
            self.tags = []
        self.process_end = False
        self.prompt_template = DECISION_PROMPT
        self.tool_path = tool_path

    def get_llm_human_action_agent_response(self, prompt_format):
        @retry(max_retries=self.max_retries, delay=self.retry_delay)
        def get_chat_guide_agent_response_():
            current_human_action_response, current_human_action_response_token_dict = self.get_agent_response(
                prompt_format,
                self.agent)
            raw_text = current_human_action_response["messages"][-1].content
            parsed = self.parse_response(current_human_action_response, self.tags)
            return parsed, current_human_action_response_token_dict, raw_text

        return get_chat_guide_agent_response_()

    def build_decision_prompt_text(
            self,
            trust,
            robot_action,
            extraversion,
            agreeableness,
            conscientiousness,
            neuroticism,
            openness,
            task_id=1,
            room_name="living room",
            room_size_rank=1,
            recent_history="None",
            age=28,
            gender="male",
            education="undergraduate degree",
            industry="a related industry",
    ):
        if robot_action == 0:
            robot_action = "Directly enter"
        else:
            robot_action = "Call for support"
        args = {
            "task_id": task_id,
            "room_name": room_name,
            "room_size_rank": room_size_rank,
            "current_trust": f"{trust} (0-1)",
            "robot_action": robot_action,
            "extraversion": _pct_rank_for_prompt(extraversion),
            "agreeableness": _pct_rank_for_prompt(agreeableness),
            "conscientiousness": _pct_rank_for_prompt(conscientiousness),
            "neuroticism": _pct_rank_for_prompt(neuroticism),
            "openness": _pct_rank_for_prompt(openness),
            "recent_history": recent_history,
            "age": age,
            "gender": gender,
            "education": education,
            "industry": industry,
        }
        return self.prompt_format(self.prompt_template, **args)

    def get_human_action(
            self,
            trust,
            robot_action,
            extraversion,
            agreeableness,
            conscientiousness,
            neuroticism,
            openness,
            task_id=1,
            room_name="living room",
            room_size_rank=1,
            recent_history="None",
            age=28,
            gender="male",
            education="undergraduate degree",
            industry="a related industry",
            prompt_format=None,
    ):
        if prompt_format is None:
            prompt_format = self.build_decision_prompt_text(
                trust, robot_action, extraversion, agreeableness, conscientiousness, neuroticism, openness,
                task_id=task_id, room_name=room_name, room_size_rank=room_size_rank, recent_history=recent_history,
                age=age, gender=gender, education=education, industry=industry,
            )
        human_action, _, raw_full = self.get_llm_human_action_agent_response(prompt_format)
        raw_text = human_action.get("Decision", "")
        decision = str(raw_text).strip()
        decision_lower = decision.lower()
        if "call for support" in decision_lower:
            return 1, raw_full
        if "directly enter" in decision_lower:
            return 0, raw_full
        raise ValueError(f"human action error: unexpected decision {decision!r}")
