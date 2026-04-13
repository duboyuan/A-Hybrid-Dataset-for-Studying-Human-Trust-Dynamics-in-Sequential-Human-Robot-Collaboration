#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/26 20:54
# @Author  : dby
# @File    : llm_trust.py
# @Software: PyCharm
from llm.llm_base.agent import Agent
from llm.llm_trust.trust_prompt import TRUST_PROMPT_
from utils.utils import retry


def _pct_rank_for_prompt(value):
    if value is None:
        return "50"
    return str(value).strip().replace("%", "")


def _robot_advice_phrase(robot_action):
    if robot_action == 0:
        return "direct entry"
    return "call for support"


def _human_action_phrase(human_action):
    if human_action == 0:
        return "directly enter"
    if human_action == 1:
        return "call for support"
    if isinstance(human_action, str):
        s = human_action.strip().lower()
        if "call" in s or "support" in s:
            return "call for support"
        return "directly enter"
    raise ValueError(f"invalid human_action: {human_action!r}")


def _environment_phrase(state):
    if state == 0:
        return "no threat"
    if state == 1:
        return "threat present"
    if isinstance(state, str):
        return state.strip()
    raise ValueError(f"invalid environment state: {state!r}")


class LlmTrust(Agent):
    def __init__(self, tool_path=None, agent_type="react", logger=None, response_type="<>", response_format=None,
                 llm_model_name='gpt-4o-mini', model_temperature=0.8, is_record_token=True, max_retries=3,
                 retry_delay=1):
        super(LlmTrust, self).__init__(
            "LlmTrust", tool_path, agent_type, logger, response_type, response_format,
            llm_model_name, model_temperature, is_record_token, max_retries, retry_delay,
        )
        self.agent = self.creat_agent()
        self.response_type = response_type
        if self.response_type == "<>":
            self.tags = ["trust"]
        else:
            self.tags = []
        self.process_end = False
        self.prompt_template = TRUST_PROMPT_
        self.tool_path = tool_path

    def get_llm_trust_agent_response(self, prompt_format):
        @retry(max_retries=self.max_retries, delay=self.retry_delay)
        def get_chat_guide_agent_response_():
            current_trust_response, current_trust_response_token_dict = self.get_agent_response(prompt_format,
                                                                                              self.agent)
            raw_text = current_trust_response["messages"][-1].content
            parsed = self.parse_response(current_trust_response, self.tags)
            return parsed, current_trust_response_token_dict, raw_text

        return get_chat_guide_agent_response_()

    def build_trust_prompt_text(
            self,
            robot_action,
            state,
            human_action,
            previous_trust,
            extraversion,
            agreeableness,
            conscientiousness,
            neuroticism,
            openness,
            task_id=1,
            room_name="living room",
            room_size_rank=1,
            trust_history="None",
            task_result="successfully",
            robot_detection_result=None,
            environment_state=None,
            age=28,
            gender="male",
            education="undergraduate degree",
            industry="a related industry",
    ):
        if robot_detection_result is None:
            robot_detection_result = _robot_advice_phrase(robot_action)
        if environment_state is None:
            environment_state = _environment_phrase(state)
        human_act = _human_action_phrase(human_action)
        args = {
            "task_id": task_id,
            "room_name": room_name,
            "room_size_rank": room_size_rank,
            "human_action": human_act,
            "robot_detection_result": robot_detection_result,
            "environment_state": environment_state,
            "task_result": task_result,
            "current_trust": previous_trust,
            "trust_history": trust_history,
            "extraversion": _pct_rank_for_prompt(extraversion),
            "agreeableness": _pct_rank_for_prompt(agreeableness),
            "conscientiousness": _pct_rank_for_prompt(conscientiousness),
            "neuroticism": _pct_rank_for_prompt(neuroticism),
            "openness": _pct_rank_for_prompt(openness),
            "age": age,
            "gender": gender,
            "education": education,
            "industry": industry,
        }
        return self.prompt_format(self.prompt_template, **args)

    def next_trust(
            self,
            robot_action,
            state,
            human_action,
            previous_trust,
            extraversion,
            agreeableness,
            conscientiousness,
            neuroticism,
            openness,
            task_id=1,
            room_name="living room",
            room_size_rank=1,
            trust_history="None",
            task_result="successfully",
            robot_detection_result=None,
            environment_state=None,
            age=28,
            gender="male",
            education="undergraduate degree",
            industry="a related industry",
            prompt_format=None,
    ):
        if prompt_format is None:
            prompt_format = self.build_trust_prompt_text(
                robot_action, state, human_action, previous_trust,
                extraversion, agreeableness, conscientiousness, neuroticism, openness,
                task_id=task_id, room_name=room_name, room_size_rank=room_size_rank,
                trust_history=trust_history, task_result=task_result,
                robot_detection_result=robot_detection_result, environment_state=environment_state,
                age=age, gender=gender, education=education, industry=industry,
            )
        parsed, _, raw_full = self.get_llm_trust_agent_response(prompt_format)
        raw_trust = parsed.get("trust", "")
        try:
            current_trust = float(str(raw_trust).strip())
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid trust value from model: {raw_trust!r}") from e
        if current_trust < 0 or current_trust > 1:
            raise ValueError("Invalid current trust: out of [0,1]")

        return round(current_trust, 2), raw_full
