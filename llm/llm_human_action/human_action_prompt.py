#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/7 17:28
# @Author  : dby
# @File    : human_action_prompt.py
# @Software: PyCharm

DECISION_PROMPT = '''
Scenario Description:
I am conducting a human-robot collaboration experiment. In each collaborative task, the participant must choose one of two possible actions:
1. directly enter the area
2. call for support before entering

Once the participant makes an action choice, the task outcome depends on both the actual environment state and the chosen action. The environment state has two possible values: "threat" or "no threat". Participants are informed before the experiment that different task outcomes may occur depending on the relationship between the actual environment state and their decision.

Task Description:
The human-robot collaboration is currently on the {task_id}-th task, corresponding to the {room_name}. The room size ranks {room_size_rank} in the house.

Robot's Advice Description:
After reconnaissance, the robot dog gives the following advice for this task:
{robot_action}

Current Trust Description:
The current trust towards the robot dog is {current_trust}.

Participant Description:
The participant to role-play is described as follows:
{age} years old, {gender}, {education}, and working in {industry}.
In the Mini-IPIP test, the scores for openness, conscientiousness, extraversion, agreeableness, and neuroticism are at the {openness}th, {conscientiousness}th, {extraversion}th, {agreeableness}th, and {neuroticism}th percentiles, respectively.

Recent History Description:
The recent interaction history is provided below for reference. To avoid over-reliance on long-term patterns, only the last two turns are shown:
{recent_history}

COT:
Please reason step by step as follows before making your decision:
1. Consider the current trust value before this task [70].
2. Note the robot’s advice for this task (directly enter).
3. Taking into account the characteristics of the current participant, set the trust threshold at [60]
4. Since the current trust value ([70]) exceeds the threshold ([60]), decide to follow the robot's advice [directly enter].

Response Format Description:
Please provide your decision, primarily considering the current trust. Output exactly one line in this form (no other text):
<Decision>Directly enter</Decision>
or
<Decision>Call for support</Decision>

'''