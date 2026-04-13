#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/27 14:40
# @Author  : dby
# @File    : trust_prompt.py
# @Software: PyCharm

TRUST_PROMPT_ = '''
Scenario Description:
I am conducting a human-robot collaboration experiment. In each collaborative task, the participant must choose one of two possible actions:
1. directly enter the area
2. call for support before entering

Once the participant makes an action choice, the task outcome depends on both the actual environment state and the chosen action. The environment state has two possible values: "threat" or "no threat". Participants are informed before the experiment that different task outcomes may occur depending on the relationship between the actual environment state and their decision.

Task Description:
The human-robot collaboration is currently on the {task_id}-th task, corresponding to the {room_name}. The room size ranks {room_size_rank} in the house.

Task Result Description:
After performing [{human_action}] and entering the room, the actual conditions are as follows:
1. Robot dog detection result: [{robot_detection_result}].
2. Actual environment status: [{environment_state}].
3. Participant [{task_result}] completed the task.

Additionally, the participant's prior knowledge includes:
1. When Actual environment status is threat present and Robot dog's advice is call for support, the advice is right.
2. When Actual environment status is no threat and Robot dog's advice is direct entry, the advice is right.

Please update the trust based on the real situation and prior knowledge. The current trust is [{current_trust}].

Context:
History: {trust_history}

Participant Description:
The participant to role-play is described as follows:
{age} years old, {gender}, {education}, and working in {industry}.
In the Mini-IPIP test, the scores for openness, conscientiousness, extraversion, agreeableness, and neuroticism are at the {openness}th, {conscientiousness}th, {extraversion}th, {agreeableness}th, and {neuroticism}th percentiles, respectively.

COT:
Please reason step by step as follows before making your decision:
1. The trust history is [0.6].
2. The robot dog’s advice for this task is [direct entry].
3. After the human made the decision and entered the area, the robot’s advice was compared with the actual environment status.
4. Considering the participant’s characteristics and the fact that the robot’s advice truly helped achieve task success, the trust should be increased
5. Therefore, the updated trust value is set to 0.85.

Response Format Description:
Trust should be within the range of 0-1. The trust update should be based primarily on the following factors:
1. Whether the robot dog’s advice truly helps the human make a correct decision (even if the advice is not chosen by the participant).
2. The current trust.

After your reasoning, output your final numeric trust in this exact form (and this tag line must appear exactly once):
<trust>0.85</trust>

'''
