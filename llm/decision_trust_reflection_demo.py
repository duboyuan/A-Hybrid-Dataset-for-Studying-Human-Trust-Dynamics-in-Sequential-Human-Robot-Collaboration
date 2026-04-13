#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.llm_human_action.llm_human_action import LlmHumanAction
from llm.llm_reflection.llm_reflection import LlmReflection
from llm.llm_trust.llm_trust import LlmTrust


def main():
    parser = argparse.ArgumentParser(description="decision making → trust update → reflection demo")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=ROOT / "outputs" / "llm_pipeline_run.txt",  
        help="save LLM output txt path",
    )
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("decision_trust_reflection_demo")


    task_id = 1
    room_name = "study room (layout index 2)"
    room_size_rank = 2
    recent_history = "None"
    age = 20
    gender = "male"
    education = "undergraduate degree"
    industry = "a related industry"
    openness = "74%"
    conscientiousness = "79%"
    extraversion = "44%"
    agreeableness = "20%"
    neuroticism = "62%"

    previous_trust = 0.7
    robot_action = 0  # 0 =  Directly enter
    env_state = 0  # 0 = no threat

    human_agent = LlmHumanAction(logger=log, llm_model_name=args.model)
    trust_agent = LlmTrust(logger=log, llm_model_name=args.model)
    reflection_agent = LlmReflection(logger=log, llm_model_name=args.model)

    decision_prompt = human_agent.build_decision_prompt_text(
        previous_trust,
        robot_action,
        extraversion,
        agreeableness,
        conscientiousness,
        neuroticism,
        openness,
        task_id=task_id,
        room_name=room_name,
        room_size_rank=room_size_rank,
        recent_history=recent_history,
        age=age,
        gender=gender,
        education=education,
        industry=industry,
    )

    human_action, raw_human = human_agent.get_human_action(
        previous_trust,
        robot_action,
        extraversion,
        agreeableness,
        conscientiousness,
        neuroticism,
        openness,
        task_id=task_id,
        room_name=room_name,
        room_size_rank=room_size_rank,
        recent_history=recent_history,
        age=age,
        gender=gender,
        education=education,
        industry=industry,
        prompt_format=decision_prompt,
    )

    judgment_human, raw_reflect_human = reflection_agent.reflect(
        decision_prompt,
        raw_human,
        task_id=task_id,
        room_name=room_name,
        room_size_rank=room_size_rank,
        age=age,
        gender=gender,
        education=education,
        industry=industry,
        extraversion=extraversion,
        agreeableness=agreeableness,
        conscientiousness=conscientiousness,
        neuroticism=neuroticism,
        openness=openness,
    )

    task_score = 5 if env_state == human_action else -5
    task_result_txt = "successfully" if task_score > 0 else "unsuccessfully"
    trust_history = f"{previous_trust:.2f}"

    trust_prompt = trust_agent.build_trust_prompt_text(
        robot_action,
        env_state,
        human_action,
        previous_trust,
        extraversion,
        agreeableness,
        conscientiousness,
        neuroticism,
        openness,
        task_id=task_id,
        room_name=room_name,
        room_size_rank=room_size_rank,
        trust_history=trust_history,
        task_result=task_result_txt,
        age=age,
        gender=gender,
        education=education,
        industry=industry,
    )

    new_trust, raw_trust = trust_agent.next_trust(
        robot_action,
        env_state,
        human_action,
        previous_trust,
        extraversion,
        agreeableness,
        conscientiousness,
        neuroticism,
        openness,
        task_id=task_id,
        room_name=room_name,
        room_size_rank=room_size_rank,
        trust_history=trust_history,
        task_result=task_result_txt,
        age=age,
        gender=gender,
        education=education,
        industry=industry,
        prompt_format=trust_prompt,
    )

    judgment_trust, raw_reflect_trust = reflection_agent.reflect(
        trust_prompt,
        raw_trust,
        task_id=task_id,
        room_name=room_name,
        room_size_rank=room_size_rank,
        age=age,
        gender=gender,
        education=education,
        industry=industry,
        extraversion=extraversion,
        agreeableness=agreeableness,
        conscientiousness=conscientiousness,
        neuroticism=neuroticism,
        openness=openness,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"=== run at {datetime.now().isoformat()} ===",
        f"model: {args.model}",
        "",
        "=== 1) Human action — input prompt ===",
        decision_prompt.rstrip(),
        "",
        "=== 1) Human action — LLM raw output ===",
        raw_human.rstrip(),
        "",
        f"parsed human_action (0=direct enter, 1=call support): {human_action}",
        "",
        "=== 1b) Reflection on human action — LLM raw output ===",
        raw_reflect_human.rstrip(),
        f"judgment (True=appropriate): {judgment_human}",
        "",
        "=== 2) Trust update — input prompt ===",
        trust_prompt.rstrip(),
        "",
        "=== 2) Trust update — LLM raw output ===",
        raw_trust.rstrip(),
        f"parsed new_trust: {new_trust} (previous {previous_trust})",
        "",
        "=== 2b) Reflection on trust update — LLM raw output ===",
        raw_reflect_trust.rstrip(),
        f"judgment (True=appropriate): {judgment_trust}",
        "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    log.info("Writed %s", args.output)


if __name__ == "__main__":
    main()
