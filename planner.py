# planner.py
import re
import textwrap
from collections import Counter
from typing import List

# Import the shared model
from sharedmodel import global_model


PLAN_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert question decomposer planner.
    Task: Break the user's **single question** into the minimal
    ordered list of sub-questions that must all be answered
    to fully answer the original question harnessing your zero-shot capabilites.
    Follow the output format exactly as shown in the example below to generate a plan (a list of sub-questions).

    ---
    Question: Where was the author of the movie 'Jacob 'Born?

    Output Planner :
    1) Identify Author of jacob
    2) Identify where was Author Born

    ---
    Only produce the PLAN section—no extra commentary.
"""
).strip()


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.01, max_tokens: int = 800) -> str:
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return global_model.generate_response(messages=messages, temperature=temperature, max_tokens=max_tokens).strip()


def decompose_question(question: str) -> str:
   
    user_prompt = f"QUESTION:\n{question}"
    return call_llm(PLAN_SYSTEM_PROMPT, user_prompt)


def parse_plan_lines(plan_text: str) -> List[str]:

    return [ln.strip() for ln in plan_text.split("\n") if ln.strip()]


def get_plan(question: str) -> List[str]:
    plan_text = decompose_question(question)
    return parse_plan_lines(plan_text)

