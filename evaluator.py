import json
import os
import logging
from typing import List, Dict, Any
import argparse, os, sys, logging, numpy as np
from datetime import datetime
from typing import List
from sentence_transformers import SentenceTransformer

import pandas as pd
from datetime import datetime
from pathlib import Path

import os
from pathlib import Path

import argparse, os, sys, logging, numpy as np
from datetime import datetime
from typing import List


import pandas as pd
from datetime import datetime
from pathlib import Path
import json

from sharedmodel import global_model

logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

if not logging.getLogger().handlers:
    log_filename = logs_dir / f"execution_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger()

def log_print(*args, **kwargs):
    
    message = '[JUDGE] ' + ' '.join(str(arg) for arg in args)
    logger.info(message)

class LLMJudge:
    def __init__(self, model_name="",  base_url=None):
      
        
        self.model = global_model  # Use the shared global model
        self.few_shot_examples = self._get_few_shot_examples()
    
    def _get_few_shot_examples(self) -> str:
       
        return """
    --------------------------------------------------------------------
    Example 1
    --------------------------------------------------------------------
    Plan:
    1) Find the protagonist in the movie 'Inception'
    2) Find the protagonist's birthplace
    3) Find the population of that city
    
    Chunks:
    - "Inception is a 2010 science fiction film starring Leonardo DiCaprio as Dom Cobb, a professional thief who infiltrates people's dreams..."
    - "Leonardo DiCaprio was born in Los Angeles, California, United States on November 11, 1974..."
    - "Los Angeles is the most populous city in California with a population of approximately 4 million people as of 2023..."
    
    For each question in the plan, judge if the information needed is clearly and explicitly present in the chunks. The chunk set should collectively cover all different types of information required by the steps.
    
    1. Score: 5. Short Explanation: Leonardo DiCaprio is explicitly named as the protagonist.
    2. Score: 5. Short Explanation: Born in Los Angeles, California is directly stated.
    3. Score: 5. Short Explanation: Population of 4 million is explicitly mentioned.
    Total Score: 15
    
    --------------------------------------------------------------------
    Example 2
    --------------------------------------------------------------------
    
    Plan:
    1) Find the director of the movie 'Titanic'
    2) Find the director's net worth
    3) Find the director's upcoming projects
    
    Chunks:
    - "Titanic is a 1997 epic romance and disaster film. The movie was a massive box office success..."
    - "The film won 11 Academy Awards including Best Picture and Best Director..."
    - "Leonardo DiCaprio and Kate Winslet starred as the main characters in this epic love story..."
    
    For each question in the plan, judge if the information needed is clearly and explicitly present in the chunks. The chunk set should collectively cover all different types of information required by the steps.
    
    1. Score: 0. Short Explanation: Director name not mentioned, only that it won Best Director
    2. Score: 0. Short Explanation: No financial information about anyone provided
    3. Score: 0. Short Explanation: No information about future projects mentioned
    Total Score: 0
    
    --------------------------------------------------------------------
    Example 3
    --------------------------------------------------------------------
    
    Plan:
    1) Find the author of 'Harry Potter'
    2) Find the author's age
    3) Find the number of books in the series
    
    Chunks:
    - "Harry potter, the book written by the sister of James Franco is fantasy book series that became globally popular..."
    -" James Franco and along with all his siblings travelled to Europe..."
    -"In Europe, Franco and JK Rowling played Hokkey like all the siblings"
    - "The series consists of seven main novels plus several companion books..."
    - "Rowling has become one of the most successful authors in modern publishing..."
    
    For each question in the plan, judge if the information needed is clearly and explicitly present in the chunks. The chunk set should collectively cover all different types of information required by the steps.
    
    1. Score: 3. Short Explanation: J.K. Rowling is inferred named as the author from the relationships.
    2. Score: 0. Short Explanation: Age not mentioned anywhere in the chunks.
    3. Score: 5. Short Explanation: Seven main novels is directly stated.
    Total Score: 8"""

    def evaluate_chunks(self, plan: List[str], chunks: List[str], lambda_value: float) -> Dict[str, Any]:
        """
        Evaluate if the provided chunks can satisfy the given plan.
        
        Args:
            plan: List of questions/steps in the plan
            chunks: List of text chunks retrieved at the given lambda
            lambda_value: The lambda value used for retrieval
            
        Returns:
            Dictionary with evaluation results
        """

        
        
        plan_text = "\n".join([f"{i+1}) {step}" for i, step in enumerate(plan)])
        chunks_text = "\n".join([f"- {chunk}" for chunk in chunks])

        user_content = f"""
Score each question 0-5 based on how directly it can be answered:
5: EXPLICIT answer with specific facts/numbers stated directly
4: Clear and complete answer with direct relevant information present 
3: Mostly complete answer, minor missing detail, but can be inferred
2: Partial information, key details missing
1: Very limited relevance, mostly vague
0: No relevant information or requires inference

STRICT RULES:
- Only count information DIRECTLY STATED in chunks
- Do NOT give points for logical leaps or inferences
- Do NOT give points for vague or tangential content

Your reasonings for the score values you provide should be within **20 tokens** (compact). **You MUST remember this.**
At the end you must answer in the format:
Total Score: <Total_Score>

Follow the examples for understanding the task: 

{self.few_shot_examples}

---
Now do the same for the following:

Plan:
{plan_text}

Chunks:
{chunks_text}

For each question in the plan, judge if the information needed is clearly and explicitly present in the chunks. The chunk set should collectively cover all different types of information required by the steps.\n

"""

        
        system_content = "You are a strict judge evaluating text chunks. Only count information that is explicitly stated."
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content":  user_content}
        ]
        
        response_text = self.model.generate_response(
            messages=messages,
            temperature=0.001,
            max_tokens=2000
        )
        
        log_print("This is a response Text")
        log_print(response_text)
        
        log_print('X'*50)
        
        #log_print(111111111)
        
        # Parse the response
        assessment = self._parse_response(response_text)

        questions_answered = 0
        
        try:
            ans = response_text.strip().split('Total Score:')[1].split('\n')[0]
            log_print(f"Questions Answered = {ans}")
            questions_answered = int(ans)
            log_print(f"Questions: Answered: {questions_answered}")
        except (IndexError, ValueError) as e:
            log_print(f"An error occurred: {e}")
            log_print("Skipped this lambda")
            questions_answered = 0
        
        return {
            "lambda": lambda_value,
            "response_text": response_text,
            "can_satisfy_plan": assessment["decision"],
            "explanation": assessment["explanation"],
            "chunks_count": len(chunks),
            "plan_steps": len(plan),
            "questions_answered": questions_answered
        }
        
    def count_answerable_questions(self, plan:List[str], chunks: List[str]) -> int:
        chunks_text = '\n'.join([f"- {chunk}" for chunk in chunks])
        answerable_count = 0
        for i,question in enumerate(plan):
            user_content = f"""Question: {question}
Chunks: 
{chunks_text}

Can this specific question be answered using the chunks? Respond only with PASS or FAIL"""

            messages = [
                {"role": "user", "content": user_content}
            ]
            
            response_text = self.model.generate_response(
                messages=messages,
                temperature=0.001,
                max_tokens=500
            )
            
            if 'PASS' in response_text:
                answerable_count+=1
            log_print(f"Answerable Counts: {answerable_count}")
        return answerable_count
            
        
    def _parse_response(self, response: str) -> Dict[str, Any]:
        
        response = response.strip()
        
        
        if "Final Answer: PASS" in response:
            decision = True
            explanation = "Plan can be satisfied"
        elif "Final Answer: FAIL" in response:
            decision = False
            explanation = "Plan cannot be satisfied"
        else:
            decision = False
            explanation = "Could not parse response"
        
        return {
            "decision": decision,
            "explanation": explanation
        }
    
    def find_best_lambda(self, plan: List[str], chunks_by_lambda: Dict[float, List[str]]) -> Dict[str, Any]:
        
        results = []
        
        for lambda_val, chunks in chunks_by_lambda.items():
            result = self.evaluate_chunks(plan, chunks, lambda_val)
            results.append(result)
        
       

        max_score = max(results, key = lambda x: x['questions_answered'])['questions_answered']
        best_results = [result for  result in results if result["questions_answered"] == max_score]
        tied_lambdas = sorted([result['lambda'] for result in best_results])
        
       
        
        if len(best_results) > 1:
            tied_lambdas = sorted([result['lambda'] for result in best_results])
            median_index =  len(tied_lambdas) // 2
            chosen_lambda = tied_lambdas[median_index]
            best_lambda = next(result for result in best_results if result['lambda'] == chosen_lambda)
        else:
            best_lambda = best_results[-1]
        
        return {
            "plausible_lambdas": tied_lambdas,
            "best_lambda": float(best_lambda["lambda"]),
            "response_text": best_lambda["response_text"],
            "can_satisfy_plan": best_lambda["can_satisfy_plan"],
            "explanation": best_lambda["explanation"],
            "all_evaluations": results
        }