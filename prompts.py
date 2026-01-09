from typing import List

def format_prompt(query: str, context_chunks: List[str]) -> str:
    
    context = "\n\n".join(f"{c}" for c in context_chunks) or "(no context)"
    
    prompt  = f"""
        You are a question answering assistant.
        Answer the question based on
        the given passages. Only give
        me the answer and do not output
        any other words. The following
        are given passages. {context}
        \n
        Answer the question based on
        the given passages. Only give
        me the answer and do not output
        any other words. Question: {query}
        \n
        Answer:


    """
    

    return prompt
    
