import hashlib

TEXT_TO_SQL_PROMPT_TEMPLATE = """You are a highly capable Text-to-SQL assistant. Your task is to generate a SQL query to answer the user's question based on the provided database schema. Return ONLY the valid SQL query without any markdown formatting or explanations.

{few_shot_examples}Schema:
{schema}

Question: {question}
SQL: """

def hash_prompt_template(template: str) -> str:
    return hashlib.sha256(template.encode('utf-8')).hexdigest()

def render_prompt(schema_str: str, question: str, few_shot_examples: list = None) -> str:
    few_shot_str = ""
    if few_shot_examples:
        for ex in few_shot_examples:
            few_shot_str += f"Schema:\n{ex['schema']}\n\nQuestion: {ex['question']}\nSQL: {ex['sql']}\n\n---\n\n"
            
    return TEXT_TO_SQL_PROMPT_TEMPLATE.format(
        few_shot_examples=few_shot_str,
        schema=schema_str,
        question=question
    )

MMLU_PROMPT_TEMPLATE = """The following are multiple choice questions (with answers) about {subcategory}.

{few_shot_examples}Question: {question}
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}
Answer:"""

def render_mmlu_prompt(subcategory: str, question: str, choices: list, few_shot_examples: list = None) -> str:
    few_shot_str = ""
    if few_shot_examples:
        for ex in few_shot_examples:
            few_shot_str += f"Question: {ex['question']}\nA. {ex['choices'][0]}\nB. {ex['choices'][1]}\nC. {ex['choices'][2]}\nD. {ex['choices'][3]}\nAnswer: {ex['answer']}\n\n"
            
    return MMLU_PROMPT_TEMPLATE.format(
        subcategory=subcategory,
        few_shot_examples=few_shot_str,
        question=question,
        choice_A=choices[0],
        choice_B=choices[1],
        choice_C=choices[2],
        choice_D=choices[3]
    )
