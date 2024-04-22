from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

PROMPT_TEMPLATE = """"
ORIGINAL_TEXT:
{original_text}
TRANSLATED_TEXT:
{translated_text}

Which words in the text dont exist in the Macedonian language or are translated badly?
Include only the words you are certain about. 
Answer in a json format like this and include a short explanation.
[{{"word": "word1", "explanation": "explanation1","substitution_word":"substitution_word1"}}, {{"word": "word2", "explanation": "explanation2","substitution_word":"substitution_word2"}}]
"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                        input_variables=["translated_text"], )

output_parser = JsonOutputParser()



def get_chain(model='gpt-4-turbo-2024-04-09'):
    model = ChatOpenAI(model=model, temperature=0,max_tokens=4000)
    return prompt | model | output_parser

def find_nonexistent_words(original_text:str,translated_text: str, model='gpt-4-turbo-2024-04-09'):
    chain = get_chain(model)
    return chain.invoke({"translated_text": translated_text, "original_text": original_text})