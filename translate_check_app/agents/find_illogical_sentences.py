from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field


class IllogicalSentenceOutputItem(BaseModel):
    original_sentence: str = Field(description="The original sentence that is illogical in the translation")
    illogical_sentence: str = Field(description="The illogical sentence in the translation")


class FindIllogicalSentencesOutputTemplate(BaseModel):
    sentences: List[IllogicalSentenceOutputItem] = Field(description="The illogical sentences in the translation")


illogical_sentences_parser = JsonOutputParser(pydantic_object=FindIllogicalSentencesOutputTemplate)

find_illogical_sentences_prompt_template = """You will be given 2 texts, one is the original text written in English and the other text is a Macedonian translation
of the text. Your task is to find ALL of the illogical sentences in the Macedonian translation.
You are also given a list of words that do not exist in the Macedonian language. 
{nonexistent_words}

ORIGINAL TEXT:
{original_text}
TRANSLATED TEXT:
{translated_text}

{format_instruction}
"""

prompt = PromptTemplate(template=find_illogical_sentences_prompt_template,
                        input_variables=["original_text", "translated_text",'nonexistent_words'],
                        partial_variables={
                            "format_instruction": illogical_sentences_parser.get_format_instructions()}, )


def get_chain(model='gpt-4-turbo-2024-04-09'):
    model = ChatOpenAI(model=model, temperature=0,max_tokens=4000)
    return prompt | model | illogical_sentences_parser


def find_illogical_sentences(original_text: str, translated_text: str, nonexistent_words: dict,
                             model='gpt-4-turbo-2024-04-09'):
    chain = get_chain(model)
    return chain.invoke({"original_text": original_text, "translated_text": translated_text,'nonexistent_words':nonexistent_words})
