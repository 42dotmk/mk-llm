from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


class FixedIllogicalSentencesOutputItem(BaseModel):
    original_sentence: str = Field(description="The original sentence that is illogical in the translation")
    fixed_sentence_translation: str = Field(
        description="The fixed sentence translation. It should be translated according to the original text  and can be restructured in a way that it makes sence in Macedonian.")


class FixIllogicalSentencesOutputTemplate(BaseModel):
    sentences: List[FixedIllogicalSentencesOutputItem] = Field(
        description="The fixed illogical sentences in the translation")


fixed_illogical_sentences_parser = JsonOutputParser(pydantic_object=FixIllogicalSentencesOutputTemplate)
FIX_ILLOGICAL_SENTENCES_PROMPT_TEMPLATE = """You will be given 2 texts, one is the original text written in English and the other text is a Macedonian translation.
Your task is to fix ALL of the illogical sentences in the Macedonian translation according to the original text, and translate them in a way that it makes sense in Macedonian.
Use standard Macedonian language in the translations.
ORIGINAL TEXT:
{original_text}
TRANSLATED TEXT:
{translated_text}
ILLOGICAL_SENTENCES:
{illogical_sentences}

NONEXISTENT WORDS:
{nonexistent_words}

{format_instruction}
"""

fix_illogical_sentences_prompt = PromptTemplate(template=FIX_ILLOGICAL_SENTENCES_PROMPT_TEMPLATE,
                                                input_variables=["original_text", "translated_text",
                                                                 "illogical_sentences", "nonexistent_words"],
                                                partial_variables={
                                                    "format_instruction": fixed_illogical_sentences_parser.get_format_instructions()}, )


def get_chain(model='gpt-4-turbo-2024-04-09'):
    model = ChatOpenAI(model=model, temperature=0.1,max_tokens=4000)
    return fix_illogical_sentences_prompt | model | fixed_illogical_sentences_parser


def fix_illogical_sentences(original_text: str, translated_text: str, illogical_sentences: list,
                            nonexistent_words: dict
                            , model='gpt-4-turbo-2024-04-09'):
    chain = get_chain(model)
    return chain.invoke(
        {"original_text": original_text, "translated_text": translated_text, "illogical_sentences": illogical_sentences,
         "nonexistent_words": nonexistent_words})
