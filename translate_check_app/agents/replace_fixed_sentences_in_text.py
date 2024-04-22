from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate

HAIKU = 'claude-3-haiku-20240307'

model = ChatAnthropic(model=HAIKU)


class FixedTranslationOutput(BaseModel):
    fixed_translation: str = Field(description="The fixed translation of the text")


fixed_translation_parser = JsonOutputParser(pydantic_object=FixedTranslationOutput)

FIX_ILLOGICAL_SENTENCES_PROMPT_TEMPLATE = """Your task is to rewrite the translated text from the beginning to the end by inserting the fixed translations in the place of the original translation.
For example, if the text is the following:
SentenceA.SentenceB.SentenceC.
and the fixed translation for SentenceB is "FixedSentenceB", the final text should be:
SentenceA.FixedSentenceB.SentenceC.

{translated_text}

FIXED_SENTENCES:
{fixed_sentences}

"""
#TODO: fix issues when the full text is not printed...
fix_illogical_sentences_prompt = PromptTemplate(template=FIX_ILLOGICAL_SENTENCES_PROMPT_TEMPLATE,
                                                input_variables=["original_text", "translated_text",
                                                                 "fixed_sentences"],
                                                partial_variables={
                                                    "format_instruction": fixed_translation_parser.get_format_instructions()}, )

str_parser = StrOutputParser()
def get_chain(model=HAIKU):
    model = ChatAnthropic(model=model, temperature=0.1,max_tokens=4000)
    return fix_illogical_sentences_prompt | model | str_parser


def replace_fixed_sentences_in_text(original_text: str, translation_text:str, fixed_sentences: list, model=HAIKU):
    chain = get_chain(model)

    return chain.invoke(
        {"original_text": original_text, "translated_text": translation_text, "fixed_sentences": fixed_sentences})
