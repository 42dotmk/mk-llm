from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

PROMPT_TEMPLATE = """"
{translated_text}


Даден е текст напишан на македонски јазик но содржи многу граматички грешки и испуштени зборчиња. 

Напиши го претходниот текст со сите грешки поправени според следните правила за македонскиот јазик:

{imenki}

{glagoli}

{ostanato}
"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                        input_variables=["translated_text"], )

output_parser = StrOutputParser()



def get_chain(model='gpt-4-turbo-2024-04-09'):
    model = ChatOpenAI(model=model, temperature=0,max_tokens=4000)
    return prompt | model | output_parser

def fix_gramatical_errors(translated_text: str, model='gpt-4-turbo-2024-04-09'):
    chain = get_chain(model)
    with open('language_rules/imenki.txt', 'r', encoding='utf-8') as file:
        imenki_rules = file.read()
    with open('language_rules/glagoli.txt', 'r', encoding='utf-8') as file:
        glagoli_rules = file.read()
    with open('language_rules/ostanato.txt', 'r', encoding='utf-8') as file:
        ostanato_rules = file.read()

    return chain.invoke({"translated_text": translated_text, "imenki":imenki_rules, "glagoli":glagoli_rules, "ostanato":ostanato_rules})
