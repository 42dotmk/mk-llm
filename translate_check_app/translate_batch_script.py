import json
import os

import anthropic
from tenacity import retry, wait_exponential, stop_after_attempt
import traceback
import pandas as pd

OPUS = 'claude-3-opus-20240229'
SONNET = 'claude-3-sonnet-20240229'
HAIKU = 'claude-3-haiku-20240307'

client = anthropic.AsyncAnthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
)


@retry(wait=wait_exponential(2), stop=stop_after_attempt(3))
async def translate_text(text: str, model=SONNET) -> str:
    try:
        system = """Ти си професионален преведувач од англиски на македонски јазик. 
Твојата задача е да го преведеш следниот текст на македонски и не мора да ги користиш истите зборови. 
Користи литературен македонски јазик и внимавај на родовите на именките. 
Не додавај дополнително објаснување само напиши го преводот."""
        text = f"{text}\nПреведи го горниот текст вклучувајки ги сите зборови кои мислиш дека се вишок во текстот. Задржи го форматот."
        response = await client.messages.create(
            system=system,
            messages=[{"role": "user", "content": text}],
            max_tokens=4000,
            temperature=1,
            model=model,
        )
        resp = response.content[0].text.strip()
        return resp
    except Exception as e:
        traceback.print_exc()


async def translate_batch(file: str, model=SONNET):
    data = pd.read_parquet(file)

    result_folder='output_data'
    os.makedirs(result_folder, exist_ok=True)

    result_file_name=file.replace('data_part','translated_data_part')
    result_file_name=result_file_name.replace('.parquet','.csv')
    result_file_name=result_file_name.replace('input_data',result_folder)

    data_dicts=data.to_dict(orient='records')
    num_batches=10

    batches=[[] for i in range(num_batches)]
    for i in range(0,len(data_dicts)):
        batches[i%num_batches].append(data_dicts[i])

    for batch_number,batch in enumerate(batches):
        print(f"Translating batch {batch_number+1} out of {num_batches}")
        tasks=[asyncio.create_task(translate_text(row['text'],model)) for row in batch]
        results=await asyncio.gather(*tasks)
        for i in range(0,len(batch)):
            batch[i]['translated_text_sonnet']=results[i]

    data=pd.DataFrame(data_dicts)
    data.to_csv(result_file_name,index=False)
    print(f"Translated file saved to {result_file_name}")

@retry(wait=wait_exponential(2), stop=stop_after_attempt(3))
async def check_translation(original_text:str,translation:str,model=HAIKU):
    system= """Given the original text in english and the translation in macedonian language, 
your job is to check if the text is translated according to these instructions:
The text needs to be translated into macedonian and it is not necessary to use the same exact words. Use standard macedonian language and watch out for the noun genders.
Do not add additional explanation and only write the translation. Keep the same format as in the original text and dont remove the extra words. The whole text must be translated."""

    prompt=(f"<ORIGINAL_TEXT>\n{original_text}\n<ORIGINAL_TEXT/>\n"
            f"<TRANSLATED_TEXT>\n{translation}\n<TRANSLATED_TEXT/>\n"
            f"The response must be in json parsable and in the following format:\n"
            f'{{"explanation":"why the text is translated well or not (do not use quotes.)","is_translated_according_to_instructions":true/false}}\n'
            f'return false in the json if at least one of the guidelines is broken.')
    try:
        start='{'
        response = await client.messages.create(
                system=system,
                messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": start}],
                max_tokens=4000,
                temperature=1,
                model=model,
            )
        resp = start+response.content[0].text.strip()
        resp=json.loads(resp)
        return resp
    except Exception as e:
        traceback.print_exc()
        raise e

@retry(wait=wait_exponential(2), stop=stop_after_attempt(3))
async def check_translation2(original_text:str,translation:str,model=HAIKU):
    system= """Given the original text in english and the translation in macedonian language, 
your job is to check if the text is translated well and fill the fields with false if there is at least one mistake in each"""

    prompt=(f"<ORIGINAL_TEXT>\n{original_text}\n<ORIGINAL_TEXT/>\n"
            f"<TRANSLATED_TEXT>\n{translation}\n<TRANSLATED_TEXT/>\n"
            f"The response must be in json parsable and in the following format:\n"
            f'{{'
            f'"format_is_the_same":true/false,'
            f'"translated_in_standard_macedonian":true/false,'
            f'"is_additional_explanation_added":true/false,'
            f'""'
            f'"whole_text_translated":true/false'
            f'}}\n'
            f'return false in the json if at least once the guidelines are broken.')
    try:
        start='{'
        response = await client.messages.create(
                system=system,
                messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": start}],
                max_tokens=4000,
                temperature=1,
                model=model,
            )
        resp = start+response.content[0].text.strip()
        resp=json.loads(resp)
        return resp
    except Exception as e:
        traceback.print_exc()
        raise e


async def check_translations_are_good(file_path:str,model=HAIKU):
    data = pd.read_csv(file_path)
    data_dicts=data.to_dict(orient='records')

    num_batches=10

    batches=[[] for i in range(num_batches)]
    for i in range(0,len(data_dicts)):
        batches[i%num_batches].append(data_dicts[i])

    for batch_number,batch in enumerate(batches):
        print(f"Translating batch {batch_number+1} out of {num_batches}")
        tasks=[asyncio.create_task(check_translation2(row['text'],row['translated_text_sonnet'],model=model)) for row in batch]
        results=await asyncio.gather(*tasks)

        for i in range(0,len(batch)):
            batch[i]['haiku_translation_check']=results[i]


    data=pd.DataFrame(data_dicts)
    data.to_csv(file_path,index=False)
    print(f"Translated file saved to {file_path}")




if __name__ == '__main__':
    # translate the data_part parquet
    import asyncio

    loop = asyncio.get_event_loop()
    # loop.run_until_complete(translate_batch('input_data/data_part_38.parquet'))
    loop.run_until_complete(check_translations_are_good('output_data/translated_data_part_39.csv'))