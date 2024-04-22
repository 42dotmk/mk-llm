from find_illogical_sentences import find_illogical_sentences
from fix_illogical_sentences import fix_illogical_sentences
from translate_check_app.agents.find_nonexistent_words import find_nonexistent_words
from input_examples import *
from translate_check_app.agents.fix_gramatical_errors import fix_gramatical_errors
from translate_check_app.agents.replace_fixed_sentences_in_text import replace_fixed_sentences_in_text


def fix_translation(original_text=ENGLISH_TEXT, translated_text=INITIAL_TRANSLATED_TEXT):
    illogical_sentences_fixed = False
    gramatical_errors_fixed = True
    max_iterations = 4
    iteration = 0  # STOP the same sentences from being fixed
    while not (illogical_sentences_fixed and gramatical_errors_fixed) and iteration < max_iterations:
        nonexistent_words_result = find_nonexistent_words(original_text=original_text, translated_text=translated_text)
        print("NONEXISTENT WORDS:")
        print(nonexistent_words_result)

        illogical_sentences_result = find_illogical_sentences(original_text=original_text,
                                                              translated_text=translated_text,
                                                              nonexistent_words=nonexistent_words_result)
        illogical_sentences_result = illogical_sentences_result['sentences']
        print("ILLOGICAL SENTENCES:")
        for sentence in illogical_sentences_result:
            print(sentence)

        if len(illogical_sentences_result) == 0:
            illogical_sentences_fixed = True
        fixed_sentences = fix_illogical_sentences(original_text=original_text, translated_text=translated_text,
                                                  illogical_sentences=illogical_sentences_result,
                                                  nonexistent_words=nonexistent_words_result)
        fixed_sentences = fixed_sentences['sentences']

        print("FIXED SENTENCES:")
        for sentence in fixed_sentences:
            print(sentence)

        sentences_for_replacing = []

        for i in range(0, len(fixed_sentences)):
            d = {
                'original_translation': illogical_sentences_result[i]['illogical_sentence'],
                'fixed_translation': fixed_sentences[i]['fixed_sentence_translation']
            }
            sentences_for_replacing.append(d)

        # find and replace with
        remaining_sentences = []
        for sentence in sentences_for_replacing:
            if sentence['original_translation'] in translated_text:
                translated_text = translated_text.replace(sentence['original_translation'],
                                                          sentence['fixed_translation'])
                print(f"{sentence['original_translation']} replaced with {sentence['fixed_translation']}")
            else:
                remaining_sentences.append(sentence)

        if len(remaining_sentences) > 0:  # print(f"FIXING {len(remaining_sentences)} REMAINING SENTENCES:")
            fixed_translation = replace_fixed_sentences_in_text(original_text=original_text,
                                                                translation_text=translated_text,
                                                                fixed_sentences=sentences_for_replacing)
            translated_text = fixed_translation

        print("NEW TRANSLATION:")
        print(translated_text)
        iteration += 1

    translated_text = fix_gramatical_errors(translated_text=translated_text)

    return translated_text


if __name__ == '__main__':
    fixed_text = fix_translation(original_text=ENGLISH_TEXT, translated_text=INITIAL_TRANSLATED_TEXT)
