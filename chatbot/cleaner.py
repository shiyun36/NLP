import re 
## Export and clean whatsapp chat 
def remove_chat_metadata(chat_export_file):
    date_time = r"\[\d{1,2}\/\d{1,2}\/\d{2}, \d{1,2}:\d{2}:\d{2}(?:AM|PM)\] " # r"\[(\d{1,2}\/\d{1,2}\/\d{2}),\s(\d{1,2}:\d{2}:\d{2})\s(AM|PM)\]"  # e.g. "[11/10/23, 4:13:56 PM]" r"\[\d+\/\d+\/\d+, \d+:\d+:\d+ (?:AM|PM)]
    username = r"([\w\s]+|~[\w\s]+)"  # e.g. "Martin"
    #metadata_end = r" \:"  # ": "
    pattern = date_time + r"\s*" + username + r"\s*:" #+ metadata_end
    #print('pattern is', pattern)

    with open(chat_export_file, "r") as corpus_file:
        content = corpus_file.read()
        content = content.replace("\u202f", "")
        content = content.replace("\u200e", "")
    matches = re.findall(pattern, content)
    print('matches is', matches)  
    cleaned_corpus = re.sub(pattern, "", content)
    print('cleaned_corpus is', cleaned_corpus)
    return tuple(cleaned_corpus.split("\n"))

def remove_non_message_text(export_text_lines):
    final = []
    messages = export_text_lines[1:-1]
    words_to_exclude = ("Media omitted","joined using this group's invite link", "image omitted", "This message was deleted.", "added this group to the community:", "created this group", "You're now an admin")
    for line in messages:
        if not any(word in line for word in words_to_exclude):
            final.append(line.strip())
    return final


def clean_corpus(chat_export_file):
    message_corpus = remove_chat_metadata(chat_export_file)
    cleaned_corpus = remove_non_message_text(message_corpus)
    
    print('cleaned_corpus is', cleaned_corpus)
    return cleaned_corpus



