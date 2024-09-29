import re

def highlight_text(text, query):
    highlighted_text = text
    for word in query.split():
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted_text = pattern.sub(lambda m: f'<span style="background-color: yellow; color: black;">{m.group()}</span>', highlighted_text)
    return highlighted_text