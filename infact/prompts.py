import re
import string

LETTERS = string.ascii_uppercase

TEMPLATE = """Watch the video and answer the multiple-choice question. Respond with the single letter of the correct option only.

Question: {q}

Options:
{opts}

Answer:"""


def build_prompt(question, options):
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(options))
    return TEMPLATE.format(q=question.strip(), opts=opts)


_LETTER = re.compile(r"\b([A-Z])\b")


def parse_answer(text, n_options):
    if not text:
        return None
    head = text.strip().splitlines()[0][:64]
    m = _LETTER.search(head)
    if not m:
        return None
    i = LETTERS.index(m.group(1))
    return i if 0 <= i < n_options else None
