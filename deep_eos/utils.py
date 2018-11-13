"""Implement various helper functions."""

from deep_eos.data import Context

EOS_CHARS = ['.', ';', '!', '?', ':']
EOS_MARKER = '</eos>'
SEPS = [' ', '\n']


def get_char_context(left_window, right_window, buffer):
    """Implement method for fetching left and right context around a potential end-of-sentence.

    :param left_window: left window size
    :param right_window: right window size
    :param buffer: text buffer
    :return:
    """
    for position, char in enumerate(buffer):
        if char in EOS_CHARS and position + 1 < len(buffer) and buffer[position + 1] in SEPS:
            if position - left_window > 0:
                left_context = buffer[position - left_window: position]
            else:
                left_context = buffer[0: position].rjust(left_window)

            right_context = buffer[position + 1: position + right_window + 1]

            if len(right_context) != right_window:
                right_context = right_context.ljust(right_window)

            label = 1 if buffer[position + 1] == '\n' else 0

            context_str = left_context + char + right_context

            context_str = context_str.replace('\n', ' ')

            context = Context(context=context_str,
                              label=int(label),
                              position=position)

            yield context


def parse_dataset_to_buffer(filename):
    """Parse dataset to string buffer.

    :param filename: filename to be read
    :return:
    """
    sentences = []
    with open(filename, 'rt') as f_p:
        for line in f_p.readlines():
            line = line.rstrip()

            if line.startswith('# text =') and line[-1] in EOS_CHARS:
                sentences.append(line.split('=')[1].lstrip())

    return "\n".join(sentences)


def parse_file_to_buffer(filename):
    """Parse file directly into a text buffer.

    :param filename: filename to be read
    :return:
    """
    with open(filename, 'rt') as f_p:
        return "\n".join(f_p.readlines())
