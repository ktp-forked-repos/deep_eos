from deep_eos.utils import get_char_context


def test_get_char_context():
    buffer = ". Dr. Prof. Müller geht zur Universität an der Str. 14.\n"

    left_ws = 2
    right_ws = 2

    contexts = list(get_char_context(left_window=left_ws, right_window=right_ws, buffer=buffer))

    print(contexts)

    assert len(contexts) == 5
    assert contexts[0].context == "  . D"  # Left padding is ok?
    assert contexts[1].context == "Dr. P"
    assert contexts[2].context == "of. M"
    assert contexts[3].context == "tr. 1"
    assert contexts[4].context == "14.  "  # Right padding is ok?
