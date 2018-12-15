from deep_eos.data import Corpus
from deep_eos.utils import get_char_context


def test_corpus():
    buffer = ". Dr. Prof. Müller geht zur Universität an der Str. 14.\nDas ist schön?\n"

    left_ws = 2
    right_ws = 2

    contexts = list(get_char_context(left_window=left_ws, right_window=right_ws, buffer=buffer))

    train_contexts = contexts[:2]
    dev_contexts = contexts[2:4]
    test_contexts = contexts[4:]

    corpus = Corpus(train_contexts=train_contexts, dev_contexts=dev_contexts,
                    test_contexts=test_contexts, batch_size=1)

    train_iter = corpus.train_iter
    test_iter = corpus.test_iter

    # Check if train example is correctly padded
    for train_example in train_iter:
        context = "".join([corpus.context_fields.vocab.itos[int(i)]
                           for i in train_example.text[0][0]])

        assert(context in {'▁▁.▁D', 'Dr.▁P'})

    # Check if test example is correctly padded
    for test_example in test_iter:
        context = "".join([corpus.context_fields.vocab.itos[int(i)]
                           for i in test_example.text[0][0]])

        assert(context in {'<unk><unk><unk>▁▁', '<unk><unk>.▁D'})
