from deep_eos.utils import get_char_context, calculate_evaluation_metrics


def test_get_char_context():
    buffer = ". Dr. Prof. Müller geht zur Universität an der Str. 14.\n"

    left_ws = 2
    right_ws = 2

    contexts = list(get_char_context(left_window=left_ws, right_window=right_ws, buffer=buffer))

    assert len(contexts) == 5
    assert contexts[0].context == "▁▁.▁D"  # Left padding is ok?
    assert contexts[1].context == "Dr.▁P"
    assert contexts[2].context == "of.▁M"
    assert contexts[3].context == "tr.▁1"
    assert contexts[4].context == "14.▁▁"  # Right padding is ok?

def test_calculate_evaluation_metrics():
    #sentence = "Dr. Prof. Müller geht zur Universität an der Str. 14.\n"

    # potential eos positions are on [2,8,48,52]

    # Real eos positions
    gold_positions = set([52])

    # First system tags something wrong...
    # tp = 52
    # fp = 8, 48
    # fn = {}
    # precision: 1 / (1 + 2) = 1/3
    # recall: 1 / (1 + 0) = 1.0
    # f-score: 2 * (1/3 * 1) / (1/3 + 1) = 0.5
    predicted_positions = set([8, 48, 52])

    prec, rec, f_score = calculate_evaluation_metrics(gold_positions=gold_positions,
                                                      predicted_positions=predicted_positions)

    assert(prec == 1/3)
    assert(rec == 1.0)
    assert(f_score == 0.5)
