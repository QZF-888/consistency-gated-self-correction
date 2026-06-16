from cgsc.consistency import gated_answer, initial_consistency, majority_answer


def test_majority_answer():
    assert majority_answer(['A', 'B', 'A'], 'choice', ['A', 'B'])[0] == 'A'


def test_initial_consistency():
    assert initial_consistency('A', ['A', 'B', 'A'], 'choice', ['A', 'B']) == 2 / 3


def test_gated_answer_replaces_low_consistency():
    answer, consistency, used = gated_answer('A', ['B', 'B', 'A'], 'choice', 0.5, ['A', 'B'])
    assert answer == 'B'
    assert consistency == 1 / 3
    assert used is True
