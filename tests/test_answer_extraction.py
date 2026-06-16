from cgsc.answer_extraction import extract_answer, is_correct, normalize_numeric


def test_normalize_numeric():
    assert normalize_numeric('1,200.00') == '1200'
    assert normalize_numeric('$-0.0') == '0'


def test_extract_numeric_answer():
    assert extract_answer('Reasoning...\nFinal answer: 42', 'numeric') == '42'
    assert extract_answer('The final answer is 3.50', 'numeric') == '3.5'


def test_extract_choice_answer():
    assert extract_answer('After checking, Final answer: C', 'choice', ['A','B','C','D']) == 'C'
    assert extract_answer('I choose option B.', 'choice', ['A','B','C','D']) == 'B'


def test_is_correct_choice():
    assert is_correct('Final answer: A', 'A', 'choice', ['A','B']) is True
