from bridgecaption.asr_eval import normalized, score_units, edit_distance
from bridgecaption.context import slang_hints

def test_normalization_handles_punctuation_and_width():
    assert normalized('  Hello, WORLD! ') == 'hello world'
    assert normalized('ＡＢＣ。') == 'abc'

def test_words_and_characters_are_distinguished():
    assert score_units('Hello, world!', 'en') == ['hello', 'world']
    assert score_units('你好，世界。', 'zh') == ['你','好','世','界']
    assert edit_distance(['a','b'], ['a','c']) == 1

def test_slang_not_matched_inside_unrelated_words():
    assert slang_hints('Bet, that is fire', 'en')
    assert not slang_hints('alphabetical betrothal', 'en')
    assert slang_hints('他今天在摸鱼', 'zh')
