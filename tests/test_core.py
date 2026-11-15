from bridgecaption.audio import audible
from bridgecaption.context import Conversation, parse_glossary, slang_hints
from bridgecaption.pipeline import Pipeline
from bridgecaption.provider import GeminiTranslator, TranslationRequest, prompt_for
from bridgecaption.training import classify, fit, holdout

def test_context_is_bounded():
    context = Conversation()
    for n in range(6):
        context.add(str(n))
    assert context.snapshot() == ["2", "3", "4", "5"]
    context.clear()
    assert context.snapshot() == []

def test_slang_is_a_hint_not_a_replacement():
    assert slang_hints("Ese código está chido", "es")
    assert slang_hints("他在摸鱼", "zh")
    assert not slang_hints("ordinary meeting", "en")

def test_glossary():
    assert parse_glossary("rama = branch\nbad\nhi = hello") == {"rama": "branch", "hi": "hello"}

def test_baseline_context_is_off():
    request = TranslationRequest("hola", "es", previous=["secret context"],
                                 glossary={"hola": "secret glossary"}, context_enabled=False)
    prompt = prompt_for(request)
    assert "secret context" not in prompt
    assert "secret glossary" not in prompt

def test_context_is_on():
    req = TranslationRequest("No cap", "en", previous=["a previous sentence"], context_enabled=True)
    assert "a previous sentence" in prompt_for(req)
    assert "no cap" in prompt_for(req).lower()

def test_missing_key_blocks():
    try:
        GeminiTranslator(api_key="").translate(TranslationRequest("hello", "en"))
    except RuntimeError as exc:
        assert "GEMINI_API_KEY" in str(exc)
    else:
        assert False

def test_capture_ignores_silence():
    assert not audible(bytes(20000))
    assert audible((5000).to_bytes(2, "little", signed=True) * 10000)

def test_topic_classifier_training():
    rows = [(f"code deploy branch {i}", "technology") for i in range(10)]
    rows += [(f"teacher class exam {i}", "education") for i in range(10)]
    train, test = holdout(rows)
    assert len(train) == 16 and len(test) == 4
    model = fit(train)
    assert classify(model, "code deploy branch")[0] == "technology"

def test_pipeline_does_not_reuse_current_utterance_as_history():
    calls = []
    class Dummy:
        def translate(self, request):
            calls.append(request)
            return "output"
    p = Pipeline(translator=Dummy())
    p.translate("one", "es", topic="general")
    p.translate("two", "es", topic="general")
    assert calls[0].previous == []
    assert calls[1].previous == ["one"]

def test_failed_translation_does_not_add_history():
    class Failure:
        def translate(self, request):
            raise RuntimeError("down")
    p = Pipeline(translator=Failure())
    try:
        p.translate("one", "es", topic="general")
    except RuntimeError:
        pass
    assert p.history.snapshot() == []
