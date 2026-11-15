from bridgecaption.pipeline import Pipeline
from bridgecaption.sense_model import SenseExample, fit
from bridgecaption.provider import TranslationRequest


def synthetic_artifact():
    rows = []
    for i in range(20):
        meaning = "amazing" if i % 2 else "ill"
        rows.append(SenseExample(str(i), str(i), "en", "sick", "That was sick",
                     "great gaming skills" if i % 2 else "a severe fever",
                     "gaming" if i % 2 else "medical", meaning, "synthetic"))
    return {"model": fit(rows), "threshold": .51}


def test_learned_senses_are_hints_not_forced_translations():
    class Dummy:
        def __init__(self):
            self.calls = []
        def translate(self, request):
            self.calls.append(request)
            return "response"
    translator = Dummy()
    pipeline = Pipeline(translator=translator, sense_artifact=synthetic_artifact())
    pipeline.translate("We are discussing a great gaming play", "en", "es", topic="gaming")
    pipeline.translate("That was sick", "en", "es", topic="gaming")
    request = translator.calls[-1]
    assert isinstance(request, TranslationRequest)
    assert isinstance(request.sense_hints, dict)
    assert isinstance(request.previous, list)


def test_no_sense_hints_are_sent_when_context_is_off():
    class Dummy:
        def __init__(self):
            self.calls = []
        def translate(self, request):
            self.calls.append(request)
            return "response"
    translator = Dummy()
    pipeline = Pipeline(translator=translator, sense_artifact=synthetic_artifact())
    pipeline.translate("That was sick", "en", "es", contextual=False)
    assert translator.calls[-1].sense_hints == {}
