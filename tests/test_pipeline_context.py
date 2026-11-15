from bridgecaption.pipeline import Pipeline


class Dummy:
    def __init__(self, supports_context=True):
        self.calls = []
        self.supports_context = supports_context

    def translate(self, request):
        self.calls.append(request)
        return "translated"


def test_slang_uses_relevant_recent_context():
    model = Dummy()
    pipe = Pipeline(translator=model)
    pipe.translate("An amazing game", "en", "fr", topic="gaming")
    result = pipe.translate("That was sick", "en", "fr", topic="gaming")
    assert model.calls[-1].previous == ["An amazing game"]
    assert "1 prior" in result.context_info


def test_irrelevant_context_is_not_sent():
    model = Dummy()
    pipe = Pipeline(translator=model)
    pipe.translate("kids playing at a pool", "en", "fr")
    pipe.translate("The invoice was issued on Monday", "en", "fr")
    assert model.calls[-1].previous == []


def test_context_free_provider_never_receives_previous_speech():
    model = Dummy(supports_context=False)
    pipe = Pipeline(translator=model)
    pipe.translate("private conversation", "en", "fr", contextual=True)
    pipe.translate("the next sentence", "en", "fr", contextual=True)
    assert model.calls[-1].previous == []
    assert model.calls[-1].context_enabled is False
    assert pipe.history.snapshot() == []


def test_context_off_clears_history():
    model = Dummy()
    pipe = Pipeline(translator=model)
    pipe.translate("The movie was fantastic", "en", "es")
    pipe.translate("This is a new meeting", "en", "es", contextual=False)
    assert pipe.history.snapshot() == []


def test_all_context_is_explicit_baseline():
    model = Dummy()
    pipe = Pipeline(translator=model, context_mode="all")
    pipe.translate("cats playing at the beach", "en", "es")
    pipe.translate("The weather forecast is sunny", "en", "es")
    assert model.calls[-1].previous == ["cats playing at the beach"]
