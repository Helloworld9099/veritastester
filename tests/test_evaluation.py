from bridgecaption.evaluate import run_pair

def test_baseline_and_context_use_same_input_and_model():
    calls = []
    class Mock:
        def translate(self, request):
            calls.append(request)
            return "context" if request.context_enabled else "baseline"
    row = {"language":"es","source":"Está chido","reference":"It's cool",
           "topic":"casual","previous":["We liked it."],"glossary":{}}
    result = run_pair(row, Mock())
    assert result["baseline"] == "baseline"
    assert result["context"] == "context"
    assert calls[0].text == calls[1].text
    assert calls[0].context_enabled is False and calls[1].context_enabled is True
