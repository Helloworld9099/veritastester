from bridgecaption.preflight import inspect


def test_empty_project_cannot_be_called_verified(tmp_path):
    result = inspect(tmp_path)
    assert result["offline_code_checks_pass"] is False
    assert result["foundation_fully_verified"] is False
    assert result["training_started"] is False
    assert result["unverified_manual_gates"]
