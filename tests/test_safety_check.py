from unittest.mock import patch

from src.xai.safety_check import model_based_check, safety_check, secret_leak_check


def _fake_pipe(label: str, score: float = 0.9):
    return lambda text: [{"label": label, "score": score}]


def test_other_label_is_not_offensive():
    with patch("src.xai.safety_check._get_pipe", return_value=_fake_pipe("OTHER")):
        result = model_based_check("Merhaba, nasılsınız?")

    assert result["is_offensive"] is False


def test_non_other_labels_are_offensive():
    for label in ("INSULT", "PROFANITY", "RACIST", "SEXIST"):
        with patch("src.xai.safety_check._get_pipe", return_value=_fake_pipe(label)):
            result = model_based_check("örnek metin")

        assert result["is_offensive"] is True, f"label={label}"


def test_secret_leak_check_catches_password_assignment():
    hits = secret_leak_check('password = "gercek-gorunumlu-sifre123"')

    assert any(h["pattern"] == "OLASI_SIFRE" for h in hits)


def test_secret_leak_check_catches_api_key_assignment():
    hits = secret_leak_check('api_key = "sk-abcdefghijklmnop"')

    assert any(h["pattern"] == "OLASI_API_ANAHTARI" for h in hits)


def test_secret_leak_check_ignores_clean_text():
    hits = secret_leak_check("Bu doküman API anahtarınızı .env dosyasında saklamanızı önerir.")

    assert hits == []


def test_safety_check_is_safe_when_no_issues():
    with patch("src.xai.safety_check._get_pipe", return_value=_fake_pipe("OTHER")):
        result = safety_check("Bu bölümde kurulum adımları anlatılır.")

    assert result["is_safe"] is True


def test_safety_check_is_unsafe_when_secret_leaked():
    with patch("src.xai.safety_check._get_pipe", return_value=_fake_pipe("OTHER")):
        result = safety_check('token = "abcd1234efgh5678"')

    assert result["is_safe"] is False
    assert result["secret_hits"]


def test_safety_check_is_unsafe_when_model_flags_offensive():
    with patch("src.xai.safety_check._get_pipe", return_value=_fake_pipe("INSULT")):
        result = safety_check("örnek metin")

    assert result["is_safe"] is False
