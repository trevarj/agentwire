from irc_bridge.paste import scan_secrets
from irc_bridge.text import preview, truncate_utf8


def test_utf8_truncation_never_splits_character() -> None:
    assert truncate_utf8("a🙂b", 4) == "a"
    assert truncate_utf8("a🙂b", 5) == "a🙂"


def test_preview_marks_truncated_reply() -> None:
    value, truncated = preview("one\ntwo\nthree", max_lines=2, max_bytes=200)
    assert truncated
    assert value.startswith("one\ntwo")
    assert "!paste" in value


def test_secret_scanner_detects_credential_assignments() -> None:
    findings = scan_secrets("API_TOKEN=abcdefghijklmno")
    assert [finding.rule for finding in findings] == ["secret assignment"]


def test_secret_scanner_allows_normal_agent_reply() -> None:
    assert scan_secrets("Tests passed. The service is ready.") == []
