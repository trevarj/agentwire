from agentwire.redaction import scan_secrets
from agentwire.text import clean_block, clean_text, truncate_utf8


def test_utf8_truncation_never_splits_character() -> None:
    assert truncate_utf8("a🙂b", 4) == "a"
    assert truncate_utf8("a🙂b", 5) == "a🙂"


def test_unicode_icons_survive_while_irc_control_codes_are_removed() -> None:
    assert clean_text("\x02bold\x02 🟢") == "bold 🟢"


def test_block_cleanup_preserves_meaningful_indentation() -> None:
    assert clean_block("\r\n M changed.py\r\n  context\x02\r\n") == " M changed.py\n  context"


def test_secret_scanner_detects_credential_assignments() -> None:
    findings = scan_secrets("API_TOKEN=abcdefghijklmno")
    assert [finding.rule for finding in findings] == ["secret assignment"]


def test_secret_scanner_allows_normal_agent_reply() -> None:
    assert scan_secrets("Tests passed. The service is ready.") == []
