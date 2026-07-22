from agentwire.irc import parse_irc_line


def test_parse_ircv3_account_tag() -> None:
    line = parse_irc_line("@account=trev;time=2026-07-19T00:00:00Z :trev!u@h PRIVMSG #codex :hello")
    assert line.command == "PRIVMSG"
    assert line.tags["account"] == "trev"
    assert line.params == ("#codex", "hello")


def test_parse_ircv3_tag_escaping() -> None:
    line = parse_irc_line("@example=hello\\sworld\\:ok :n!u@h TAGMSG #c")
    assert line.tags["example"] == "hello world;ok"
