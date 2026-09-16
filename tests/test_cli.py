import pytest

from genval.cli import main


def test_help_lists_all_commands(capsys):
    main(["--help"])
    out = capsys.readouterr().out
    for command in ("generate", "validate", "classify", "llm-generate", "llm-parse", "filter", "verify"):
        assert command in out


def test_no_args_prints_help(capsys):
    main([])
    assert "usage: genval" in capsys.readouterr().out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "genval" in capsys.readouterr().out


def test_unknown_command_exits_with_error(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["bogus"])
    assert excinfo.value.code == 2
    assert "invalid command" in capsys.readouterr().err


def test_subcommand_help_via_lazy_dispatch(capsys):
    # filter has no torch dependency at module level, so this exercises
    # the lazy import path end to end.
    with pytest.raises(SystemExit) as excinfo:
        main(["filter", "--help"])
    assert excinfo.value.code == 0
    assert "--profile" in capsys.readouterr().out
