from typer.testing import CliRunner

from speculators.__main__ import app


def test_convert_help_lists_domino():
    result = CliRunner().invoke(app, ["convert", "--help"])

    assert result.exit_code == 0
    assert "domino" in result.stdout
