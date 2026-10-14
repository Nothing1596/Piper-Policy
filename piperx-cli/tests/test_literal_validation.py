"""Manual and model entrypoints share rules without changing their budgets."""
import pytest

from piperx_middleware.console import ConsoleController
from piperx_middleware.manual_parser import parse_manual_command, validate_literal


@pytest.mark.parametrize('value', [float('nan'), float('inf'), (1,), {1: 'x'}, set(), 10**1000])
def test_both_entrypoints_reject_non_json_literals(value):
    for validate in (validate_literal, ConsoleController._validate_finite_literals):
        with pytest.raises(ValueError):
            validate(value)


@pytest.mark.parametrize('value', ['x' * 10001, [0] * 501])
def test_model_budget_remains_larger_than_manual(value):
    with pytest.raises(ValueError):
        validate_literal({'nested': value})
    ConsoleController._validate_finite_literals({'nested': value})


@pytest.mark.parametrize('value', ['x' * 100001, [0] * 1001, {str(i): 0 for i in range(501)}])
def test_model_budget_still_bounded(value):
    with pytest.raises(ValueError):
        ConsoleController._validate_finite_literals(value)


def test_shared_depth_and_total_budgets():
    value = 0
    for _ in range(22):
        value = [value]
    for validate in (validate_literal, ConsoleController._validate_finite_literals):
        with pytest.raises(ValueError):
            validate(value)
    with pytest.raises(ValueError):
        validate_literal(['x'*10000]*11)
    with pytest.raises(ValueError):
        ConsoleController._validate_finite_literals(['x'*100000]*3)


def test_manual_remains_literal_only():
    assert parse_manual_command('tool(x=[1, 2])') == ('tool', {'x': [1, 2]})
    for text in ('tool(x=1e999)', 'tool(x=(1,))', 'tool(x=other())', 'tool(**{})'):
        with pytest.raises(ValueError):
            parse_manual_command(text)
