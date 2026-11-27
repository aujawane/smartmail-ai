from src.evaluation.metrics import accuracy


def test_accuracy_smoke():
    assert accuracy([1, 0, 1, 0], [1, 0, 0, 0]) == 0.75
