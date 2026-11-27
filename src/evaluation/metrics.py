def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)
