import torch


def save(model: torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load(model: torch.nn.Module, path: str) -> torch.nn.Module:
    model.load_state_dict(torch.load(path))
    return model
