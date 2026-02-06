import torch
import math

def find_stripiness(image, device):
    dx = torch.roll(image, 1, 0) - image
    dy = torch.roll(image, 0, 1) - image
    a = (dx * dx).sum()
    b = (dx * dy).sum()
    c = (dy * dy).sum()

    trace = a + c
    det = a*c - b*b
    disc = math.sqrt(max(trace*trace - 4*det, 0))
    l1 = (trace + disc) / 2
    l2 = (trace - disc) / 2
    return (l1 - l2) / (l1 + l2 + 0.001)