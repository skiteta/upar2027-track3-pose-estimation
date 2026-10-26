"""Explicit validation-driven continuation policy; no convergence guarantee."""
from dataclasses import dataclass
import math


@dataclass
class ValidationPlateau:
    best_ap: float
    reference_ap: float
    min_delta: float = 0.0001
    minimum_epochs: int = 10
    patience: int = 8
    reduce_patience: int = 3
    factor: float = 0.3
    minimum_lr: float = 0.000002
    bad_checks: int = 0
    since_reduce: int = 0
    reductions: int = 0

    def observe(self, epoch, ap, lr):
        if not all(math.isfinite(x) for x in (ap, lr)) or not 0 <= ap <= 1 or lr <= 0:
            raise ValueError('Nonfinite/invalid validation or learning rate')
        best = ap > self.best_ap
        if best:
            self.best_ap = ap
        if ap > self.reference_ap + self.min_delta:
            self.reference_ap, self.bad_checks, self.since_reduce = ap, 0, 0
        else:
            self.bad_checks += 1
            self.since_reduce += 1
        # Give the reduced LR at least two full validation intervals before
        # stopping. A new reduction must not indefinitely postpone stopping.
        stop = (epoch >= self.minimum_epochs and self.bad_checks >= self.patience
                and self.reductions > 0 and self.since_reduce >= 2)
        reduced = False
        if not stop and self.since_reduce >= self.reduce_patience and lr > self.minimum_lr:
            lr = max(self.minimum_lr, lr * self.factor)
            self.reductions += 1
            self.since_reduce = 0
            reduced = True
        return {'save_best': best, 'stop': stop, 'lr': lr, 'reduced': reduced}

    def state_dict(self):
        return dict(vars(self))


def assess_undertraining(history, *, has_new_modules):
    """A declared trigger for an experiment, NOT proof of undertraining.

    Only training history exists for the parent phases. Flat training loss does
    not rule out undertraining; new modules warrant a validation-controlled test.
    """
    if len(history) < 6:
        raise ValueError('Need at least six complete epochs')
    values = [float(row['train_loss']) for row in history]
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Invalid parent learning curve')
    before, after = sum(values[-6:-3]) / 3, sum(values[-3:]) / 3
    improvement = (before - after) / before
    return {'undertraining_proven': False,
            'continue_trial': bool(has_new_modules or improvement >= .005),
            'last_two_3epoch_blocks_relative_loss_decrease': improvement,
            'reason': 'New modules lack a validation learning curve, or training loss is still decreasing.'
                      if has_new_modules or improvement >= .005 else
                      'No predefined trigger; lack of decline does not prove sufficient training.',
            'limitations': 'Training loss and a single final development AP cannot identify convergence.'}
