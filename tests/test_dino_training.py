import tempfile
import unittest
import argparse
from pathlib import Path

import torch
from torch import nn

from rgc_dino.dino_training import load_checkpoint_into_model, load_training_state


class DinoTrainingHelpersTest(unittest.TestCase):
    def test_load_checkpoint_into_model_uses_model_payload(self) -> None:
        source = nn.Linear(2, 3)
        with torch.no_grad():
            source.weight.fill_(2.0)
            source.bias.fill_(0.5)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pth"
            torch.save({"model": source.state_dict(), "epoch": 4}, checkpoint)
            target = nn.Linear(2, 3)

            report = load_checkpoint_into_model(target, checkpoint)

            self.assertEqual(report.missing_keys, ())
            self.assertEqual(report.unexpected_keys, ())
            self.assertTrue(torch.equal(target.weight, source.weight))
            self.assertTrue(torch.equal(target.bias, source.bias))

    def test_load_checkpoint_accepts_official_payload_with_namespace(self) -> None:
        source = nn.Linear(2, 3)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint_with_args.pth"
            torch.save(
                {
                    "model": source.state_dict(),
                    "args": argparse.Namespace(epoch=1),
                },
                checkpoint,
            )
            target = nn.Linear(2, 3)

            report = load_checkpoint_into_model(target, checkpoint)

            self.assertEqual(report.missing_keys, ())
            self.assertEqual(report.unexpected_keys, ())

    def test_load_checkpoint_can_skip_mismatched_shapes(self) -> None:
        source = nn.Sequential(
            nn.Linear(2, 3),
            nn.Linear(3, 5),
        )
        with torch.no_grad():
            source[0].weight.fill_(2.0)
            source[0].bias.fill_(0.5)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "shape_mismatch_checkpoint.pth"
            torch.save({"model": source.state_dict()}, checkpoint)
            target = nn.Sequential(
                nn.Linear(2, 3),
                nn.Linear(3, 2),
            )

            report = load_checkpoint_into_model(target, checkpoint, skip_mismatched_shapes=True)

            self.assertTrue(torch.equal(target[0].weight, source[0].weight))
            self.assertTrue(torch.equal(target[0].bias, source[0].bias))
            self.assertEqual(report.skipped_keys, ("1.bias", "1.weight"))
            self.assertEqual(report.unexpected_keys, ())

    def test_load_training_state_restores_optimizer_scheduler_and_start_epoch(self) -> None:
        source = nn.Linear(2, 3)
        with torch.no_grad():
            source.weight.fill_(3.0)
            source.bias.fill_(0.25)
        source_optimizer = torch.optim.SGD(source.parameters(), lr=0.3, momentum=0.9)
        source_scheduler = torch.optim.lr_scheduler.StepLR(source_optimizer, step_size=1, gamma=0.1)
        source_optimizer.zero_grad()
        source(torch.ones(1, 2)).sum().backward()
        source_optimizer.step()
        source_scheduler.step()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "training_checkpoint.pth"
            torch.save(
                {
                    "model": source.state_dict(),
                    "optimizer": source_optimizer.state_dict(),
                    "lr_scheduler": source_scheduler.state_dict(),
                    "epoch": 2,
                },
                checkpoint,
            )
            target = nn.Linear(2, 3)
            target_optimizer = torch.optim.SGD(target.parameters(), lr=0.1, momentum=0.9)
            target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=1, gamma=0.1)

            report = load_training_state(
                target,
                checkpoint,
                optimizer=target_optimizer,
                lr_scheduler=target_scheduler,
            )

            self.assertEqual(report.start_epoch, 3)
            self.assertTrue(report.optimizer_loaded)
            self.assertTrue(report.lr_scheduler_loaded)
            self.assertEqual(report.missing_keys, ())
            self.assertEqual(report.unexpected_keys, ())
            self.assertTrue(torch.equal(target.weight, source.weight))
            self.assertTrue(torch.equal(target.bias, source.bias))
            self.assertEqual(target_optimizer.param_groups[0]["lr"], source_optimizer.param_groups[0]["lr"])
            self.assertEqual(target_scheduler.last_epoch, source_scheduler.last_epoch)


if __name__ == "__main__":
    unittest.main()
