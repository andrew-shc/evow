"""Unit coverage for replay Accordion headers in the execution flow."""

import unittest

from GREENFIELD.replay.workflow import workflow_header


class WorkflowHeaderTest(unittest.TestCase):
    """Verify trace state is faithfully reflected in each Accordion header."""

    def test_waiting_stage_has_neutral_status_and_no_duration(self) -> None:
        label, state_class = workflow_header({"mode": "explicit", "events": []}, 0)

        self.assertEqual(state_class, "evow-stage-waiting")
        self.assertTrue(label.startswith("1. Choose source · "))
        self.assertNotIn("●", label)
        self.assertNotIn("○", label)
        self.assertNotIn("!", label)
        self.assertIn("Waiting", label)
        self.assertIn("—", label)

    def test_running_stage_shows_blue_status_and_elapsed_duration(self) -> None:
        label, state_class = workflow_header({
            "mode": "explicit",
            "events": [
                {"stage": "Video depth prior", "status": "running", "elapsed_seconds": 2},
                {"stage": "3D initialization", "status": "running", "elapsed_seconds": 7},
            ],
        }, 2)

        self.assertEqual(state_class, "evow-stage-running")
        self.assertIn("Running", label)
        self.assertIn("5.00s", label)

    def test_completed_stage_shows_green_status_and_final_duration(self) -> None:
        label, state_class = workflow_header({
            "mode": "explicit",
            "events": [
                {"stage": "Gaussian initialization", "status": "running", "elapsed_seconds": 3},
                {"stage": "3D initialization", "status": "complete", "elapsed_seconds": 8},
            ],
        }, 3)

        self.assertEqual(state_class, "evow-stage-complete")
        self.assertIn("Completed", label)
        self.assertIn("5.00s", label)

    def test_failed_stage_uses_failed_label_for_error_trace_event(self) -> None:
        label, state_class = workflow_header({
            "mode": "explicit",
            "events": [{"stage": "Input clip", "status": "error", "elapsed_seconds": 4}],
        }, 0)

        self.assertEqual(state_class, "evow-stage-failed")
        self.assertIn("Failed", label)
        self.assertIn("0.00s", label)

    def test_saved_completed_trace_keeps_its_recorded_duration(self) -> None:
        label, state_class = workflow_header({
            "mode": "implicit",
            "events": [
                {"stage": "Camera setup", "status": "running", "elapsed_seconds": 3},
                {"stage": "Camera setup", "status": "complete", "elapsed_seconds": 9},
                {"stage": "Run complete", "status": "complete", "elapsed_seconds": 10},
            ],
        }, 2)

        self.assertEqual(state_class, "evow-stage-complete")
        self.assertIn("Completed", label)
        self.assertIn("6.00s", label)


if __name__ == "__main__":
    unittest.main()
