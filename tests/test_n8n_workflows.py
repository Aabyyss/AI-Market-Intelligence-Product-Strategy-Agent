"""Regression tests for the n8n automation layer.

These pin the workflow JSON contracts that the live demo depends on.
The Route bug (config dropped on loop-back, so the second Wait got
``poll_seconds=None`` and the run died with "Invalid wait amount") is
covered by test_route_node_carries_config_on_every_iteration.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
N8N_DIR = REPO / "n8n"


def load_workflow(filename: str) -> tuple[dict, str]:
    path = N8N_DIR / filename
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw), raw


@pytest.fixture(scope="module")
def workflows() -> dict[str, dict]:
    return {
        "refresh": load_workflow("corpus_refresh.json")[0],
        "report": load_workflow("market_report.json")[0],
    }


def node(wf: dict, name: str) -> dict:
    for n in wf["nodes"]:
        if n.get("name") == name:
            return n
    raise AssertionError(f"node {name!r} not found in workflow {wf.get('name')!r}")


class TestScheduleTriggers:
    """The manual-run recipe starts from the trigger node by name."""

    def test_refresh_has_0600_trigger(self, workflows):
        names = [
            n["name"]
            for n in workflows["refresh"]["nodes"]
            if n["type"] == "n8n-nodes-base.scheduleTrigger"
        ]
        assert "Daily 06:00" in names

    def test_report_has_0700_trigger(self, workflows):
        names = [
            n["name"]
            for n in workflows["report"]["nodes"]
            if n["type"] == "n8n-nodes-base.scheduleTrigger"
        ]
        assert "Daily 07:00" in names


class TestConfigNode:
    """Config (Set) nodes pull knobs from the n8n process env."""

    def test_config_reads_api_base_env(self, workflows):
        for label, wf in workflows.items():
            config = json.dumps(node(wf, "Config"))
            assert "MARKET_INTEL_API_BASE" in config, f"{label}: api_base not from env"

    def test_config_reads_slack_webhook_env(self, workflows):
        for label, wf in workflows.items():
            config = json.dumps(node(wf, "Config"))
            assert "SLACK_WEBHOOK_URL" in config, f"{label}: webhook not from env"

    def test_config_has_poll_budget(self, workflows):
        report_config = json.dumps(node(workflows["report"], "Config"))
        assert "MARKET_INTEL_MAX_POLLS" in report_config
        assert "MARKET_INTEL_POLL_SECONDS" in report_config
        refresh_config = json.dumps(node(workflows["refresh"], "Config"))
        assert "MARKET_INTEL_MAX_POLLS" in refresh_config
        # the refresh has its own poll interval variable
        assert "MARKET_INTEL_REFRESH_POLL_SECONDS" in refresh_config


class TestRouteCarry:
    """The poll loop must re-attach config on every iteration.

    Regression: the original Route only forwarded job_id, so on the
    second loop pass ``poll_seconds`` was undefined and the Wait node
    raised NodeOperationError("Invalid wait amount").
    """

    def test_route_node_carries_config_on_every_iteration(self, workflows):
        for label, wf in workflows.items():
            route = node(wf, "Route")
            assert route["type"] == "n8n-nodes-base.code", f"{label}: Route is not Code"
            code = route["parameters"]["jsCode"]
            # reads config via the Config node...
            assert "$('Config')" in code, f"{label}: Route does not re-read Config"
            # ...and re-attaches every knob the downstream nodes need
            for knob in ("poll_seconds", "max_attempts", "api_base"):
                assert knob in code, f"{label}: Route drops {knob!r} on loop-back"

    def test_every_route_assignment_lists_poll_seconds(self, workflows):
        for label, wf in workflows.items():
            code = node(wf, "Route")["parameters"]["jsCode"]
            assignments = re.findall(r"(\w+):", code)
            assert "poll_seconds" in assignments, (
                f"{label}: poll_seconds missing from Route assignments"
            )


class TestSlackNotification:
    def test_report_guards_empty_webhook(self, workflows):
        wf = workflows["report"]
        guard = node(wf, "Slack configured?")
        conditions = guard["parameters"]["conditions"]["conditions"]
        assert any(
            "slack_webhook" in c["leftValue"] and c["operator"]["operation"] == "notEmpty"
            for c in conditions
        ), "report does not skip Slack when the webhook is unset"

    def test_notify_posts_json_text_payload(self, workflows):
        wf = workflows["report"]
        notify = node(wf, "Notify Slack")
        assert notify["type"] == "n8n-nodes-base.httpRequest"
        assert notify["parameters"]["method"] == "POST"
        assert "$json.slack_webhook" in notify["parameters"]["url"]
        assert "text" in notify["parameters"]["jsonBody"]

    def test_slack_text_is_ascii_only(self, workflows):
        """Emojis written through a Windows heredoc arrived as mojibake.

        Keep the notification text ASCII so any editor/toolchain can
        round-trip the workflow file without corrupting it.
        """
        _, raw_report = load_workflow("market_report.json")
        _, raw_refresh = load_workflow("corpus_refresh.json")
        for label, raw in (("report", raw_report), ("refresh", raw_refresh)):
            assert raw.encode("ascii", errors="strict").decode("ascii") == raw, (
                f"{label}: workflow JSON contains non-ASCII characters"
            )


class TestLauncher:
    """start_n8n.cmd must reproduce the environment the demo was verified with."""

    def test_launcher_pins_user_folder_and_env_access(self):
        script = (N8N_DIR / "start_n8n.cmd").read_text(encoding="utf-8")
        # n8n 2.x nests .n8n under N8N_USER_FOLDER: pointing it deeper than
        # the install's data dir creates a fresh, empty instance.
        folder = re.search(r'set "N8N_USER_FOLDER=(.+)"', script).group(1)
        assert folder.rstrip("\\/").endswith("data")
        # $env.* reads from nodes are denied unless this is false.
        assert 'set "N8N_BLOCK_ENV_ACCESS_IN_NODE=false"' in script
        # The launcher loads the repo .env so secrets stay out of git.
        assert ".env" in script
