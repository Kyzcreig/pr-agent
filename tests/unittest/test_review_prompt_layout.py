"""Prompt layout for cross-member prefix caching (FleetReview v2 spec §5.7, D11).

`pr_reviewer.extra_instructions_after_diff` (default false) moves the per-lens
`extra_instructions` block out of the system prompt and into the user turn AFTER
the diff, so reviewers that differ only in their extra instructions send a
byte-identical `system + user-up-to-and-including-the-diff` prefix.
"""
import hashlib

import pytest
from jinja2 import Environment, StrictUndefined

from pr_agent.config_loader import get_settings

DIFF = "## File: 'src/cache.py'\n\n@@ ... @@ def put():\n__new hunk__\n11  a = 1\n12 +b = 2\n"


def _vars(extra: str, after_diff) -> dict:
    variables = {
        "title": "Fix cache write",
        "branch": "fix/cache",
        "description": "Stops a stale write.",
        "language": "Python",
        "diff": DIFF,
        "num_pr_files": 1,
        "num_max_findings": 3,
        "require_score": False,
        "require_tests": True,
        "require_estimate_effort_to_review": True,
        "require_estimate_contribution_time_cost": False,
        "require_can_be_split_review": False,
        "require_security_review": True,
        "require_todo_scan": False,
        "question_str": "",
        "answer_str": "",
        "extra_instructions": extra,
        "skills_context": "",
        "repo_context": "",
        "commit_messages_str": "",
        "custom_labels": "",
        "enable_custom_labels": False,
        "is_ai_metadata": False,
        "related_tickets": [],
        "duplicate_prompt_examples": False,
        "date": "2026-09-25",
    }
    if after_diff is not None:
        variables["extra_instructions_after_diff"] = after_diff
    return variables


def _render(variables: dict) -> tuple[str, str]:
    environment = Environment(undefined=StrictUndefined)
    system = environment.from_string(get_settings().pr_review_prompt.system).render(variables)
    user = environment.from_string(get_settings().pr_review_prompt.user).render(variables)
    return system, user


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# sha256 of the system/user prompts rendered by the UNMODIFIED template
# (Kyzcreig/pr-agent@12042f6e) for _vars("LENS-A: audit state mutation.", None).
GOLDEN_SYSTEM_SHA = "9410cc495c17c1b7db1b9e8dfce8c664a0d73676bfe0eae94642ba766c11cf5e"
GOLDEN_USER_SHA = "7977541c910b3de6025f87d4ba58b354851897d26a3a55d0134dd8858c7ea522"


@pytest.mark.parametrize("after_diff", [None, False])
def test_flag_off_renders_byte_identical_to_pre_change_template(after_diff):
    system, user = _render(_vars("LENS-A: audit state mutation.", after_diff))
    assert _sha(system) == GOLDEN_SYSTEM_SHA
    assert _sha(user) == GOLDEN_USER_SHA


def test_flag_off_keeps_extra_instructions_in_system_prompt():
    system, user = _render(_vars("LENS-A: audit state mutation.", False))
    assert "LENS-A: audit state mutation." in system
    assert "LENS-A: audit state mutation." not in user


def test_flag_on_moves_extra_instructions_after_the_diff():
    system, user = _render(_vars("LENS-A: audit state mutation.", True))
    assert "LENS-A: audit state mutation." not in system
    assert user.index(DIFF.strip()) < user.index("LENS-A: audit state mutation.")


def test_flag_on_members_share_system_and_diff_prefix():
    system_a, user_a = _render(_vars("LENS-A: audit state mutation.", True))
    system_b, user_b = _render(_vars("LENS-B: audit test quality.\nSecond line.", True))
    assert system_a == system_b
    diff_end = user_a.index(DIFF.strip()) + len(DIFF.strip())
    assert user_a[:diff_end] == user_b[:diff_end]
    assert user_a != user_b


def test_flag_on_without_extra_instructions_matches_flag_off():
    assert _render(_vars("", True)) == _render(_vars("", False))


def test_extra_instructions_after_diff_defaults_false():
    assert get_settings().pr_reviewer.get("extra_instructions_after_diff", None) is False
