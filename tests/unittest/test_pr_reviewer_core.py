from types import SimpleNamespace
from unittest.mock import MagicMock

from pr_agent.config_loader import get_settings
from pr_agent.tools.pr_reviewer import PRReviewer, normalize_review_data


def _make_reviewer(git_provider=None):
    reviewer = PRReviewer.__new__(PRReviewer)
    reviewer.git_provider = git_provider or MagicMock()
    reviewer.pr_url = "https://example/pr/1"
    return reviewer


def test_normalize_review_data_emits_confidence_findings_and_tokens():
    data = {
        "review": {
            "confidence": "4",
            "key_issues_to_review": [{
                "relevant_file": "src/cache.py",
                "start_line": 12,
                "end_line": 15,
                "severity": "P1",
                "issue_header": "Stale write",
                "issue_content": "A retry can overwrite the newer value.",
                "action": "auto-fix",
            }],
        }
    }

    normalized = normalize_review_data(data, {"prompt_tokens": 100, "completion_tokens": 20})

    assert normalized == {
        "score": 4,
        "findings": [{
            "file": "src/cache.py",
            "line_start": 12,
            "line_end": 15,
            "severity": "P1",
            "title": "Stale write",
            "body": "A retry can overwrite the newer value.",
            "action": "auto-fix",
        }],
        "tokens": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def test_normalize_review_data_rejects_invalid_confidence_and_severity():
    data = {
        "review": {
            "confidence": 7,
            "key_issues_to_review": [{
                "relevant_file": "x.py",
                "start_line": "bad",
                "end_line": None,
                "severity": "critical",
                "issue_header": "Bad output",
                "issue_content": "Malformed model fields must not become trusted values.",
                "action": "fix-it",
            }],
        }
    }

    normalized = normalize_review_data(data, {})

    assert normalized["score"] is None
    assert normalized["findings"][0]["severity"] == "P2"
    assert normalized["findings"][0]["line_start"] is None
    assert normalized["findings"][0]["line_end"] is None
    assert normalized["findings"][0]["action"] == "ask-user"


def test_normalize_review_data_strips_model_string_fields():
    normalized = normalize_review_data({
        "review": {
            "confidence": "4\n",
            "key_issues_to_review": [{
                "relevant_file": "src/cache.py\n",
                "start_line": "12\n",
                "end_line": "15\n",
                "severity": "P1\n",
                "issue_header": "Stale write\n",
                "issue_content": "A retry can overwrite the newer value.\n",
                "action": "no-op\n",
            }],
        }
    })

    assert normalized["findings"] == [{
        "file": "src/cache.py",
        "line_start": 12,
        "line_end": 15,
        "severity": "P1",
        "title": "Stale write",
        "body": "A retry can overwrite the newer value.",
        "action": "ask-user",
    }]


def test_normalize_review_data_defaults_missing_action_fail_closed():
    normalized = normalize_review_data({
        "review": {
            "confidence": 3,
            "key_issues_to_review": [{
                "relevant_file": "src/design.py",
                "start_line": 8,
                "end_line": 8,
                "severity": "P1",
                "issue_header": "Intent question",
                "issue_content": "This challenges a product choice.",
            }],
        }
    })

    assert normalized["findings"][0]["action"] == "ask-user"


def test_should_publish_review_no_suggestions_respects_config():
    reviewer = _make_reviewer()
    settings = get_settings()
    original_publish_no_suggestions = settings.pr_reviewer.publish_output_no_suggestions

    try:
        settings.pr_reviewer.publish_output_no_suggestions = False
        assert reviewer._should_publish_review_no_suggestions("No major issues detected") is False
        assert reviewer._should_publish_review_no_suggestions("A major issue was detected") is True

        settings.pr_reviewer.publish_output_no_suggestions = True
        assert reviewer._should_publish_review_no_suggestions("No major issues detected") is True
    finally:
        settings.pr_reviewer.publish_output_no_suggestions = original_publish_no_suggestions


def test_can_run_incremental_review_skips_auto_mode_without_new_commit():
    reviewer = _make_reviewer()
    reviewer.is_auto = True
    reviewer.incremental = SimpleNamespace(first_new_commit_sha=None)

    assert reviewer._can_run_incremental_review() is False


def test_set_review_labels_replaces_stale_review_labels_and_keeps_user_labels():
    settings = get_settings()
    original = {
        "publish_output": settings.config.publish_output,
        "require_estimate_effort_to_review": settings.pr_reviewer.require_estimate_effort_to_review,
        "require_security_review": settings.pr_reviewer.require_security_review,
        "enable_review_labels_effort": settings.pr_reviewer.enable_review_labels_effort,
        "enable_review_labels_security": settings.pr_reviewer.enable_review_labels_security,
    }
    settings.config.publish_output = True
    settings.pr_reviewer.require_estimate_effort_to_review = True
    settings.pr_reviewer.require_security_review = True
    settings.pr_reviewer.enable_review_labels_effort = True
    settings.pr_reviewer.enable_review_labels_security = True
    git_provider = MagicMock()
    git_provider.get_pr_labels.return_value = ["Review effort 1/5", "Possible security concern", "keep-me"]
    reviewer = _make_reviewer(git_provider)
    data = {
        "review": {
            "estimated_effort_to_review_[1-5]": "3, moderate",
            "security_concerns": "yes",
        }
    }

    try:
        reviewer.set_review_labels(data)

        git_provider.publish_labels.assert_called_once_with([
            "Review effort 3/5",
            "Possible security concern",
            "keep-me",
        ])
    finally:
        settings.config.publish_output = original["publish_output"]
        settings.pr_reviewer.require_estimate_effort_to_review = original["require_estimate_effort_to_review"]
        settings.pr_reviewer.require_security_review = original["require_security_review"]
        settings.pr_reviewer.enable_review_labels_effort = original["enable_review_labels_effort"]
        settings.pr_reviewer.enable_review_labels_security = original["enable_review_labels_security"]


def test_get_user_answers_collects_question_and_answer_from_issue_comments():
    git_provider = MagicMock()
    git_provider.get_issue_comments.return_value = SimpleNamespace(reversed=[
        SimpleNamespace(body="Unrelated"),
        SimpleNamespace(body="Questions to better understand the PR:\n- Why?"),
        SimpleNamespace(body="/answer Because it fixes production."),
    ])
    reviewer = _make_reviewer(git_provider)
    reviewer.is_answer = True

    question, answer = reviewer._get_user_answers()

    assert question == "Questions to better understand the PR:\n- Why?"
    assert answer == "/answer Because it fixes production."


def test_init_maps_user_question_and_answer_to_correct_prompt_vars(monkeypatch):
    """Behavioral regression for the swapped-unpacking bug (#2496).

    The bug lived in ``PRReviewer.__init__``: ``_get_user_answers()`` returns
    ``(question, answer)`` but the tuple was unpacked as ``answer, question``,
    so the review prompt rendered the user's answer under ``{{ question_str }}``
    and the question under ``{{ answer_str }}``. This drives the real ``__init__``
    (external collaborators stubbed) and asserts each value lands in ``self.vars``
    under the correct key — so it fails if the unpack is ever swapped again,
    regardless of how the line is formatted.
    """
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    provider = MagicMock()
    provider.is_supported.return_value = True
    provider.get_languages.return_value = {}
    provider.get_files.return_value = []
    provider.get_issue_comments.return_value = SimpleNamespace(reversed=[
        SimpleNamespace(body="Questions to better understand the PR:\n- Why?"),
        SimpleNamespace(body="/answer Because it fixes production."),
    ])
    provider.get_pr_description.return_value = ("desc", [])

    monkeypatch.setattr(pr_reviewer_module, "get_git_provider_with_context", lambda pr_url: provider)
    monkeypatch.setattr(pr_reviewer_module, "get_main_pr_language", lambda languages, files: "Python")
    monkeypatch.setattr(pr_reviewer_module, "TokenHandler", MagicMock())

    reviewer = PRReviewer(
        "https://example/pr/1",
        is_answer=True,
        ai_handler=lambda: SimpleNamespace(main_pr_language=None),
    )

    assert reviewer.vars["question_str"] == "Questions to better understand the PR:\n- Why?"
    assert reviewer.vars["answer_str"] == "/answer Because it fixes production."
