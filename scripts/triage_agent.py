import os, json
import anthropic
from github import Github

gh = Github(os.environ["GITHUB_TOKEN"])
repo = gh.get_repo(os.environ["REPO_NAME"])
issue = repo.get_issue(int(os.environ["ISSUE_NUMBER"]))

client = anthropic.Anthropic()

# tools Claude can use
tools = [
    {
        "name": "apply_label",
        "description": "Apply one or more labels to the issue. Use labels like: bug, feature-request, question, documentation, needs-info, good-first-issue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of labels to apply"
                }
            },
            "required": ["labels"]
        }
    },
    {
        "name": "post_comment",
        "description": "Post a comment on the issue, e.g. to ask for clarification or acknowledge receipt.",
        "input_schema": {
            "type": "object",
            "properties": {
                "body": {"type": "string", "description": "The comment text (markdown supported)"}
            },
            "required": ["body"]
        }
    }
]

system_prompt = """You are an issue triage assistant for a GitHub repository.
Given an issue, you must:
1. Classify it by applying appropriate labels (bug, feature-request, question, documentation, needs-info, good-first-issue).
2. If the issue is missing key info (steps to reproduce for bugs, use case for features, etc.), post a friendly comment asking for it.
3. Always post a short acknowledgment comment letting the user know their issue was received.
Keep comments concise and friendly."""

# GitHub helpers

def get_existing_issues(limit: int = LATEST_ISSUES_LIMIT) -> str:
    """
    Fetches the most recent open issues (excluding the current one)
    and formats them into a string for the prompt.
    """
    open_issues = repo.get_issues(state="open")
    lines = []
    count = 0
    for existing in open_issues:
        if existing.number == issue.number:
            continue
        lines.append(
            f"- #{existing.number}: {existing.title}\n"
            f"  {(existing.body or '').strip()[:200]}"  # truncate long bodies
        )
        count += 1
        if count >= limit:
            break
    return "\n".join(lines) if lines else "(no other open issues)"


def apply_label(labels: list[str]) -> str:
    existing_label_names = [l.name for l in repo.get_labels()]
    for label in labels:
        if label not in existing_label_names:
            repo.create_label(label, "ededed")
    issue.add_to_labels(*labels)
    return f"Applied labels: {labels}"


def post_comment(body: str) -> str:
    issue.create_comment(body)
    return "Comment posted."


def mark_duplicate(original_issue_number: int, reason: str) -> str:
    original = repo.get_issue(original_issue_number)
    issue.create_comment(
        f"Thanks for the report! This looks like a duplicate of #{original_issue_number} "
        f"({original.html_url}).\n\n> {reason}\n\n"
        f"Please edit this issue to add any distinguishing details if you believe it's not a duplicate."
    )
    issue.add_to_labels("duplicate")
    return f"Marked as duplicate of #{original_issue_number}."


# Tool dispatch

def handle_tool_call(name: str, inputs: dict) -> str:
    if name == "apply_label":
        return apply_label(inputs["labels"])
    elif name == "post_comment":
        return post_comment(inputs["body"])
    elif name == "mark_duplicate":
        return mark_duplicate(inputs["original_issue_number"], inputs["reason"])
    elif name == "suggest_possible_duplicate":
        return suggest_possible_duplicate(inputs["related_issue_number"], inputs["reason"])
    return f"Unknown tool: {name}"

# Agentic loop

def build_initial_message() -> str:
    return (
        f"Please triage this new GitHub issue:\n\n"
        f"Title: {os.environ['ISSUE_TITLE']}\n"
        f"Body:\n{os.environ.get('ISSUE_BODY') or '(no description provided)'}\n\n"
        f"---\n"
        f"Here are the currently open issues for duplicate detection:\n\n"
        f"{get_existing_issues()}"
    )


def run_triage_agent():
    messages = [{"role": "user", "content": build_initial_message()}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = handle_tool_call(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    run_triage_agent()
