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

user_message = f"""Please triage this GitHub issue:

Title: {os.environ['ISSUE_TITLE']}

Body:
{os.environ['ISSUE_BODY'] or '(no description provided)'}"""

messages = [{"role": "user", "content": user_message}]

# loop until all tool calls are done
while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        tools=tools,
        messages=messages
    )

    # add assistant turn to history
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "end_turn":
        break

    tool_results = []
    for block in response.content:
        if block.type != "tool_use":
            continue

        result = ""
        if block.name == "apply_label":
            labels = block.input["labels"]
            # create labels if they don't exist, then apply
            existing = [l.name for l in repo.get_labels()]
            for label in labels:
                if label not in existing:
                    repo.create_label(label, "ededed")
            issue.add_to_labels(*labels)
            result = f"Applied labels: {labels}"

        elif block.name == "post_comment":
            issue.create_comment(block.input["body"])
            result = "Comment posted."

        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result
        })

    messages.append({"role": "user", "content": tool_results})
