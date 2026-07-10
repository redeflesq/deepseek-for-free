"""Interactive terminal chat client.

Replaces old/chat_cli.py, which was a bare top-level script: it read
dsk/cookies.json directly to extract userToken, then dropped straight into
a while-True REPL with no error handling around a missing cookies file, a
missing userToken, or a network error mid-conversation - any of those would
crash with a raw traceback.

Behavior is otherwise identical: same prompt style ("> "), same /exit
command, same streaming text output, same parent_message_id threading via
the 'meta' chunk type. The only functional additions are a friendlier error
message when cookies.json/userToken is missing, and Ctrl+C no longer prints
a traceback.
"""

import json
import sys

from deepseek4free.client.api import DeepSeekAPI
from deepseek4free.config import get_settings


def _load_token_from_cookies() -> str:
    settings = get_settings()
    if not settings.cookies_path.is_file():
        print(
            f"error: {settings.cookies_path} not found - run "
            "`python -m deepseek4free.cloudflare.cookie_refresher --manual` first "
            "to log in and capture a userToken.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(settings.cookies_path, encoding="utf-8") as f:
        cookies_file = json.load(f)

    token_raw = cookies_file.get("cookies", {}).get("userToken")
    if not token_raw:
        print(
            f"error: no userToken found in {settings.cookies_path} - run "
            "`python -m deepseek4free.cloudflare.cookie_refresher --manual` to capture one.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        return json.loads(token_raw)["value"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return token_raw


def main() -> None:
    token = _load_token_from_cookies()
    api = DeepSeekAPI(token)
    chat_id = api.create_chat_session()
    parent_id = None

    print("Chat started. Type /exit to quit.\n")

    try:
        while True:
            try:
                prompt = input("> ")
            except EOFError:
                break

            if prompt.strip() == "/exit":
                break
            if not prompt.strip():
                continue

            for chunk in api.chat_completion(chat_id, prompt, parent_message_id=parent_id):
                if chunk["type"] == "text":
                    print(chunk["content"], end="", flush=True)
                elif chunk["type"] == "meta":
                    parent_id = chunk.get("parent_message_id")
            print("\n")
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
