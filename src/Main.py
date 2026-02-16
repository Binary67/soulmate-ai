from __future__ import annotations

from Agents.FriendAgent import build_friend_agent


def main() -> None:
    agent = build_friend_agent()
    query = "I have been feeling overwhelmed and a little lonely lately."
    response_text = agent.invoke({"messages": [{"role": "user", "content": query}]})

    if response_text:
        print(response_text)
    else:
        print("No response received from the agent.")


if __name__ == "__main__":
    main()
