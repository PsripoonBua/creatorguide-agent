"""Command line: python -m creatorguide.cli "How long do I have to appeal a strike?" [--mode rag]"""
from __future__ import annotations

import argparse
import json

from .config import load_settings
from .service import build_service


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question")
    ap.add_argument("--mode", choices=["agent", "rag"], default="agent")
    args = ap.parse_args()
    service = build_service(load_settings())
    print(f"[retrieval: {service.retrieval_used}]")
    for event in service.stream(args.question, args.mode):
        if event["type"] == "tool_call":
            print(f"  -> {event['name']}({json.dumps(event['args'], ensure_ascii=False)})")
        elif event["type"] == "tool_result":
            print(f"  <- {'ok' if event['ok'] else 'ERROR'}: {event['preview'][:120]}")
        elif event["type"] == "note":
            print(f"  .. {event['message']}")
        elif event["type"] == "final":
            print(f"\n{'ANSWER' if event['answerable'] else 'NO ANSWER'} ({event['stop_reason']}): {event['answer']}")
            for s in event.get("sources", []):
                print(f"  source: {s['title']} - {s['url']}")


if __name__ == "__main__":
    main()
