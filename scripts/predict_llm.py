"""Prompted-LLM baseline through Groq (free tier). Needs GROQ_API_KEY in .env.

    python scripts/predict_llm.py --model openai/gpt-oss-120b
    python scripts/predict_llm.py --model openai/gpt-oss-20b --limit 10

Writes predictions/<model>.jsonl one line at a time. If the daily token limit is
hit it stops; run the same command again later and it continues where it left off.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from logreport.dataset import read_jsonl
from logreport.prompt import SYSTEM, user_message


def is_daily_limit(err: Exception) -> bool:
    text = str(err)
    return "429" in text and ("per day" in text or "TPD" in text or "RPD" in text)


async def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq

    kwargs = {"reasoning_effort": "low"} if "gpt-oss" in args.model else {}
    llm = ChatGroq(model=args.model, temperature=0, max_retries=10, **kwargs)

    out = ROOT / "predictions" / f"{args.model.split('/')[-1]}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = {p["id"] for p in read_jsonl(out)} if out.exists() else set()
    test = read_jsonl(ROOT / "data" / "test.jsonl")[: args.limit]
    todo = [ex for ex in test if ex["id"] not in done]
    print(f"{args.model}: {len(done)} done, {len(todo)} to go -> {out.relative_to(ROOT)}")

    with out.open("a") as f:
        for i, ex in enumerate(todo, 1):
            t = time.perf_counter()
            try:
                msg = await llm.ainvoke([SystemMessage(SYSTEM),
                                         HumanMessage(user_message(ex["log"]))])
            except Exception as e:
                if is_daily_limit(e):
                    print("\nDaily token limit reached. Progress is saved; run again later.")
                    return
                print(f"  {ex['id']}: error {str(e)[:150]}")
                continue
            f.write(json.dumps({"id": ex["id"], "output": msg.content,
                                "seconds": round(time.perf_counter() - t, 2)}) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
