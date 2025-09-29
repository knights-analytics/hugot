#!/usr/bin/env python
"""
Benchmark long-prompt text generation in Python (Hugging Face Transformers)
Analogous to Go test: TestTextGenerationLongPromptCUDA

Features:
- Loads a causal LM (default: microsoft/Phi-3.5-mini-instruct)
- Builds a chat-style prompt (system + user) similar to Go test
- Optional warmup pass
- Measures wall time, tokens/sec, prompt/new token counts
- Supports custom stop token IDs (default includes Phi end token 32007 if present)
- Custom stopping criteria so generation halts cleanly when any stop token appears
- Works on CUDA or CPU; picks an efficient dtype automatically (bfloat16/float16 on GPU)

Usage examples:
  python scripts/benchmark_text_generation.py \
      --model microsoft/Phi-3.5-mini-instruct \
      --device cuda \
      --max-new-tokens 500

  python scripts/benchmark_text_generation.py --device cpu --repetitions 3 --warmup

Install deps (minimal):
  pip install --upgrade transformers accelerate torch --index-url https://download.pytorch.org/whl/cu121  # adjust CUDA version as needed
Optional (int4 / 4bit quantization):
  pip install bitsandbytes
Then add: --load-in-4bit to arguments.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    GenerationConfig,
)

# ----------------------------- Long Prompt (same content as Go const) -----------------------------
LONG_TEST_PROMPT = r"""
Summarise this list of jsons:

{
    "_id": "68c19f301b7efa6a20ad1184",
    "index": 0,
    "guid": "ecee3e65-b8b3-4377-9948-b52d861455b0",
    "isActive": false,
    "balance": "$1,257.94",
    "picture": "http://placehold.it/32x32",
    "age": 37,
    "eyeColor": "brown",
    "name": "Mcfarland Coleman",
    "gender": "male",
    "company": "KOFFEE",
    "email": "mcfarlandcoleman@koffee.com",
    "phone": "+1 (949) 541-2357",
    "address": "209 Livonia Avenue, Faxon, Ohio, 8074",
    "about": "Aute est fugiat quis officia cillum tempor duis amet tempor sunt ad duis ut ea. Ex tempor aliqua aute quis labore labore dolore duis consequat deserunt. Eiusmod tempor culpa cillum nulla consectetur duis deserunt quis voluptate dolore incididunt eiusmod.\r\n",
    "registered": "2016-04-09T09:01:23 -02:00",
    "latitude": 23.201234,
    "longitude": -120.665901,
    "tags": ["ex", "et", "exercitation", "irure", "nisi", "minim", "minim"],
    "friends": [
      {"id": 0, "name": "Dorothea Kelly"},
      {"id": 1, "name": "Sally Espinoza"},
      {"id": 2, "name": "Whitney Wolfe"}
    ],
    "greeting": "Hello, Mcfarland Coleman! You have 1 unread messages.",
    "favoriteFruit": "banana"
  },
  {
    "_id": "68c19f30bb8a142738db1b4a",
    "index": 1,
    "guid": "aa666967-b69f-4a40-bb13-c1c1140036dc",
    "isActive": true,
    "balance": "$2,321.80",
    "picture": "http://placehold.it/32x32",
    "age": 20,
    "eyeColor": "brown",
    "name": "Daniels Webster",
    "gender": "male",
    "company": "CALLFLEX",
    "email": "danielswebster@callflex.com",
    "phone": "+1 (918) 488-3620",
    "address": "995 Lawrence Avenue, Harold, Illinois, 3159",
    "about": "Qui non proident minim do ad cillum eu mollit excepteur est laboris in incididunt. Incididunt adipisicing eu Lorem minim irure fugiat exercitation ullamco proident occaecat. Fugiat anim reprehenderit irure ex officia ad.\r\n",
    "registered": "2017-11-26T05:46:21 -01:00",
    "latitude": 2.345942,
    "longitude": 153.228303,
    "tags": ["non", "sint", "nulla", "aliqua", "laborum", "in", "esse"],
    "friends": [
      {"id": 0, "name": "Juliet Leonard"},
      {"id": 1, "name": "Melva Waters"},
      {"id": 2, "name": "Margarita Clark"}
    ],
    "greeting": "Hello, Daniels Webster! You have 5 unread messages.",
    "favoriteFruit": "banana"
  },
  {
    "_id": "68c19f30397bc7dadecfa155",
    "index": 2,
    "guid": "2d1d915c-327a-4c16-a4b3-84474dbbd391",
    "isActive": false,
    "balance": "$1,714.61",
    "picture": "http://placehold.it/32x32",
    "age": 31,
    "eyeColor": "green",
    "name": "Laverne Bean",
    "gender": "female",
    "company": "FISHLAND",
    "email": "lavernebean@fishland.com",
    "phone": "+1 (816) 444-2065",
    "address": "227 Holt Court, Russellville, Federated States Of Micronesia, 8054",
    "about": "Ea cupidatat occaecat consectetur quis Lorem quis sint duis. Do veniam aute cillum elit sit culpa amet sint ut magna incididunt eiusmod eiusmod minim. Cillum esse nulla nulla nisi laboris magna dolor.\r\n",
    "registered": "2019-05-24T05:10:44 -02:00",
    "latitude": 63.682722,
    "longitude": 167.857033,
    "tags": ["voluptate", "velit", "ipsum", "ipsum", "do", "consectetur", "cupidatat"],
    "friends": [
      {"id": 0, "name": "Leila Marquez"},
      {"id": 1, "name": "Margery Valdez"},
      {"id": 2, "name": "Liz Salazar"}
    ],
    "greeting": "Hello, Laverne Bean! You have 2 unread messages.",
    "favoriteFruit": "banana"
  },
  {
    "_id": "68c19f307fd1b9d20a4660c3",
    "index": 3,
    "guid": "aa4c6e09-21b8-4a99-a8fe-7650dd16576c",
    "isActive": false,
    "balance": "$2,378.44",
    "picture": "http://placehold.it/32x32",
    "age": 40,
    "eyeColor": "green",
    "name": "Elisabeth Galloway",
    "gender": "female",
    "company": "JIMBIES",
    "email": "elisabethgalloway@jimbies.com",
    "phone": "+1 (893) 576-2684",
    "address": "226 Seigel Court, Greenock, Washington, 2699",
    "about": "Et cupidatat nostrud veniam culpa nostrud aliquip consequat qui enim. Et reprehenderit sit est duis elit. Occaecat aute consectetur tempor reprehenderit incididunt id nisi commodo quis exercitation consequat aliquip reprehenderit incididunt. Esse magna aliquip nulla in magna.\r\n",
    "registered": "2016-03-12T11:51:43 -01:00",
    "latitude": -9.0195,
    "longitude": 23.194577,
    "tags": ["do", "ad", "veniam", "consequat", "ipsum", "tempor", "occaecat"],
    "friends": [
      {"id": 0, "name": "Trudy Bray"},
      {"id": 1, "name": "Crane Spence"},
      {"id": 2, "name": "Mullen Solis"}
    ],
    "greeting": "Hello, Elisabeth Galloway! You have 8 unread messages.",
    "favoriteFruit": "apple"
  },
  {
    "_id": "68c19f30c03d508648be1d07",
    "index": 4,
    "guid": "c6f6c4cc-8fce-4cca-96c5-483cdfd2deec",
    "isActive": false,
    "balance": "$3,110.34",
    "picture": "http://placehold.it/32x32",
    "age": 33,
    "eyeColor": "blue",
    "name": "Lee Hall",
    "gender": "male",
    "company": "LOVEPAD",
    "email": "leehall@lovepad.com",
    "phone": "+1 (876) 422-3809",
    "address": "905 Broadway , Albrightsville, Puerto Rico, 8907",
    "about": "Nostrud aute ea pariatur labore aliqua dolor enim aliqua nulla ullamco enim. Qui eu aliqua anim sunt non ea nisi enim aliquip eu aliquip duis consequat quis. Commodo ullamco sit aute officia laborum esse cillum ex consequat nostrud. Ex commodo exercitation minim aliquip quis fugiat Lorem ullamco commodo. Consectetur in culpa ut ex amet mollit ut dolor cupidatat. Esse irure tempor qui qui eiusmod.\r\n",
    "registered": "2024-09-08T07:18:31 -02:00",
    "latitude": 74.955239,
    "longitude": -173.268385,
    "tags": ["magna", "non", "pariatur", "nulla", "adipisicing", "commodo", "velit"],
    "friends": [
      {"id": 0, "name": "Frost Schroeder"},
      {"id": 1, "name": "Josefa Buck"},
      {"id": 2, "name": "Mann Hill"}
    ],
    "greeting": "Hello, Lee Hall! You have 10 unread messages.",
    "favoriteFruit": "strawberry"
  }
""".strip()

SYSTEM_PROMPT = (
    "You are an assistant that helps summarise json documents. "
    "For a list of json documents, provide a concise summary of the key details and differences."
)

# ----------------------------- Stopping Criteria -----------------------------
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids: List[int]):
        super().__init__()
        self.stop_ids = set(stop_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # type: ignore
        # input_ids shape: (batch, seq_len)
        last_token = input_ids[0, -1].item()
        return last_token in self.stop_ids

# ----------------------------- Config Dataclass -----------------------------
@dataclass
class BenchmarkResult:
    wall_time: float
    prompt_tokens: int
    new_tokens: int
    tokens_per_sec: float
    output_text: str

# ----------------------------- Utility Functions -----------------------------

def get_chat_formatted_input(tokenizer, system: str, user: str) -> str:
    """Attempt to use tokenizer chat template, fallback to manual Phi-style tags."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    # Fallback (Phi 3.5 typical style)
    return (
        f"<|system|>\n{system}\n<|end|>\n"
        f"<|user|>\n{user}\n<|end|>\n"
        f"<|assistant|>\n"
    )


def load_model_and_tokenizer(model_name: str, device: str, load_in_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    kwargs = {
        "device_map": "auto" if device == "cuda" else None,
        "trust_remote_code": True,
    }

    if load_in_4bit:
        try:
            kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            })
        except Exception:
            print("[WARN] 4-bit load requested but bitsandbytes not available or model unsupported. Falling back.")

    if device == "cuda" and not load_in_4bit:
        # Choose a half precision dtype automatically
        if torch.cuda.is_available():
            # Prefer bfloat16 if GPU supports it
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:  # Ampere+ generally good for bf16
                kwargs["torch_dtype"] = torch.bfloat16
            else:
                kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if device == "cpu" and not load_in_4bit:
        model = model.to(torch.float32)

    return model, tokenizer


def generate_once(model, tokenizer, formatted_input: str, max_new_tokens: int, stop_ids: List[int], temperature: float, top_p: float) -> BenchmarkResult:
    inputs = tokenizer(formatted_input, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    stopping = StoppingCriteriaList([StopOnTokens(stop_ids)]) if stop_ids else None

    prompt_tokens = input_ids.shape[1]
    torch.cuda.synchronize() if model.device.type == "cuda" else None
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature,
            top_p=top_p,
            stopping_criteria=stopping,
            eos_token_id=None,  # We'll handle custom stops
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )
    torch.cuda.synchronize() if model.device.type == "cuda" else None
    wall = time.time() - start

    new_tokens = output_ids.shape[1] - prompt_tokens
    gen_text = tokenizer.decode(output_ids[0][prompt_tokens:], skip_special_tokens=True)
    tps = new_tokens / wall if wall > 0 else float('inf')
    return BenchmarkResult(wall_time=wall, prompt_tokens=prompt_tokens, new_tokens=new_tokens, tokens_per_sec=tps, output_text=gen_text.strip())

# ----------------------------- Main -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark long prompt generation (Transformers)")
    p.add_argument('--model', default='microsoft/Phi-3.5-mini-instruct', help='HF model name or path')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-new-tokens', type=int, default=500)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--repetitions', type=int, default=1, help='Number of timed runs (excludes warmup)')
    p.add_argument('--warmup', action='store_true', help='Do an untimed warmup generation first')
    p.add_argument('--stop-token-ids', type=str, default='32007', help='Comma-separated custom stop token IDs (e.g. 32007,151643)')
    p.add_argument('--load-in-4bit', action='store_true', help='Attempt 4-bit load (bitsandbytes)')
    p.add_argument('--print-output', action='store_true', help='Print generated text')
    p.add_argument('--json', action='store_true', help='Emit final stats as JSON only')
    p.add_argument('--prompt-file', type=str, help='Optional path to alternate long prompt file')
    return p.parse_args()


def main():
    args = parse_args()

    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            print(f"Prompt file not found: {args.prompt_file}", file=sys.stderr)
            sys.exit(1)
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            long_prompt = f.read().strip()
    else:
        long_prompt = LONG_TEST_PROMPT

    stop_ids = []
    if args.stop_token_ids:
        try:
            stop_ids = [int(x) for x in args.stop_token_ids.split(',') if x.strip()]
        except ValueError:
            print('[WARN] Could not parse stop token IDs; ignoring.')
            stop_ids = []

    print(f"Loading model: {args.model} (device={args.device}, 4bit={args.load_in_4bit})")
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.load_in_4bit)

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted_input = get_chat_formatted_input(tokenizer, SYSTEM_PROMPT, long_prompt)

    if args.warmup:
        print('Warmup generation...')
        _ = generate_once(model, tokenizer, formatted_input, max_new_tokens=min(64, args.max_new_tokens), stop_ids=stop_ids, temperature=args.temperature, top_p=args.top_p)
        print('Warmup done.')

    results: List[BenchmarkResult] = []
    for i in range(args.repetitions):
        print(f"Run {i+1}/{args.repetitions}...")
        res = generate_once(model, tokenizer, formatted_input, args.max_new_tokens, stop_ids, args.temperature, args.top_p)
        results.append(res)
        print(f"  Wall: {res.wall_time:.3f}s | New tokens: {res.new_tokens} | {res.tokens_per_sec:.2f} tok/s")

    avg_wall = sum(r.wall_time for r in results) / len(results)
    avg_new = sum(r.new_tokens for r in results) / len(results)
    avg_tps = sum(r.tokens_per_sec for r in results) / len(results)

    summary = {
        'model': args.model,
        'device': args.device,
        'dtype': str(next(model.parameters()).dtype),
        'stop_token_ids': stop_ids,
        'repetitions': args.repetitions,
        'max_new_tokens': args.max_new_tokens,
        'avg_wall_time_sec': avg_wall,
        'avg_new_tokens': avg_new,
        'avg_tokens_per_sec': avg_tps,
        'prompt_tokens_first_run': results[0].prompt_tokens if results else None,
        'library_versions': {
            'torch': torch.__version__,
            'transformers': __import__('transformers').__version__,
        },
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print('\n=== Benchmark Summary ===')
        for k, v in summary.items():
            print(f"{k}: {v}")

    if args.print_output and results:
        print('\n=== Sample Output (last run) ===')
        print(results[-1].output_text)


if __name__ == '__main__':
    main()
