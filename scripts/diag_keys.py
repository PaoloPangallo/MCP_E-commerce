import os
import sys
import asyncio
import logging
import time

# Ensure we can import from the parent directory where l2r_user_simulator.py is
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.l2r_user_simulator import LLMCaller, PERSONAS

# Disable verbose logging
logging.getLogger("l2r_user_simulator").setLevel(logging.ERROR)

async def diag():
    sem = asyncio.Semaphore(1)
    caller = LLMCaller(sem)
    
    print("\n" + "="*40)
    print("      LTR SIMULATOR KEY DIAGNOSTIC")
    print("="*40)
    
    # 1. Key Discovery Report
    print(f"\n[1] Key Discovery Report:")
    for p, pool in caller.key_pools.items():
        if pool:
            print(f"  - {p:10}: {len(pool):2} keys found")
        else:
            print(f"  - {p:10}: NO KEYS FOUND")

    # 2. Persona Model Check (OpenRouter :free check)
    print(f"\n[2] Persona Model Check:")
    or_warnings = []
    for p_name, cfg in PERSONAS.items():
        prov = cfg.get("provider")
        model = cfg.get("model", "")
        if prov == "openrouter":
            if not model.endswith(":free"):
                or_warnings.append(f"Persona '{p_name}' uses paid model: {model}")
            else:
                print(f"  - {p_name:18}: {model} (OK)")
        else:
            print(f"  - {p_name:18}: {model} ({prov})")

    if or_warnings:
        print("\n[!] WARNING: Paid OpenRouter models detected!")
        for w in or_warnings:
            print(f"  CAUTION: {w}")
    else:
        print("\n[OK] All OpenRouter models are in the free tier.")

    print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    asyncio.run(diag())
