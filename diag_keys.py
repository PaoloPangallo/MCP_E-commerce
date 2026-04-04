
import os
import asyncio
import logging
from dotenv import load_dotenv
from scripts.l2r_user_simulator import LLMCaller

# Disable logger output for diagnostic
logging.getLogger("l2r_user_simulator").setLevel(logging.ERROR)

async def diag():
    sem = asyncio.Semaphore(1)
    caller = LLMCaller(sem)
    print("--- Key Discovery Report ---")
    for p, pool in caller.key_pools.items():
        if pool:
            print(f"{p:10}: {len(pool):2} keys found")
            # Mask keys for privacy but show first/last chars
            masked = [f"{k[:6]}...{k[-4:]}" for k in pool]
            # print(f"  {masked}")
    print("----------------------------")

if __name__ == "__main__":
    asyncio.run(diag())
