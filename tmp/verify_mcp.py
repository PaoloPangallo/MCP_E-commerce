import asyncio
import logging
import json
from app.mcp.client import MCPToolClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    async with MCPToolClient("http://127.0.0.1:8050/mcp/mcp") as client:
        try:
            # 1. Verifica Risorse
            logger.info("Listing resources...")
            resources = await client.list_resources_async()
            resource_uris = [r.get("uri") for r in resources]
            logger.info(f"Resources found: {resource_uris}")
            
            assert "ebay://categories" in resource_uris
            assert "ebay://market-logic" in resource_uris
            
            # 2. Verifica Prompt
            logger.info("Listing prompts...")
            prompts = await client.list_prompts_async()
            prompt_names = [p.get("name") for p in prompts]
            logger.info(f"Prompts found: {prompt_names}")
            
            assert "deal_hunter" in prompt_names
            assert "tech_expert" in prompt_names
            
            # 3. Lettura Risorsa
            logger.info("Reading ebay://categories...")
            cat_content = await client.read_resource_async("ebay://categories")
            if cat_content:
                logger.info(f"Categories content preview: {cat_content[:100]}...")
            else:
                logger.error("Failed to read ebay://categories")
            
            logger.info("VERIFICATION SUCCESSFUL!")
            
        except Exception as e:
            logger.error(f"VERIFICATION FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(main())
