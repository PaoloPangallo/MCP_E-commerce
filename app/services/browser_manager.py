"""
browser_manager.py
Servizio singleton per gestire una sessione Playwright persistente in un thread dedicato.
Supporta Windows (ProactorEventLoop) e chiusura automatica per inattività.
"""

import asyncio
import threading
import logging
import base64
import time
import sys
from typing import Optional, Dict, Any, List
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

logger = logging.getLogger(__name__)

class BrowserManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright: Optional[Any] = None
        
        self._last_activity = 0
        self._timeout_sec = 60  # Default 1 minuto
        self._timer_task: Optional[asyncio.Task] = None
        self._visible = True

    @classmethod
    def get_instance(cls) -> "BrowserManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _run_loop(self):
        """Esegue l'event loop nel thread dedicato."""
        if sys.platform == "win32":
            self.loop = asyncio.ProactorEventLoop()
        else:
            self.loop = asyncio.new_event_loop()
        
        asyncio.set_event_loop(self.loop)
        logger.info("BrowserManager loop started in thread %s", threading.current_thread().name)
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()

    def start(self):
        """Inizia il thread dell'event loop se non è già partito."""
        with self._lock:
            if self.thread and self.thread.is_alive():
                return
            self.thread = threading.Thread(target=self._run_loop, daemon=True, name="BrowserManagerThread")
            self.thread.start()
            
            # Attende che il loop sia pronto
            max_wait = 50
            while (self.loop is None or not self.loop.is_running()) and max_wait > 0:
                time.sleep(0.1)
                max_wait -= 1
            
            if self.loop is None or not self.loop.is_running():
                raise RuntimeError("Failed to start BrowserManager event loop")
            
            logger.info("BrowserManager ready")

    def stop(self):
        """Ferma il loop e il thread."""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
            self.loop = None

    def set_timeout(self, seconds: int):
        self._timeout_sec = seconds
        logger.info("Browser timeout set to %d seconds", seconds)

    # ── Metodi Interni (eseguiti nel thread del loop) ────────────────────────

    async def _ensure_browser(self, visible: Optional[bool] = None):
        """Garantisce che il browser sia lanciato."""
        if visible is not None:
            self._visible = visible

        if self.browser is None:
            logger.info("Launching browser (visible=%s)...", self._visible)
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=not self._visible,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            self.context = await self.browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
            )
            self.page = await self.context.new_page()
            logger.info("Browser launched and page created")
        
        self._reset_timer()

    def _reset_timer(self):
        """Resetta il timer di inattività."""
        self._last_activity = time.time()
        if self._timer_task:
            self._timer_task.cancel()
        
        # Schedula il check nel loop
        self._timer_task = self.loop.create_task(self._inactivity_check())

    async def _inactivity_check(self):
        """Chiude il browser se inattivo per troppo tempo."""
        try:
            while True:
                await asyncio.sleep(5)
                elapsed = time.time() - self._last_activity
                if elapsed > self._timeout_sec:
                    logger.info("Inactivity timeout (%.1fs > %ds), closing browser", elapsed, self._timeout_sec)
                    await self._close_browser()
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Error in inactivity check: %s", e)

    async def _close_browser(self):
        """Chiude le risorse Playwright."""
        try:
            if self.page: await self.page.close()
            if self.context: await self.context.close()
            if self.browser: await self.browser.close()
            if self.playwright: await self.playwright.stop()
        except Exception:
            logger.exception("Error closing browser resources")
        finally:
            self.page = None
            self.context = None
            self.browser = None
            self.playwright = None
            logger.info("All browser resources released")

    async def _get_state(self) -> Dict[str, Any]:
        """Cattura lo stato corrente (URL, Screenshot, etc.)."""
        if not self.page:
            return {"status": "closed", "message": "Browser is not running"}
        
        screenshot_b64 = ""
        try:
            # Screenshot leggero per velocità
            screenshot = await self.page.screenshot(type="jpeg", quality=60)
            screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")
        except Exception as e:
            logger.warning("Could not take screenshot: %s", e)

        title = ""
        try:
            title = await self.page.title()
        except Exception:
            pass

        return {
            "status": "open",
            "url": self.page.url,
            "title": title,
            "screenshot": screenshot_b64,
        }

    # ── Metodi Pubblici (Thread-safe) ────────────────────────────────────────

    def run_coroutine(self, coro) -> Any:
        """Esegue una coroutine nel thread del loop e attende il risultato."""
        self.start()
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def navigate(self, url: str) -> Dict[str, Any]:
        async def _nav():
            await self._ensure_browser()
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return await self._get_state()
        return self.run_coroutine(_nav())

    def click(self, selector: str) -> Dict[str, Any]:
        async def _click():
            await self._ensure_browser()
            await self.page.click(selector, timeout=15000)
            return await self._get_state()
        return self.run_coroutine(_click())

    def type(self, selector: str, text: str, press_enter: bool = False) -> Dict[str, Any]:
        async def _type():
            await self._ensure_browser()
            await self.page.fill(selector, text, timeout=10000)
            if press_enter:
                await self.page.press(selector, "Enter")
            return await self._get_state()
        return self.run_coroutine(_type())

    def get_view(self) -> Dict[str, Any]:
        return self.run_coroutine(self._get_state())

    def close(self) -> Dict[str, Any]:
        self.run_coroutine(self._close_browser())
        return {"status": "closed"}
