# -*- coding: utf-8 -*-
"""WebSocket 日志广播服务器 — 解决 Streamlit 重渲染导致的日志重复问题。

架构:
  LogBroadcaster (线程安全单例)
    ├─ asyncio event loop (daemon 线程)
    ├─ WebSocket server (自动分配本地端口)
    ├─ 日志历史缓冲区 (最近 500 条)
    └─ 线程安全 emit() → 广播到所有连接的 WebSocket 客户端

客户端 (浏览器):
  内嵌 JavaScript WebSocket 客户端
  连接 ws://localhost:{port} → 实时接收日志 → DOM 渲染

Streamlit 重渲染时 DOM 重建, WebSocket 自动重连,
服务端发送历史缓冲区实现无缝恢复。
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import socket
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_HISTORY: int = 500
_LOG_PORT: int | None = None

# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════
def _find_free_port() -> int:
    """自动查找可用本地端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ══════════════════════════════════════════════════════════════════════
# 日志广播器
# ══════════════════════════════════════════════════════════════════════
class LogBroadcaster:
    """WebSocket 日志广播器 — 线程安全单例。
    
    Example:
        >>> broadcaster = LogBroadcaster.get_instance()
        >>> broadcaster.emit("INFO", "训练开始")
        >>> print(broadcaster.port)  # 例如 9876
    """

    _instance: LogBroadcaster | None = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> LogBroadcaster:
        """获取或创建单例（双重检查锁保证线程安全）。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance.start()
        return cls._instance

    @classmethod
    def get_port(cls) -> int | None:
        """返回 WebSocket 端口 (未启动则返回 None)。"""
        return _LOG_PORT

    def __init__(self) -> None:
        self._port: int = 0
        self._clients: set[Any] = set()
        self._history: collections.deque[dict[str, Any]] = collections.deque(maxlen=_MAX_HISTORY)
        self._history_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._shutdown_event = threading.Event()

    @property
    def port(self) -> int:
        return self._port

    @property
    def history(self) -> list[dict[str, Any]]:
        with self._history_lock:
            return list(self._history)

    def start(self) -> None:
        """启动 WebSocket 服务器 (后台 daemon 线程)。"""
        if self._thread is not None:
            return
        
        self._port = _find_free_port()
        global _LOG_PORT
        _LOG_PORT = self._port
        
        self._thread = threading.Thread(
            target=self._run_event_loop, daemon=True, name="ws-log-server",
        )
        self._thread.start()
        self._ready.wait(timeout=5)
        logger.info("WebSocket 日志服务器已启动: ws://127.0.0.1:%d", self._port)

    def stop(self) -> None:
        """停止 WebSocket 服务器。"""
        self._shutdown_event.set()
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def emit(self, level: str, message: str) -> None:
        """线程安全地发送日志到所有 WebSocket 客户端。

        Args:
            level: 日志级别 (INFO/WARNING/ERROR/DEBUG)。
            message: 格式化后的日志文本。
        """
        entry = {"level": level, "message": message}
        with self._history_lock:
            self._history.append(entry)
            
        if self._loop is not None and self._loop.is_running() and self._clients:
            payload = json.dumps(entry, ensure_ascii=False)
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    # ── 内部方法 ─────────────────────────────────────────────────────
    def _run_event_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception:
            logger.exception("WebSocket 日志服务器异常退出")
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        import websockets
        
        async def handler(ws: Any) -> None:
            self._clients.add(ws)
            try:
                # 连接时先发送历史记录
                with self._history_lock:
                    history = list(self._history)
                if history:
                    await ws.send(json.dumps(
                        {"type": "history", "entries": history},
                        ensure_ascii=False,
                    ))
                # 保持连接，等待客户端断开或新消息
                async for _ in ws:
                    pass
            except websockets.ConnectionClosed:
                pass
            except Exception:
                logger.exception("WebSocket 客户端处理异常")
            finally:
                self._clients.discard(ws)

        async with websockets.serve(handler, "127.0.0.1", self._port) as server:
            self._ready.set()
            # 阻塞等待 shutdown 信号
            await self._loop.run_in_executor(None, self._shutdown_event.wait)

    async def _broadcast(self, payload: str) -> None:
        """广播消息到所有连接的客户端 (必须在 event loop 线程中调用)。"""
        if not self._clients:
            return
        dead: list[Any] = []
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


# ══════════════════════════════════════════════════════════════════════
# 日志 Handler 与辅助函数
# ══════════════════════════════════════════════════════════════════════
class WebSocketLogHandler(logging.Handler):
    """将标准 logging 记录转发到 WebSocket LogBroadcaster。
    
    安装到目标 logger 后，所有 emit 的日志都会实时推送到浏览器。
    线程安全——可在训练线程中安全使用。
    """

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(message)s", datefmt="%H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            broadcaster = LogBroadcaster.get_instance()
            broadcaster.emit(record.levelname, msg)
        except Exception:
            self.handleError(record)


def ensure_server() -> LogBroadcaster:
    """确保 WebSocket 日志服务器已启动。幂等——多次调用安全。"""
    return LogBroadcaster.get_instance()


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><style>
  body {{ margin: 0; background: #1e1e1e; color: #d4d4d4; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; line-height: 1.5; }}
  #log-container {{ padding: 8px; overflow-y: auto; height: {height}px; }}
  .log-entry {{ padding: 2px 4px; border-bottom: 1px solid #333; white-space: pre-wrap; word-break: break-all; }}
  .log-info {{ color: #6a9955; }}
  .log-warning {{ color: #dcdcaa; }}
  .log-error {{ color: #f44747; }}
  .log-debug {{ color: #808080; }}
  #status {{ position: fixed; top: 4px; right: 8px; font-size: 11px; color: #888; z-index: 10; background: rgba(30,30,30,0.8); padding: 2px 6px; border-radius: 4px; }}
</style></head>
<body>
<div id="status">● connecting...</div>
<div id="log-container"></div>
<script>
  const container = document.getElementById('log-container');
  const status = document.getElementById('status');
  let ws;

  function connect() {{
    ws = new WebSocket('ws://127.0.0.1:{port}');
    ws.onopen = () => {{ status.textContent = '● connected'; status.style.color = '#6a9955'; }};
    ws.onclose = () => {{ status.textContent = '● disconnected'; status.style.color = '#f44747'; setTimeout(connect, 3000); }};
    ws.onerror = () => {{ status.textContent = '● error'; status.style.color = '#f44747'; }};
    ws.onmessage = (e) => {{
      try {{
        const data = JSON.parse(e.data);
        if (data.type === 'history' && Array.isArray(data.entries)) {{
          data.entries.forEach(en => appendLog(en.message, en.level, true));
        }} else {{
          appendLog(data.message, data.level);
        }}
      }} catch (err) {{ console.warn('Log parse error', err); }}
    }};
  }}

  function appendLog(msg, level, isHistory = false) {{
    const div = document.createElement('div');
    div.className = `log-entry log-${{level.toLowerCase()}}`;
    div.textContent = msg;
    container.appendChild(div);
    if (!isHistory) container.scrollTop = container.scrollHeight;
  }}

  connect();
</script></body></html>"""


def write_log_html(target_dir: Path, height: int = 320) -> Path:
    """将 WebSocket 日志查看器 HTML 写入静态文件，返回文件路径。
    
    配合 `st.html(html_path.read_text())` 或 `st.iframe()` 使用。
    HTML 内嵌自动重连 JS，避免 Streamlit 重渲染导致 WebSocket 断开闪烁。
    """
    broadcaster = LogBroadcaster.get_instance()
    port = broadcaster.port
    html = _HTML_TEMPLATE.format(port=port, height=height)
    
    html_path = target_dir / "ws_log_viewer.html"
    current = html_path.read_text(encoding="utf-8") if html_path.exists() else ""
    if current != html:
        html_path.write_text(html, encoding="utf-8")
    return html_path