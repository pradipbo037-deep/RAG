from __future__ import annotations


def render_status_badge(label: str, is_ok: bool = True) -> str:
    color = "success" if is_ok else "error"
    return f"<div class='status-pill {color}'>{label}</div>"
