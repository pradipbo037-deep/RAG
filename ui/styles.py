def get_app_styles() -> str:
    return """
    <style>
    .status-pill {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 12px;
        margin-bottom: 10px;
    }
    .status-pill.success {
        background: rgba(76, 175, 80, 0.15);
        color: #2e7d32;
        border: 1px solid rgba(76, 175, 80, 0.45);
    }
    .status-pill.error {
        background: rgba(244, 67, 54, 0.12);
        color: #c62828;
        border: 1px solid rgba(244, 67, 54, 0.35);
    }
    .status-pill.warning {
        background: rgba(255, 193, 7, 0.15);
        color: #b26a00;
        border: 1px solid rgba(255, 193, 7, 0.35);
    }
    .stApp {
        background: #f5f7fb;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """
