def color_bar(contrib_str: str, importance: float) -> str:
    color = (
        '\033[91m' if abs(importance) >= 0.5 else
        '\033[93m' if abs(importance) >= 0.2 else
        '\033[92m'
    )
    return f"{color}{contrib_str}\033[0m"