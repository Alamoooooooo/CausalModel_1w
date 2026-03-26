from pathlib import Path

p = Path("backtest_full_pipeline_v3.py")
s = p.read_text(encoding="utf-8")
start = s.find("    if not has_ps:")
end = s.find("\n\nif __name__ == \"__main__\":", start)
print(start, end)
print(s[start:end])
