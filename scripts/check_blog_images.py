"""Verify every image reference in Blog.md resolves to a real file."""
from pathlib import Path
import re
import sys

blog = Path("Blog.md")
content = blog.read_text()
refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content)

ok = True
for ref in refs:
    p = (blog.parent / ref).resolve()
    exists = p.exists()
    ok = ok and exists
    size = p.stat().st_size if exists else 0
    print(
        f"{'OK ' if exists else 'MISS'}  {ref:60s}  "
        f"({size:,} bytes)" if exists else f"MISS  {ref}",
    )

sys.exit(0 if ok else 1)
