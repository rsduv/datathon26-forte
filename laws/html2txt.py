"""HTML с zakon.uchet.kz -> чистый текст (абзацы с новой строки)."""
import html, re, sys
for p in sys.argv[1:]:
    t = open(p, encoding="utf-8", errors="ignore").read()
    m = re.search(r'<div[^>]+class="[^"]*(?:document|doc-text|text)[^"]*"', t)
    t = re.sub(r"(?is)<(script|style|head|nav|footer).*?</\1>", " ", t)
    t = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h\d>|</li>|</tr>", "\n", t)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    t = re.sub(r"[ \t\xa0]+", " ", t)
    t = "\n".join(l.strip() for l in t.splitlines() if l.strip())
    open(p.rsplit(".", 1)[0] + ".txt", "w", encoding="utf-8").write(t)
    print(p, len(t))
