# Class presentation — 10 min

`slides.md` is a Marp-formatted Markdown deck (14 slides: 10 main + 1 Q&A + 2 backup + 1 notes).

## Render to PDF or HTML

```bash
# One-off via npx (no install)
npx @marp-team/marp-cli slides.md -o slides.pdf
npx @marp-team/marp-cli slides.md -o slides.html

# Or install once
npm install -g @marp-team/marp-cli
marp slides.md -o slides.pdf
```

Or use the VS Code [Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode) extension for live preview.

## Running time

Slides 1–11 target ~10 min (pacing notes at the bottom of `slides.md`).
Slides 12–14 are backup material for Q&A.
