### Hi there 👋  

#### 🤖 AI Models

Tiny, honest, code-trained models — each does one narrow thing a script can't, runs on a CPU in milliseconds, and ships with its own benchmark, tests, and technical report.

- **[nano-proofread](https://github.com/vukrosic/nano-proofread)** — fixes the writing errors a spell-checker can't see (`their going to win` → `they're going to win`, `its raining` → `it's raining`), context-dependent homophones a lookup table can only guess at — 100% on the slice where context decides (any context-free script is 0% there), and 92% vs 36% on hand-written phrases it never trained on. A ~1M-param byte-level transformer trained 100% on code-generated data. → [model on Hugging Face](https://huggingface.co/vukrosic/nano-proofread)
- **[nano-g2p](https://github.com/vukrosic/nano-g2p)** — spells out how a word is pronounced (`though` → `DH OW`, `tough` → `T AH F`, `through` → `TH R UW`), the mapping English spelling rules genuinely can't compute — and it generalises to words it never saw: 72.1% vs 8.4% for a per-letter "sound it out" script (on held-out words). A ~1M-param byte-level transformer trained on a frozen public lexicon. → [model on Hugging Face](https://huggingface.co/vukrosic/nano-g2p)
- **[nano-spell](https://github.com/vukrosic/nano-spell)** — corrects a misspelled word to the word it meant (`recieve` → `receive`, `thier` → `their`), where nearest-dictionary-by-edit-distance keeps snapping to the wrong word — 86.8% vs a frequency dictionary's 73.7% (68.7% vs 29.8% on the hard slice). A ~1M-param byte-level transformer trained 100% on code-generated data. → [model on Hugging Face](https://huggingface.co/vukrosic/nano-spell)
- **[nano-vowel](https://github.com/vukrosic/nano-vowel)** — restores the vowels to a vowel-less word (`wrld` → `world`, `brd` → `bread`), the inverse of a lossy map no rule can compute — 84.7% vs a naive script's 68.7%. A ~1M-param byte-level transformer trained 100% on code-generated data. → [model on Hugging Face](https://huggingface.co/vukrosic/nano-vowel)
- **[nano-case](https://github.com/vukrosic/nano-case)** — converts messy, separator-less identifiers like `sdkmodel` or `HTTPREQUESTHANDLER` into any case style (snake/kebab/camel/pascal/const) — the slice a regex provably can't split (99.7% vs 8.2%). A ~1M-param byte-level transformer trained 100% on code-generated data. → [model on Hugging Face](https://huggingface.co/vukrosic/nano-case)
- **[nano-dates](https://github.com/vukrosic/nano-dates)** — turns natural date phrases like "next friday" or "third thursday of march" into ISO-8601 dates, with an honest map of where it breaks. A ~1M-param byte-level transformer trained 100% on code-generated data. → [model on Hugging Face](https://huggingface.co/vukrosic/nano-dates)

---

#### 🔗 Connect with me
- 📺 **YouTube**: [@vukrosic](https://www.youtube.com/@vukrosic)
- 🐦 **X (Twitter)**: [VukRosic99](https://x.com/VukRosic99)
- 💼 **LinkedIn**: [Vuk Rosić](https://www.linkedin.com/in/vuk-r-b71561164/)
- 🔴 **Red (Xiaohongshu)**: [vuk_plus](https://www.xiaohongshu.com/user/profile/679762ae000000000a03d5d1) 
- 📺 **Bilibili**: [vuk_ai](https://space.bilibili.com/3546833932519662) 
- 🎵 **Douyin**: [Vuk 武克](https://www.douyin.com/user/MS4wLjABAAAAwbJkSymbHOPgLLBchwAqSkT2Veb8sSft-FgLEtpsaq6KOApcEOQYn68ZD4Ggx7ht) (`ID: 92865879653`)
- 🌊 **Weibo**: [Vuk 武克](https://weibo.com/u/8007189005)

---

**Languages**: 
- 🇷🇸 **Serbian** (Mother Tongue)
- 🇺🇸 **English** (Bilingual / Completely Fluent)
- 🇨🇳 **Chinese** (Learning/Progressing — Goal: Professional proficiency in 2027)

**Let’s build open-source AI together.** ☕️
