# Figure style

All figures (paper and seminar) follow one colour concept and one font. Palette and font code: `notebooks/sf_colors.py`.

| colour | meaning | used for |
|---|---|---|
| grey / black | data and reference conditions | observations (black), inputs (dark grey), the Kim et al. (2026) setting — FISM2 input, the authors' arrays and models, benchmark-input runs (grey, dashed line or hollow marker) — and other comparison conditions such as GOES direct input (light grey) |
| green `#34604A` | **what** we forecast | the operational forecast (GOES → FCN converter → GRU-z / FCN-z ensemble), its 90 % conformal interval (light green), the HF-absorption product; intermediate steps in mid green `#719080` |
| orange `#C8621A` | **when** a forecast is issued or ends | the armed peak detector, the two-stage issue (+2 min, re-issue at +10 min), validity until the next declared flare ("minutes in force"), the span after that declaration (tint `#F4E0D1`) |

Rules: prior work is never coloured; orange never marks a forecast value by itself; no third accent colour; text and axes
in greys. Font: Times New Roman only — `sf_colors.use_times()` sets it for text and mathtext and stops if it is not installed
(Windows has it; on Linux copy times*.ttf to `~/.fonts`, run `fc-cache -f` and delete `~/.cache/matplotlib`). The TikZ
diagram (`paper/fig01_pipeline.tex`) is set with XeLaTeX and `\setmainfont{Times New Roman}`.
