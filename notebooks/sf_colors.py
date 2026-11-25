"""Solar Flare AI - figure colour concept v1 + font rule v2 (2026-11-24). Canonical palette and font for every paper and talk
figure of this project. Human-readable version: docs/figure_style.md. Change both together.

One question per colour:
  grey / black  DATA and REFERENCE: observations, inputs, the Kim et al. (2026) setting (FISM2 input, the authors' arrays and
                models, benchmark-input runs of our models) and other comparison conditions (GOES direct, simple rules).
  green         WHAT we forecast: the operational forecast (GOES -> FCN converter -> GRU-z / FCN-z ensemble), its 90 %
                conformal interval and the HF-absorption product.
  orange        WHEN we issue and stop: the armed peak detector, the two-stage issue (first forecast at detection,
                re-issue at peak + 10 min) and the validity of a forecast until the next declaration ("minutes in force").
Rules: prior work is never coloured (grey, dashed line or hollow marker); orange never marks a forecast value by itself;
no third accent colour; text and axes in greys.
Font: Times New Roman ONLY (text and math letters/digits); use_times() refuses to run without it - no substitute fonts.
"""
SF = {
    # grey / black: data and reference
    "truth": "#1a1a1a",       # observation / target (FISM2), solid line
    "input": "#4d4d4d",       # input series (GOES XRS, FISM2 0.1-0.8 nm input)
    "prior": "#8a8a8a",       # Kim et al. setting: benchmark input, authors' arrays and models (dashed line or hollow marker)
    "ref": "#b4b4b4",         # other comparison conditions (GOES direct, simple detector rules)
    "axis": "#7a7a7a",        # reference lines (zero, nominal 0.9 coverage), secondary text
    "soft": "#ececec",        # background spans (input window)
    # green: what we forecast
    "ours": "#34604A",        # operational forecast / final configuration, HF product
    "ours_mid": "#719080",    # second model or intermediate step (FCN-z next to GRU-z, linear converter)
    "ours_light": "#A4B7AE",  # interval shading, tints
    # orange: when we issue / stop
    "rule": "#C8621A",        # detector, adopted re-issue (+10 min), minutes in force, next declaration
    "rule_mid": "#DEA176",    # timing variant tested but not adopted (+5 min); secondary detector category
    "rule_pale": "#E9C0A3",   # second timing variant (+2 min-trained model)
    "rule_light": "#F4E0D1",  # span after the next declaration (forecast superseded)
}

# the same colours for LaTeX / TikZ (\usepackage{xcolor})
TIKZ = "\n".join(rf"\definecolor{{sf{k.title().replace('_', '')}}}{{HTML}}{{{v[1:].upper()}}}" for k, v in SF.items())

def use_times():
    """Make Times New Roman the text and math font of matplotlib figures (project rule). Raises if it is not installed, so that
    no figure is ever drawn with a substitute (Liberation Serif, DejaVu, ...). Install: copy times.ttf, timesbd.ttf, timesi.ttf,
    timesbi.ttf (Windows: C:\\Windows\\Fonts; Linux: conda-forge package mscorefonts) to ~/.fonts, run fc-cache -f and delete
    ~/.cache/matplotlib. Returns the path of the regular face."""
    import matplotlib
    from matplotlib import font_manager as fm
    faces = {}
    for st, w in (("normal", "normal"), ("italic", "normal"), ("normal", "bold"), ("italic", "bold")):
        try:
            p = fm.findfont(fm.FontProperties(family="Times New Roman", style=st, weight=w), fallback_to_default=False)
        except ValueError as e:
            raise RuntimeError("Times New Roman is not installed - figures must not be drawn with another font. " + use_times.__doc__) from e
        if fm.get_font(p).family_name != "Times New Roman":
            raise RuntimeError(f"Times New Roman {st}/{w} resolves to {p} ({fm.get_font(p).family_name})")
        faces[(st, w)] = p
    matplotlib.rcParams.update({"font.family": ["Times New Roman"], "mathtext.fontset": "custom",
                                "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
                                "mathtext.bf": "Times New Roman:bold", "mathtext.sf": "Times New Roman",
                                "mathtext.tt": "Times New Roman", "mathtext.cal": "Times New Roman:italic", "mathtext.fallback": "stix"})
    return faces[("normal", "normal")]


if __name__ == "__main__":
    print(TIKZ)
