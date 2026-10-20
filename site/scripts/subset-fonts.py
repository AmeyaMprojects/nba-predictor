"""Build the three self-hosted fonts from the full Google variable files in fonts-src/.

  archivo-text.woff2     Archivo, width 100, weight 400-700 (body)
  archivo-display.woff2  Archivo, width 64, weight 700-800 (condensed display)
  jetbrains-mono.woff2   JetBrains Mono, weight 400-500 (data)

Each keeps Basic Latin plus the few marks the site sets (· ’ “ ” … → ← − ‑ ● ○).
Run from site/:  python3 scripts/subset-fonts.py   (needs fonttools + brotli)
"""
import io
from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

U = list(range(0x20, 0x7F)) + [0xA0, 0xB7, 0x2011, 0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D,
                               0x2026, 0x2190, 0x2192, 0x2212, 0xD7, 0xB0, 0x25CF, 0x25CB]


def make(src, axes, out, features):
    f = instancer.instantiateVariableFont(TTFont(src), axes)
    buf = io.BytesIO()
    f.flavor = None
    f.save(buf)
    buf.seek(0)
    f = TTFont(buf)
    opts = subset.Options()
    opts.flavor = 'woff2'
    opts.layout_features = features
    opts.notdef_outline = True
    opts.name_IDs = ['*']
    s = subset.Subsetter(opts)
    s.populate(unicodes=U)
    s.subset(f)
    f.flavor = 'woff2'
    f.save(out)


make('fonts-src/archivo-full.woff2', {'wdth': 100, 'wght': (400, 700)}, 'public/fonts/archivo-text.woff2', ['kern', 'liga', 'calt', 'tnum', 'lnum'])
make('fonts-src/archivo-full.woff2', {'wdth': 64, 'wght': (700, 800)}, 'public/fonts/archivo-display.woff2', ['kern', 'liga', 'calt', 'tnum', 'lnum'])
make('fonts-src/jetbrains-mono-full.woff2', {'wght': (400, 500)}, 'public/fonts/jetbrains-mono.woff2', ['kern', 'tnum'])
