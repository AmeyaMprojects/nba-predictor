// All site copy and links live here.
// Owner name and LinkedIn profile confirmed by the owner (2026-10-21).
export const site = {
  name: 'Sealed before tip-off',
  short: 'Sealed',
  owner: { name: 'Ameya Mhatre', placeholder: false },
  url: 'https://ameyamprojects.github.io/nba-predictor/',
  description:
    'Every NBA game called before tip-off and sealed in a public git commit. Tonight\'s calls, the running record against the market, and the proof of when each call was made.',
  repo: 'https://github.com/AmeyaMprojects/nba-predictor',
  commit: (hash: string) => `https://github.com/AmeyaMprojects/nba-predictor/commit/${hash}`,
  rawFile: (path: string) => `https://github.com/AmeyaMprojects/nba-predictor/blob/main/${path}`,
  linkedin: { href: 'https://www.linkedin.com/in/ameya-mhatre-553003307/', placeholder: false },
  nav: [
    { label: 'Tonight', path: '/#tonight' },
    { label: 'Record', path: '/#record' },
    { label: 'Ledger', path: '/#ledger' },
    { label: 'Method', path: '/method/' },
  ],
};

// Walk-forward test, 2020-21 to 2025-26, games with a closing line (project spec, result of 2026-10-08).
export const backtest = {
  seasons: '2020-21 to 2025-26',
  games: 7227,
  model: { accuracy: 65.4, brier: 0.2158, logLoss: 0.6205 },
  market: { accuracy: 67.9, brier: 0.2052, logLoss: 0.596 },
  disagreements: { games: 2372, model: 58.5, market: 65.9 },
  againstSpread: { games: 7166, model: 50.0 },
  marketLeadRange: '0.9 to 3.6',
};

export const SMALL_SAMPLE = 100;
