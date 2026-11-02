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

// The player model (v2) over the same walk-forward test (spec, replay result of plan 3).
export const backtestV2: null | { games: number; model: { accuracy: number; brier: number; logLoss: number } } = {
  games: 7227,
  model: { accuracy: 66.9, brier: 0.2109, logLoss: 0.6093 },
};

// Each team's own measured home edge in points against the league value, unshrunk,
// over its last 164 home and 164 away games to the end of 2025-26 (plan 3 replay).
// Measured only: the model uses one league-wide value because per-team did not score better.
export const homeEdges: { team: string; points: number }[] = [
  { team: 'OKC', points: 3.8 },
  { team: 'MIL', points: 2.1 },
  { team: 'ORL', points: 2.0 },
  { team: 'IND', points: 1.8 },
  { team: 'LAL', points: 1.8 },
  { team: 'MEM', points: 1.8 },
  { team: 'HOU', points: 1.7 },
  { team: 'DEN', points: 1.4 },
  { team: 'UTA', points: 1.2 },
  { team: 'NOP', points: 0.8 },
  { team: 'NYK', points: 0.6 },
  { team: 'LAC', points: 0.3 },
  { team: 'BKN', points: 0.0 },
  { team: 'DAL', points: -0.0 },
  { team: 'POR', points: -0.1 },
  { team: 'TOR', points: -0.3 },
  { team: 'SAS', points: -0.8 },
  { team: 'ATL', points: -1.0 },
  { team: 'PHX', points: -1.1 },
  { team: 'BOS', points: -1.2 },
  { team: 'CLE', points: -1.3 },
  { team: 'MIN', points: -1.3 },
  { team: 'CHA', points: -1.6 },
  { team: 'MIA', points: -2.0 },
  { team: 'SAC', points: -2.0 },
  { team: 'CHI', points: -2.2 },
  { team: 'GSW', points: -2.4 },
  { team: 'DET', points: -2.7 },
  { team: 'WAS', points: -3.6 },
  { team: 'PHI', points: -4.8 },
];
export const homeCourtReplay = { flat: 0.2109, perTeam: 0.2110 };

export const SMALL_SAMPLE = 100;
