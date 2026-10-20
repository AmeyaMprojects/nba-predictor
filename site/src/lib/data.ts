// Build-time data. Reads the prediction log and grade log, and finds the git commit that
// sealed each prediction line (git blame). Nothing here runs in the browser.
//
//   default:              ../predictions/<season>.jsonl and <season>-grades.jsonl
//   SITE_DATA=fixtures:   ./fixtures/predictions/*.jsonl with seals from ./fixtures/seals.json
import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';

export type Terms = { rating: number; home: number; rest: number; travel: number; altitude: number };

export type PredictionLine = {
  game_id: string;
  game_date: string;
  season?: string;
  away_team: string;
  home_team: string;
  tip_off_utc: string;
  status: 'predicted' | 'not_predicted';
  reason: string | null;
  p_home: number | null;
  spread: number | null;
  sentence: string | null;
  terms: Terms | null;
  market_p_home: number | null;
  market_spread: number | null;
  market_books: number | null;
  market_label: string | null;
  market_observed_at: string | null;
  predicted_at: string;
  stale_results?: boolean;
  code_version?: string;
  code_dirty?: boolean;
};

export type GradeLine = {
  game_id: string;
  home_team: string;
  away_team: string;
  home_points: number;
  away_points: number;
  predicted_at: string;
  p_home: number;
  home_won: boolean;
  correct: boolean;
  market_correct: boolean | null;
  graded_at: string;
  code_version?: string;
};

export type Seal = { hash: string; short: string; time: string } | null;

export type Game = PredictionLine & {
  file: string;
  line: number;
  seal: Seal;
  /** true/false once a seal exists; null while the line is not yet committed. */
  sealedBeforeTip: boolean | null;
  grade: GradeLine | null;
};

export type Tally = { right: number; of: number };

export type Day = {
  date: string;
  games: Game[];
  called: Game[];
  seal: Seal; // earliest commit among the day's calls
  unsealed: number; // calls not yet in a commit
  firstTip: string;
  lastTip: string;
  graded: number;
  model: Tally;
  home: Tally;
  market: Tally;
};

export type Totals = {
  calls: number;
  graded: number;
  model: Tally;
  home: Tally;
  market: Tally;
  days: number;
};

const FIXTURES = process.env.SITE_DATA === 'fixtures';
const SITE_DIR = process.cwd();
const PRED_DIR = FIXTURES
  ? path.resolve(SITE_DIR, 'fixtures/predictions')
  : path.resolve(process.env.PREDICTIONS_DIR ?? path.resolve(SITE_DIR, '../predictions'));

export const dataSource = FIXTURES ? 'fixtures' : 'predictions';

function readJsonl<T>(file: string): T[] {
  if (!existsSync(file)) return [];
  return readFileSync(file, 'utf8')
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => JSON.parse(l) as T);
}

function gitRoot(dir: string): string | null {
  try {
    return execFileSync('git', ['rev-parse', '--show-toplevel'], { cwd: dir, encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
}

/** line number (1-based) → seal, from `git blame --line-porcelain`. Uncommitted lines → null. */
function blame(file: string): Map<number, Seal> {
  const out = new Map<number, Seal>();
  const root = gitRoot(path.dirname(file));
  if (!root) return out;
  let text = '';
  try {
    text = execFileSync('git', ['blame', '--line-porcelain', '--', path.relative(root, file)], {
      cwd: root,
      encoding: 'utf8',
      maxBuffer: 256 * 1024 * 1024,
    });
  } catch {
    return out; // untracked file: every line is unsealed
  }
  let hash = '';
  let final = 0;
  let time = 0;
  for (const row of text.split('\n')) {
    const head = /^([0-9a-f]{40}) \d+ (\d+)/.exec(row);
    if (head) {
      hash = head[1];
      final = Number(head[2]);
      continue;
    }
    if (row.startsWith('committer-time ')) time = Number(row.slice(15));
    else if (row.startsWith('\t')) {
      const committed = !/^0+$/.test(hash);
      out.set(final, committed ? { hash, short: hash.slice(0, 7), time: new Date(time * 1000).toISOString() } : null);
    }
  }
  return out;
}

function fixtureSeals(): Record<string, { hash: string; time: string }> {
  const f = path.resolve(SITE_DIR, 'fixtures/seals.json');
  return existsSync(f) ? JSON.parse(readFileSync(f, 'utf8')) : {};
}

const tally = (games: Game[], pick: (g: GradeLine) => boolean | null): Tally => {
  let right = 0;
  let of = 0;
  for (const g of games) {
    if (!g.grade) continue;
    const v = pick(g.grade);
    if (v === null || v === undefined) continue;
    of++;
    if (v) right++;
  }
  return { right, of };
};

function load() {
  const files = existsSync(PRED_DIR) ? readdirSync(PRED_DIR).filter((f) => f.endsWith('.jsonl')) : [];
  const predFiles = files.filter((f) => !f.endsWith('-grades.jsonl')).sort();
  const gradeFiles = files.filter((f) => f.endsWith('-grades.jsonl')).sort();

  const grades = new Map<string, GradeLine>();
  for (const f of gradeFiles) for (const g of readJsonl<GradeLine>(path.join(PRED_DIR, f))) grades.set(g.game_id, g);

  const seals = FIXTURES ? fixtureSeals() : null;
  // One entry per (game, date): a later line for the same game and date (a call made after a
  // "tip-off time not announced" line) replaces the earlier one. A postponed game keeps both dates.
  const byGame = new Map<string, Game>();
  for (const f of predFiles) {
    const full = path.join(PRED_DIR, f);
    const lines = readJsonl<PredictionLine>(full);
    const blamed = FIXTURES ? null : blame(full);
    lines.forEach((p, i) => {
      const fx = seals?.[p.game_id];
      const seal: Seal = FIXTURES
        ? fx ? { hash: fx.hash, short: fx.hash.slice(0, 7), time: fx.time } : null
        : blamed?.get(i + 1) ?? null;
      byGame.set(`${p.game_id}|${p.game_date}`, {
        ...p,
        file: `predictions/${f}`,
        line: i + 1,
        seal,
        sealedBeforeTip: seal ? Date.parse(seal.time) < Date.parse(p.tip_off_utc) : null,
        grade: null,
      });
    });
  }
  // A grade belongs to the call it names (same predicted_at); failing that, the latest call of that game.
  const calls = [...byGame.values()].filter((g) => g.status === 'predicted');
  for (const [id, grade] of grades) {
    const mine = calls.filter((g) => g.game_id === id);
    const exact = mine.find((g) => g.predicted_at === grade.predicted_at);
    const target = exact ?? mine.sort((a, b) => a.game_date.localeCompare(b.game_date)).at(-1);
    if (target) target.grade = grade;
  }

  const all = [...byGame.values()].sort(
    (a, b) => a.game_date.localeCompare(b.game_date) || a.tip_off_utc.localeCompare(b.tip_off_utc) || a.game_id.localeCompare(b.game_id),
  );

  const dayMap = new Map<string, Game[]>();
  for (const g of all) {
    if (!dayMap.has(g.game_date)) dayMap.set(g.game_date, []);
    dayMap.get(g.game_date)!.push(g);
  }

  const days: Day[] = [...dayMap.entries()].map(([date, games]) => {
    const called = games.filter((g) => g.status === 'predicted');
    const sealed = called.map((g) => g.seal).filter((s): s is NonNullable<Seal> => !!s);
    sealed.sort((a, b) => a.time.localeCompare(b.time));
    const tips = games.map((g) => g.tip_off_utc).sort();
    return {
      date,
      games,
      called,
      seal: sealed[0] ?? null,
      unsealed: called.length - sealed.length,
      firstTip: tips[0],
      lastTip: tips[tips.length - 1],
      graded: called.filter((g) => g.grade).length,
      model: tally(called, (r) => r.correct),
      home: tally(called, (r) => r.home_won),
      market: tally(called, (r) => r.market_correct),
    };
  });

  const called = all.filter((g) => g.status === 'predicted');
  const record: Totals = {
    calls: called.length,
    graded: called.filter((g) => g.grade).length,
    model: tally(called, (r) => r.correct),
    home: tally(called, (r) => r.home_won),
    market: tally(called, (r) => r.market_correct),
    days: days.length,
  };

  return { games: all, days, record };
}

let cache: ReturnType<typeof load> | null = null;
export function getData() {
  if (!cache) cache = load();
  return cache;
}

/** The slate the home page leads with: the most recent day in the log. */
export function latestDay() {
  const { days } = getData();
  return days[days.length - 1] ?? null;
}
