// Build-time data. Reads the prediction log and grade log, and finds the git commit that
// sealed each prediction line (git blame). Nothing here runs in the browser.
//
//   default:              ../predictions/<season>.jsonl and <season>-grades.jsonl
//   SITE_DATA=fixtures:   ./fixtures/predictions/*.jsonl with seals from ./fixtures/seals.json
import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { callTypeOf, earlyCorrect, gradedCall, groupCalls, latestCall, modelSwitchDate, revisionCount, revisionNote, type CallType } from './calls';

/** Points terms. v1 lines carry the first five; v2 lines add roster and injuries. */
export type Terms = { rating?: number; roster?: number; home?: number; rest?: number; travel?: number; altitude?: number; injuries?: number };

export type Absence = { team: string; player: string; status: string };

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
  // Calls (plan 3). Absent on older lines, which read as a single early call.
  call_type?: CallType;
  supersedes?: string | null;
  revision_reason?: string | null;
  injury_report_at?: string | null;
  availability?: Absence[] | null;
  model_version?: string;
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
  linked_game_id?: string | null;
  // Calls (plan 3): the graded call's type and the game's early call. Absent on older grade lines.
  call_type?: CallType;
  early_predicted_at?: string | null;
  early_p_home?: number | null;
  early_correct?: boolean | null;
};

export type Seal = { hash: string; short: string; time: string } | null;

/** One log line with where it sits and the commit that sealed it. */
export type CallEntry = PredictionLine & {
  file: string;
  line: number;
  seal: Seal;
  /** true/false once a seal exists; null while the line is not yet committed. */
  sealedBeforeTip: boolean | null;
};

/**
 * One game on one date: the fields of its latest predicted call (or of its last line when it was
 * never called), plus every line of its chain. `seal` is the graded call's seal, else the latest's.
 */
export type Game = CallEntry & {
  grade: GradeLine | null;
  calls: CallEntry[];
  /** The call the grade names (the last call sealed before tip-off); null until graded. */
  graded: CallEntry | null;
  revisions: number;
  note: string | null;
  /** Was the early call right? null until graded, or with no predicted early call. */
  early: boolean | null;
};

export type Tally = { right: number; of: number };

export type Day = {
  date: string;
  games: Game[];
  called: Game[];
  seal: Seal; // earliest commit among the day's early calls
  writtenAt: string | null; // when the slate's first early call was written
  unsealed: number; // calls not yet in a commit
  revisions: number;
  firstTip: string;
  lastTip: string;
  graded: number;
  model: Tally;
  home: Tally;
  market: Tally;
  early: Tally;
};

export type Totals = {
  calls: number;
  graded: number;
  /** The graded calls (the last call sealed before tip-off). */
  model: Tally;
  home: Tally;
  market: Tally;
  /** The early calls (18:00 IST) of the same graded games. */
  early: Tally;
  v1: Tally;
  v2: Tally;
  /** First game date with a model v2 call; null while every call is v1. */
  modelSwitch: string | null;
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

const tally = (games: Game[], pick: (r: GradeLine, g: Game) => boolean | null | undefined): Tally => {
  let right = 0;
  let of = 0;
  for (const g of games) {
    if (!g.grade) continue;
    const v = pick(g.grade, g);
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
  const lines: CallEntry[] = [];
  for (const f of predFiles) {
    const full = path.join(PRED_DIR, f);
    const blamed = FIXTURES ? null : blame(full);
    readJsonl<PredictionLine>(full).forEach((p, i) => {
      const fx = seals?.[`${p.game_id}|${p.predicted_at}`];
      const seal: Seal = FIXTURES
        ? fx ? { hash: fx.hash, short: fx.hash.slice(0, 7), time: fx.time } : null
        : blamed?.get(i + 1) ?? null;
      lines.push({
        ...p,
        file: `predictions/${f}`,
        line: i + 1,
        seal,
        sealedBeforeTip: seal ? Date.parse(seal.time) < Date.parse(p.tip_off_utc) : null,
      });
    });
  }

  // One game per (game, date), with every line of its chain in log order. A "tip-off time not
  // announced" line followed by a call reads as that call; a postponed game keeps both dates.
  const byGame = new Map<string, Game>();
  for (const [key, calls] of groupCalls(lines)) {
    const shown = latestCall(calls) ?? calls[calls.length - 1];
    byGame.set(key, { ...shown, grade: null, calls, graded: null, revisions: revisionCount(calls), note: revisionNote(calls), early: null });
  }
  // A grade belongs to the game whose chain holds the call it names (same predicted_at);
  // failing that, the latest called date of that game.
  const called0 = [...byGame.values()].filter((g) => g.status === 'predicted');
  for (const [id, grade] of grades) {
    const mine = called0.filter((g) => g.game_id === id);
    const exact = mine.find((g) => g.calls.some((c) => c.status === 'predicted' && c.predicted_at === grade.predicted_at));
    const target = exact ?? mine.sort((a, b) => a.game_date.localeCompare(b.game_date)).at(-1);
    if (!target) continue;
    target.grade = grade;
    target.graded = gradedCall(target.calls, grade);
    target.early = earlyCorrect(grade, target.calls);
    const sealedBy = target.graded ?? target;
    target.seal = sealedBy.seal;
    target.sealedBeforeTip = sealedBy.sealedBeforeTip;
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
    const earlySeals = called
      .flatMap((g) => g.calls)
      .filter((c) => c.status === 'predicted' && callTypeOf(c) === 'early')
      .map((c) => c.seal)
      .filter((s): s is NonNullable<Seal> => !!s);
    earlySeals.sort((a, b) => a.time.localeCompare(b.time));
    const earlyTimes = called
      .flatMap((g) => g.calls)
      .filter((c) => c.status === 'predicted' && callTypeOf(c) === 'early')
      .map((c) => c.predicted_at)
      .sort((a, b) => Date.parse(a) - Date.parse(b));
    const tips = games.map((g) => g.tip_off_utc).sort();
    return {
      date,
      games,
      called,
      seal: earlySeals[0] ?? null,
      writtenAt: earlyTimes[0] ?? null,
      unsealed: called.filter((g) => !g.seal).length,
      revisions: games.reduce((n, g) => n + g.revisions, 0),
      firstTip: tips[0],
      lastTip: tips[tips.length - 1],
      graded: called.filter((g) => g.grade).length,
      model: tally(called, (r) => r.correct),
      home: tally(called, (r) => r.home_won),
      market: tally(called, (r) => r.market_correct),
      early: tally(called, (_, g) => g.early),
    };
  });

  const called = all.filter((g) => g.status === 'predicted');
  const record: Totals = {
    calls: called.length,
    graded: called.filter((g) => g.grade).length,
    model: tally(called, (r) => r.correct),
    home: tally(called, (r) => r.home_won),
    market: tally(called, (r) => r.market_correct),
    early: tally(called, (_, g) => g.early),
    v1: tally(called, (r, g) => ((g.graded ?? g).model_version ?? 'v1') === 'v1' ? r.correct : null),
    v2: tally(called, (r, g) => (g.graded ?? g).model_version === 'v2' ? r.correct : null),
    modelSwitch: modelSwitchDate(lines),
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
