// Per-game call chains (pure: built-ins only, no imports, so `node --test` can load it as is).
//
// A game now has several log lines: one early call, any number of revisions and one final call
// (`predicted`, or `not_predicted` when the final was missed). Lines written before calls existed
// have no `call_type`; each is read as a single early call.
//
//   latest call   the last predicted line: what the site shows as the game's call
//   graded call   the line the grade names (same predicted_at, the later line on a tie)

export type CallType = 'early' | 'revision' | 'final';

export type CallLine = {
  game_id: string;
  game_date: string;
  home_team: string;
  away_team: string;
  status: 'predicted' | 'not_predicted';
  p_home: number | null;
  predicted_at: string;
  call_type?: CallType;
  supersedes?: string | null;
  revision_reason?: string | null;
  model_version?: string;
};

export type GradeRef = { predicted_at: string };
export type GradeOutcome = GradeRef & { home_won: boolean; correct: boolean; early_correct?: boolean | null };

export const callTypeOf = (line: { call_type?: CallType | null }): CallType => line.call_type ?? 'early';

const isPredicted = (l: CallLine) => l.status === 'predicted' && l.p_home != null;

/** Lines grouped per game and date (a postponed game is one group per date), in log order. */
export function groupCalls<T extends CallLine>(lines: T[]): Map<string, T[]> {
  const out = new Map<string, T[]>();
  for (const l of lines) {
    const key = `${l.game_id}|${l.game_date}`;
    const list = out.get(key);
    if (list) list.push(l);
    else out.set(key, [l]);
  }
  return out;
}

/** The last predicted line, or null when the game was never called. */
export function latestCall<T extends CallLine>(calls: T[]): T | null {
  for (let i = calls.length - 1; i >= 0; i--) if (isPredicted(calls[i])) return calls[i];
  return null;
}

/** The call a grade names: the last predicted line with its predicted_at; else the latest call. */
export function gradedCall<T extends CallLine>(calls: T[], grade: GradeRef | null): T | null {
  if (grade) {
    for (let i = calls.length - 1; i >= 0; i--) {
      if (isPredicted(calls[i]) && calls[i].predicted_at === grade.predicted_at) return calls[i];
    }
  }
  return latestCall(calls);
}

const istHM = new Intl.DateTimeFormat('en-GB', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });

/** The side a line favours and its chance, as format.ts `lean` computes it: "DEN 61%". */
export function pickText(l: { home_team: string; away_team: string; p_home: number | null }): string | null {
  if (l.p_home == null) return null;
  return l.p_home >= 0.5
    ? `${l.home_team} ${Math.round(l.p_home * 100)}%`
    : `${l.away_team} ${Math.round((1 - l.p_home) * 100)}%`;
}

/** "Revised 23:45 IST: <reason>. Was DEN 61%, now LAL 52%." from the last line with a reason. */
export function revisionNote(calls: CallLine[]): string | null {
  const predicted = calls.filter(isPredicted);
  for (let i = predicted.length - 1; i >= 1; i--) {
    const cur = predicted[i];
    if (!cur.revision_reason) continue;
    const prev = predicted[i - 1];
    const at = istHM.format(new Date(cur.predicted_at));
    return `Revised ${at} IST: ${cur.revision_reason}. Was ${pickText(prev)}, now ${pickText(cur)}.`;
  }
  return null;
}

/** Was the game's early call right? The grade's own field when it has one; else the first early line. */
export function earlyCorrect(grade: GradeOutcome, calls: CallLine[]): boolean | null {
  if (grade.early_correct !== undefined) return grade.early_correct;
  const early = calls.find((l) => isPredicted(l) && callTypeOf(l) === 'early');
  if (!early) return null;
  return (early.p_home! >= 0.5) === grade.home_won;
}

/** The first game date with a model v2 line, or null while every call is v1. */
export function modelSwitchDate(lines: { game_date: string; model_version?: string | null }[]): string | null {
  let first: string | null = null;
  for (const l of lines) if (l.model_version === 'v2' && (first === null || l.game_date < first)) first = l.game_date;
  return first;
}

/** How many revision lines a game has. */
export const revisionCount = (calls: { call_type?: CallType | null }[]): number =>
  calls.filter((l) => callTypeOf(l) === 'revision').length;
