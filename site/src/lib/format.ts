// Display rules (DESIGN.md, Data):
// - spreads only ever as "TEAM by X.X"; never a raw signed number
// - model spread: home margin (+ = home favoured)
// - market_spread: betting home line (− = home favoured), so the market's home margin is −market_spread
// - times in US Eastern first, India time second
import type { Game, Tally, Terms } from './data';

const ET = 'America/New_York';
const IST = 'Asia/Kolkata';

const fmt = (tz: string, opts: Intl.DateTimeFormatOptions) => new Intl.DateTimeFormat('en-US', { timeZone: tz, ...opts });
const etTime = fmt(ET, { hour: 'numeric', minute: '2-digit', hour12: true });
const istTime = fmt(IST, { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
const istDay = fmt(IST, { weekday: 'short', day: 'numeric', month: 'short' });

export const timeET = (iso: string) => `${etTime.format(new Date(iso)).replace(' AM', ' am').replace(' PM', ' pm')} ET`;
export const timeIST = (iso: string) => `${istTime.format(new Date(iso))} IST`;
export const dayIST = (iso: string) => istDay.format(new Date(iso));
const etDay = fmt(ET, { weekday: 'short', day: 'numeric', month: 'short' });
export const dayET = (iso: string) => etDay.format(new Date(iso)).replace(',', '');
/** "8:40 am ET" split for display: { hm: '8:40', ampm: 'am' } */
export function partsET(iso: string) {
  const [hm, ap] = etTime.format(new Date(iso)).split(' ');
  return { hm, ampm: ap.toLowerCase() };
}
/** "market line at 8:22 am ET" from the observation time, ET-first. */
export const marketWhen = (iso: string | null) => (iso ? `market line at ${timeET(iso)}` : 'market line');

/** game_date is the league's (Eastern) calendar date: format it without shifting time zones. */
export function gameDate(date: string, style: 'long' | 'short' | 'weekday' = 'long') {
  const d = new Date(`${date}T12:00:00Z`);
  const o: Intl.DateTimeFormatOptions =
    style === 'long' ? { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' }
    : style === 'weekday' ? { weekday: 'long' }
    : { day: 'numeric', month: 'short' };
  return new Intl.DateTimeFormat('en-GB', { timeZone: 'UTC', ...o }).format(d).replace(',', '');
}
export const gameDayNum = (date: string) => String(Number(date.slice(8, 10)));
export const gameMonth = (date: string) =>
  new Intl.DateTimeFormat('en-GB', { timeZone: 'UTC', month: 'long' }).format(new Date(`${date}T12:00:00Z`));

/** "6h 20m" (always positive; the caller says before/after). */
export function duration(ms: number) {
  const m = Math.round(Math.abs(ms) / 60000);
  const h = Math.floor(m / 60);
  const r = m % 60;
  if (h >= 48) return `${Math.round(h / 24)} days`;
  return h ? `${h}h ${String(r).padStart(2, '0')}m` : `${r}m`;
}

const margin = (x: number) => Math.abs(x).toFixed(1);

/** Model call as "DET by 1.5". */
export function modelSpread(g: Game) {
  if (g.spread == null) return null;
  if (Math.abs(g.spread) < 0.05) return 'even';
  return `${g.spread > 0 ? g.home_team : g.away_team} by ${margin(g.spread)}`;
}
/** Market line as "BOS by 1.5" (home margin is −market_spread). */
export function marketSpread(g: Game) {
  if (g.market_spread == null) return null;
  const m = -g.market_spread;
  if (Math.abs(m) < 0.05) return 'even';
  return `${m > 0 ? g.home_team : g.away_team} by ${margin(m)}`;
}

/** The side a probability favours, and its chance, e.g. { team: 'DET', pct: 55 }. */
export function lean(g: Game, p: number | null) {
  if (p == null) return null;
  return p >= 0.5 ? { team: g.home_team, pct: Math.round(p * 100) } : { team: g.away_team, pct: Math.round((1 - p) * 100) };
}
export const modelPick = (g: Game) => lean(g, g.p_home);
export const marketPick = (g: Game) => lean(g, g.market_p_home);

export const disagree = (g: Game) => {
  const a = modelPick(g);
  const b = marketPick(g);
  return !!a && !!b && a.team !== b.team;
};

export function winner(g: Game) {
  if (!g.grade) return null;
  return g.grade.home_won ? g.home_team : g.away_team;
}

export const pct = (t: Tally) => (t.of ? Math.round((t.right / t.of) * 1000) / 10 : null);
export const pctText = (t: Tally) => (t.of ? `${pct(t)!.toFixed(1)}%` : 'no games yet');

/** Term → "NYK 7.2" | "even". Positive terms favour the home team. */
export function termText(g: Game, v: number) {
  if (Math.abs(v) < 0.05) return 'even';
  return `${v > 0 ? g.home_team : g.away_team} ${margin(v)}`;
}
export const TERM_LABELS: [keyof Terms, string][] = [
  ['rating', 'Team rating'],
  ['home', 'Home court'],
  ['rest', 'Rest'],
  ['travel', 'Travel'],
  ['altitude', 'Altitude'],
];

export const nf = (n: number) => n.toLocaleString('en-US');
