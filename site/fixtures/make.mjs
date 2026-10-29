// Writes a small, clearly fictional fixture log for QA of populated states:
// five days (2026-10-16 to 2026-10-20), four of them graded, one not_predicted line,
// one call with no market line, one call sealed after its tip-off.
// Calls (plan 3):
//   2026-10-16, -17   legacy lines: no call_type, one line per game
//   2026-10-18        model v1, an early and a final call per game (same p)
//   2026-10-19 on     model v2 (the "Model v2 from" marker): one game with early + revision + final,
//                     one with a missed final, one postponed to 2026-10-20 (fresh early call there)
//   2026-10-20        ungraded: a revision before the final, a provisional espn: game with early +
//                     final, and a game whose tip-off was not announced (TBD)
// Seals are keyed `${game_id}|${predicted_at}`.
//   node fixtures/make.mjs     then     SITE_DATA=fixtures npm run build
import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = path.dirname(fileURLToPath(import.meta.url));
let seed = 7;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const teams = ['ATL','BOS','BKN','CHA','CHI','CLE','DAL','DEN','DET','GSW','HOU','IND','LAC','LAL','MEM','MIA','MIL','MIN','NOP','NYK','OKC','ORL','PHI','PHX','POR','SAC','SAS','TOR','UTA','WAS'];
const phi = (x) => 0.5 * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x ** 3))); // normal CDF approx
const fake40 = (s) => createHash('sha1').update(`fixture:${s}`).digest('hex');

const days = [
  { date: '2026-10-16', tips: ['23:00', '23:30', '00:00', '02:00', '02:30'], graded: true },
  { date: '2026-10-17', tips: ['23:00', '00:30', '02:00'], graded: true },
  { date: '2026-10-18', tips: ['19:30', '23:00', '23:00', '00:00', '01:00', '03:00'], graded: true },
  { date: '2026-10-19', tips: ['23:30', '00:00', '02:00', '02:30'], graded: true },
  { date: '2026-10-20', tips: ['23:00', '23:30', '00:00', '02:00'], graded: false },
];

const preds = [];
const grades = [];
const seals = {};
const MISSED = 'no final call was made in the 45 minutes before tip-off (no run reached it in time)';
const TBD = 'tip-off time not announced when the prediction run happened';
const iso = (ms) => new Date(ms).toISOString().replace('.000Z', '+00:00');
const istHM = (ms) => new Date(ms + 330 * 60e3).toISOString().slice(11, 16);
const seal = (l, key, ms) => { seals[`${l.game_id}|${l.predicted_at}`] = { hash: fake40(key), time: iso(ms) }; };
const nudge = (p, dp) => Math.round(Math.min(0.97, Math.max(0.03, p + dp)) * 1e6) / 1e6;
let postponed = null;
let n = 0;
for (const [d, day] of days.entries()) {
  const pool = [...teams].sort(() => rnd() - 0.5);
  const sealIso = `${day.date}T12:${String(40 + d).padStart(2, '0')}:03+00:00`;
  const commit = fake40(day.date);
  const legacy = d < 2;
  const version = d >= 3 ? 'v2' : 'v1';
  const tips = [...day.tips];
  if (d === 3) tips.push('01:30'); // the game postponed to the next day
  if (d === 4) tips.push('03:30', '04:00'); // provisional espn: game, TBD game
  tips.forEach((t, i) => {
    n++;
    let [away, home] = [pool[i * 2], pool[i * 2 + 1]];
    const nextDay = Number(t.slice(0, 2)) < 12;
    const tipDate = nextDay ? new Date(Date.parse(`${day.date}T00:00:00Z`) + 86400e3).toISOString().slice(0, 10) : day.date;
    const tip = `${tipDate}T${t}:00+00:00`;
    const tipMs = Date.parse(tip);
    let game_id = `00226${String(90000 + n).padStart(5, '0')}`;
    const isPostponedOrigin = d === 3 && i === day.tips.length;
    const isPostponedNew = d === 4 && i === 0 && postponed;
    if (isPostponedNew) ({ game_id, away, home } = postponed);
    const isEspn = d === 4 && i === day.tips.length;
    const isTbd = d === 4 && i === day.tips.length + 1;
    if (isEspn) game_id = 'espn:401900001';
    const notPredicted = (d === 2 && i === 0) || isTbd; // an afternoon game the run missed; a TBD tip-off
    const noMarket = d === 3 && i === 1;
    const terms = {
      rating: Math.round((rnd() * 16 - 8) * 1e4) / 1e4,
      home: 1.730081,
      rest: rnd() < 0.3 ? Math.round((rnd() * 3 - 1.5) * 1e4) / 1e4 : 0,
      travel: rnd() < 0.3 ? Math.round((rnd() * 1.4 - 0.7) * 1e4) / 1e4 : 0,
      altitude: home === 'DEN' || home === 'UTA' ? 1.1 : 0,
    };
    if (version === 'v2') {
      terms.roster = Math.round((rnd() * 4 - 2) * 1e4) / 1e4;
      terms.injuries = rnd() < 0.4 ? Math.round((rnd() * 5 - 2.5) * 1e4) / 1e4 : 0;
    }
    const spread = Object.values(terms).reduce((a, b) => a + b, 0);
    const p_home = phi(spread / 12.8);
    const mSpread = -(spread + (rnd() * 6 - 3));
    const roundedM = Math.round(mSpread * 2) / 2;
    const market_p_home = phi(-roundedM / 12.8);
    const sign = (v) => (v >= 0 ? '+' : '') + v.toFixed(1);
    const winner = spread > 0 ? home : away;
    const order = version === 'v2' ? ['rating', 'roster', 'home', 'rest', 'travel', 'altitude', 'injuries'] : ['rating', 'home', 'rest', 'travel', 'altitude'];
    const line = {
      away_team: away, code_dirty: false, code_version: 'c45925d', game_date: day.date, game_id, home_team: home,
      market_books: noMarket || notPredicted ? null : 9,
      market_label: noMarket || notPredicted ? null : 'market line at 17:52 IST',
      market_observed_at: noMarket || notPredicted ? null : `${day.date}T12:22:00+00:00`,
      market_p_home: noMarket || notPredicted ? null : Math.round(market_p_home * 1e6) / 1e6,
      market_spread: noMarket || notPredicted ? null : roundedM,
      p_home: notPredicted ? null : Math.round(p_home * 1e6) / 1e6,
      predicted_at: `${day.date}T12:${String(40 + d).padStart(2, '0')}:01+00:00`,
      reason: isTbd ? TBD : notPredicted ? 'game had already started (or was within 30 minutes of tip-off) when the prediction run happened' : null,
      season: '2026-27',
      sentence: notPredicted ? null : `${away} at ${home}: ${order.map((k) => `${k} ${sign(terms[k])}`).join(', ')} -> ${winner} by ${Math.abs(spread).toFixed(1)} (${winner} ${Math.round((spread > 0 ? p_home : 1 - p_home) * 100)}% to win)`,
      settings: { half_life: null, hca_window: 1230, k: 0.04, margin_cap: 40.0, season_regression: 0.5, sigma: 12.8 },
      spread: notPredicted ? null : Math.round(spread * 1e6) / 1e6,
      stale_results: false,
      status: notPredicted ? 'not_predicted' : 'predicted',
      terms: notPredicted ? null : terms,
      tip_off_utc: tip,
    };
    if (!legacy) {
      Object.assign(line, {
        call_type: 'early', supersedes: null, revision_reason: null, model_version: version,
        injury_report_at: version === 'v2' && !notPredicted ? `${day.date}T12:30:00+00:00` : null,
        availability: version === 'v2' && !notPredicted ? [] : null,
      });
    }
    if (isPostponedOrigin) postponed = { game_id, away, home };
    preds.push(line);
    // One call in the fixture is committed late, to show the "after tip-off" state.
    const late = d === 1 && i === 0;
    seals[`${game_id}|${line.predicted_at}`] = { hash: late ? fake40(`${day.date}-late`) : commit, time: late ? new Date(tipMs + 65 * 60e3).toISOString() : sealIso };
    if (notPredicted) return;

    // Later calls of new-style chains.
    const chain = [line];
    const later = (over) => {
      const prev = chain.filter((c) => c.status === 'predicted').at(-1);
      const l = { ...line, supersedes: prev.predicted_at, ...over };
      chain.push(l);
      preds.push(l);
      seal(l, `${game_id}-${l.call_type}-${l.predicted_at}`, Date.parse(l.predicted_at) + 2 * 60e3);
      return l;
    };
    const finalAt = iso(tipMs - 40 * 60e3);
    const revise = (atMs, dp, who) => {
      const reportMs = atMs - 15 * 60e3;
      return later({
        call_type: 'revision', p_home: nudge(line.p_home, dp), predicted_at: iso(atMs),
        revision_reason: `new injury report ${istHM(reportMs)} IST: ${who} (${home}) out`,
        injury_report_at: iso(reportMs), availability: [{ team: home, player: who, status: 'Out' }],
      });
    };
    const withRevision = (d === 3 && i === 0) || (d === 4 && i === 1);
    const missedFinal = d === 3 && i === 2;
    if (!legacy && !isPostponedOrigin) {
      if (withRevision) {
        const rev = revise(Date.parse(`${day.date}T18:15:00Z`), line.p_home >= 0.5 ? -0.13 : 0.13, d === 3 ? 'Okafor' : 'Lindqvist');
        if (d === 3) later({ call_type: 'final', p_home: rev.p_home, predicted_at: finalAt, revision_reason: null, supersedes: rev.predicted_at });
      } else if (missedFinal) {
        later({
          call_type: 'final', status: 'not_predicted', reason: MISSED, p_home: null, spread: null, sentence: null, terms: null,
          market_books: null, market_label: null, market_observed_at: null, market_p_home: null, market_spread: null,
          predicted_at: iso(tipMs - 10 * 60e3), availability: null, injury_report_at: null,
        });
      } else if (day.graded || isEspn) {
        later({ call_type: 'final', predicted_at: finalAt }); // same p as the early call
      }
    }

    if (day.graded && !isPostponedOrigin) {
      const graded = chain.filter((c) => c.status === 'predicted' && Date.parse(c.predicted_at) < tipMs).at(-1);
      const homeWon = rnd() < graded.p_home;
      const base = 100 + Math.floor(rnd() * 22);
      const diff = 1 + Math.floor(rnd() * 16);
      const grade = {
        game_id, home_team: home, away_team: away,
        home_points: homeWon ? base + diff : base, away_points: homeWon ? base : base + diff,
        predicted_at: graded.predicted_at, p_home: graded.p_home, home_won: homeWon,
        correct: homeWon === graded.p_home >= 0.5,
        market_correct: graded.market_p_home == null ? null : homeWon === graded.market_p_home >= 0.5,
        graded_at: `${new Date(Date.parse(`${day.date}T00:00:00Z`) + 86400e3).toISOString().slice(0, 10)}T06:30:00+00:00`, code_version: 'c45925d',
      };
      if (!legacy) {
        Object.assign(grade, {
          linked_game_id: null, call_type: graded.call_type,
          early_predicted_at: line.predicted_at, early_p_home: line.p_home, early_correct: homeWon === line.p_home >= 0.5,
        });
      }
      grades.push(grade);
    }
  });
}
mkdirSync(path.join(dir, 'predictions'), { recursive: true });
const sortTop = (r) => Object.fromEntries(Object.keys(r).sort().map((k) => [k, r[k]]));
const jsonl = (rows) => rows.map((r) => JSON.stringify(sortTop(r))).join('\n') + '\n';
writeFileSync(path.join(dir, 'predictions/2026-27.jsonl'), jsonl(preds));
writeFileSync(path.join(dir, 'predictions/2026-27-grades.jsonl'), jsonl(grades));
writeFileSync(path.join(dir, 'seals.json'), JSON.stringify(seals, null, 2) + '\n');
console.log(`${preds.length} prediction lines, ${grades.length} grade lines`);
