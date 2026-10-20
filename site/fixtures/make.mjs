// Writes a small, clearly fictional fixture log for QA of populated states:
// five days (2026-10-16 to 2026-10-20), four of them graded, one not_predicted line,
// one call with no market line, one call sealed after its tip-off.
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
let n = 0;
for (const [d, day] of days.entries()) {
  const pool = [...teams].sort(() => rnd() - 0.5);
  const sealIso = `${day.date}T12:${String(40 + d).padStart(2, '0')}:03+00:00`;
  const commit = fake40(day.date);
  day.tips.forEach((t, i) => {
    n++;
    const [away, home] = [pool[i * 2], pool[i * 2 + 1]];
    const nextDay = Number(t.slice(0, 2)) < 12;
    const tipDate = nextDay ? new Date(Date.parse(`${day.date}T00:00:00Z`) + 86400e3).toISOString().slice(0, 10) : day.date;
    const tip = `${tipDate}T${t}:00+00:00`;
    const game_id = `00226${String(90000 + n).padStart(5, '0')}`;
    const notPredicted = d === 2 && i === 0; // an afternoon game the run missed
    const noMarket = d === 3 && i === 1;
    const terms = {
      rating: Math.round((rnd() * 16 - 8) * 1e4) / 1e4,
      home: 1.730081,
      rest: rnd() < 0.3 ? Math.round((rnd() * 3 - 1.5) * 1e4) / 1e4 : 0,
      travel: rnd() < 0.3 ? Math.round((rnd() * 1.4 - 0.7) * 1e4) / 1e4 : 0,
      altitude: home === 'DEN' || home === 'UTA' ? 1.1 : 0,
    };
    const spread = Object.values(terms).reduce((a, b) => a + b, 0);
    const p_home = phi(spread / 12.8);
    const mSpread = -(spread + (rnd() * 6 - 3));
    const roundedM = Math.round(mSpread * 2) / 2;
    const market_p_home = phi(-roundedM / 12.8);
    const sign = (v) => (v >= 0 ? '+' : '') + v.toFixed(1);
    const winner = spread > 0 ? home : away;
    const line = {
      away_team: away, code_dirty: false, code_version: 'c45925d', game_date: day.date, game_id, home_team: home,
      market_books: noMarket || notPredicted ? null : 9,
      market_label: noMarket || notPredicted ? null : 'market line at 17:52 IST',
      market_observed_at: noMarket || notPredicted ? null : `${day.date}T12:22:00+00:00`,
      market_p_home: noMarket || notPredicted ? null : Math.round(market_p_home * 1e6) / 1e6,
      market_spread: noMarket || notPredicted ? null : roundedM,
      p_home: notPredicted ? null : Math.round(p_home * 1e6) / 1e6,
      predicted_at: `${day.date}T12:${String(40 + d).padStart(2, '0')}:01+00:00`,
      reason: notPredicted ? 'game had already started (or was within 30 minutes of tip-off) when the prediction run happened' : null,
      season: '2026-27',
      sentence: notPredicted ? null : `${away} at ${home}: rating ${sign(terms.rating)}, home ${sign(terms.home)}, rest ${sign(terms.rest)}, travel ${sign(terms.travel)}, altitude ${sign(terms.altitude)} -> ${winner} by ${Math.abs(spread).toFixed(1)} (${winner} ${Math.round((spread > 0 ? p_home : 1 - p_home) * 100)}% to win)`,
      settings: { half_life: null, hca_window: 1230, k: 0.04, margin_cap: 40.0, season_regression: 0.5, sigma: 12.8 },
      spread: notPredicted ? null : Math.round(spread * 1e6) / 1e6,
      stale_results: false,
      status: notPredicted ? 'not_predicted' : 'predicted',
      terms: notPredicted ? null : terms,
      tip_off_utc: tip,
    };
    preds.push(line);
    // One call in the fixture is committed late, to show the "after tip-off" state.
    const late = d === 1 && i === 0;
    seals[game_id] = { hash: late ? fake40(`${day.date}-late`) : commit, time: late ? new Date(Date.parse(tip) + 65 * 60e3).toISOString() : sealIso };
    if (day.graded && !notPredicted) {
      const homeWon = rnd() < p_home;
      const base = 100 + Math.floor(rnd() * 22);
      const diff = 1 + Math.floor(rnd() * 16);
      grades.push({
        game_id, home_team: home, away_team: away,
        home_points: homeWon ? base + diff : base, away_points: homeWon ? base : base + diff,
        predicted_at: line.predicted_at, p_home: line.p_home, home_won: homeWon,
        correct: homeWon === p_home >= 0.5,
        market_correct: line.market_p_home == null ? null : homeWon === line.market_p_home >= 0.5,
        graded_at: `${new Date(Date.parse(`${day.date}T00:00:00Z`) + 86400e3).toISOString().slice(0, 10)}T06:30:00+00:00`, code_version: 'c45925d',
      });
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
