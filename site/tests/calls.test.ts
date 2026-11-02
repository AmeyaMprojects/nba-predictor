// Pure call logic: grouping, the latest and graded call, the revision note, early-call grading
// and the model-switch date. Run with `npm test` (node --test, built-in type stripping).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  callTypeOf,
  earlyCorrect,
  isStale,
  modelBoundaryDay,
  gradedCall,
  hasEarlyEra,
  groupCalls,
  latestCall,
  modelSwitchDate,
  revisionCount,
  revisionNote,
  type CallLine,
} from '../src/lib/calls.ts';

const base = {
  game_id: '0022600201',
  game_date: '2026-11-05',
  away_team: 'LAL',
  home_team: 'DEN',
  status: 'predicted' as const,
};
const line = (over: Partial<CallLine>): CallLine => ({
  ...base,
  p_home: 0.6,
  predicted_at: '2026-11-05T12:37:00+00:00',
  ...over,
});

const early = line({ call_type: 'early', p_home: 0.61, predicted_at: '2026-11-05T12:37:00+00:00', model_version: 'v2' });
const revision = line({
  call_type: 'revision',
  p_home: 0.48,
  predicted_at: '2026-11-05T18:15:00+00:00',
  revision_reason: 'new injury report 23:30 IST: Murray (DEN) out',
  supersedes: '2026-11-05T12:37:00+00:00',
  model_version: 'v2',
});
const final = line({ call_type: 'final', p_home: 0.48, predicted_at: '2026-11-05T23:52:00+00:00', model_version: 'v2', revision_reason: null });

test('groups three calls of one game in log order', () => {
  const other = line({ game_id: '0022600202', home_team: 'BOS', away_team: 'NYK' });
  const groups = groupCalls([early, other, revision, final]);
  assert.equal(groups.size, 2);
  assert.deepEqual(groups.get('0022600201|2026-11-05'), [early, revision, final]);
  assert.deepEqual(groups.get('0022600202|2026-11-05'), [other]);
});

test('a postponed game is grouped once per date', () => {
  const moved = line({ game_date: '2026-11-06', predicted_at: '2026-11-06T12:37:00+00:00' });
  const groups = groupCalls([early, moved]);
  assert.equal(groups.size, 2);
});

test('callTypeOf reads an old line as early', () => {
  assert.equal(callTypeOf({ ...base, p_home: 0.5, predicted_at: 'x' }), 'early');
  assert.equal(callTypeOf(final), 'final');
});

test('latestCall is the last predicted line; null when nothing was predicted', () => {
  const missed = line({ call_type: 'final', status: 'not_predicted', p_home: null, predicted_at: '2026-11-06T00:10:00+00:00' });
  assert.equal(latestCall([early, revision, missed]), revision);
  assert.equal(latestCall([missed]), null);
});

test('revisionNote', () => {
  assert.equal(
    revisionNote([early, revision]),
    'Revised 23:45 IST: new injury report 23:30 IST: Murray (DEN) out. Was DEN 61%, now LAL 52%.',
  );
});

test('revisionNote survives an unchanged final', () => {
  assert.equal(
    revisionNote([early, revision, final]),
    'Revised 23:45 IST: new injury report 23:30 IST: Murray (DEN) out. Was DEN 61%, now LAL 52%.',
  );
});

test('revisionNote is null without a revision', () => {
  assert.equal(revisionNote([early]), null);
  assert.equal(revisionNote([early, line({ call_type: 'final', p_home: 0.61 })]), null);
});

test('gradedCall picks the call the grade names', () => {
  const grade = { predicted_at: revision.predicted_at };
  assert.equal(gradedCall([early, revision, final], grade), revision);
  // A tie on predicted_at goes to the later line.
  const sameRun = line({ call_type: 'final', p_home: 0.61, predicted_at: early.predicted_at });
  assert.equal(gradedCall([early, sameRun], { predicted_at: early.predicted_at }), sameRun);
  // A grade naming no logged call falls back to the latest call.
  assert.equal(gradedCall([early, revision], { predicted_at: 'nope' }), revision);
  assert.equal(gradedCall([early, revision], null), revision);
});

test('earlyCorrect falls back for old grade lines', () => {
  const old = line({ p_home: 0.7, model_version: undefined });
  delete (old as Partial<CallLine>).call_type;
  for (const home_won of [true, false]) {
    const grade = { predicted_at: old.predicted_at, home_won, correct: home_won === true };
    assert.equal(earlyCorrect(grade, [old]), grade.correct);
  }
  // The grade's own field wins when present, null included.
  assert.equal(earlyCorrect({ predicted_at: 'x', home_won: true, correct: true, early_correct: false }, [old]), false);
  assert.equal(earlyCorrect({ predicted_at: 'x', home_won: true, correct: true, early_correct: null }, [old]), null);
  // No predicted early call and no field: null.
  const tbd = line({ status: 'not_predicted', p_home: null });
  assert.equal(earlyCorrect({ predicted_at: 'x', home_won: true, correct: true }, [tbd]), null);
});

test('modelSwitchDate', () => {
  const v1 = line({ game_date: '2026-11-08', model_version: 'v1' });
  const v2 = line({ game_date: '2026-11-10', model_version: 'v2' });
  const v2b = line({ game_date: '2026-11-12', model_version: 'v2' });
  assert.equal(modelSwitchDate([v1, v2b, v2]), '2026-11-10');
  assert.equal(modelSwitchDate([v1, line({ game_date: '2026-11-09' })]), null);
});

test('modelBoundaryDay is the first v2 day, when an older day exists', () => {
  const days = ['2026-11-12', '2026-11-10', '2026-11-08'];
  assert.equal(modelBoundaryDay(days, '2026-11-10'), '2026-11-10');
  // No day on the switch date itself: the oldest v2 day still carries the boundary.
  assert.equal(modelBoundaryDay(days, '2026-11-09'), '2026-11-10');
  // Nothing to divide.
  assert.equal(modelBoundaryDay(days, '2026-11-08'), null);
  assert.equal(modelBoundaryDay(days, '2026-11-20'), null);
  assert.equal(modelBoundaryDay(days, null), null);
});

test('isStale only after 36 hours past tip-off', () => {
  const tip = Date.parse('2026-11-05T00:00:00Z');
  const hour = 3600e3;
  assert.equal(isStale(tip, tip - hour), false);
  assert.equal(isStale(tip, tip + 36 * hour), false);
  assert.equal(isStale(tip, tip + 36 * hour + 1), true);
});

test('revisionCount counts only revisions', () => {
  assert.equal(revisionCount([early, revision, { ...revision, predicted_at: '2026-11-05T20:00:00+00:00' }, final]), 2);
  assert.equal(revisionCount([early]), 0);
});

test('hasEarlyEra: only grades with a call_type count, not legacy-only records', () => {
  assert.equal(hasEarlyEra([]), false);
  assert.equal(hasEarlyEra([{}, {}]), false);
  assert.equal(hasEarlyEra([{}, { call_type: 'final' }]), true);
  assert.equal(hasEarlyEra([{ call_type: 'early' }]), true);
});
