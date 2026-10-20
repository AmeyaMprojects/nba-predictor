// The concept's driver: ONE variable, the real clock, mapped onto every visual that carries
// "sealed, then tip-off, then final". It moves the now-marker along the rail, fills the elapsed
// part of the rail, marks passed tips, ticks the countdown and each game's status.
// Entrances: the stamp presses (back.out 1.4, 0.35 s); markers slide along their rails (expo.out, 0.9 s).
// Reduced motion: no tweens; every marker sits at its true position.
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { ease, onPage, prefersReducedMotion } from '../engine';

gsap.registerPlugin(ScrollTrigger);

const HOUR = 3600e3;
const FINAL_AFTER = 2.5 * HOUR;
const istClock = new Intl.DateTimeFormat('en-GB', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
const etClock = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit' });
const pad = (n: number) => String(n).padStart(2, '0');

function hm(ms: number) {
  const m = Math.max(0, Math.round(ms / 60000));
  const h = Math.floor(m / 60);
  if (h >= 48) return `${Math.round(h / 24)} days`;
  return h ? `${h}h ${pad(m % 60)}m` : `${m}m`;
}
const ist = (ms: number) => `${istClock.format(new Date(ms))} IST`;
function clockText(ms: number) {
  const s = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(s / 3600);
  if (h >= 48) return `${Math.round(h / 24)} days`;
  return `${h}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`;
}
const et = (ms: number) => `${etClock.format(new Date(ms)).replace(' AM', ' am').replace(' PM', ' pm')} ET`;

export function startSeal() {
  onPage(({ intro }) => {
    const reduced = prefersReducedMotion();
    const off: (() => void)[] = [];

    // --- the stamp press -------------------------------------------------------------------
    document.querySelectorAll<HTMLElement>('[data-stamp]').forEach((s) => {
      if (reduced) return void s.classList.add('is-pressed');
      // The stamp-down: tilted and lifted (CSS) to its resting -2deg, shadow collapsing to 1px.
      // Tweens CSS variables, so transform stays the only thing that changes (no opacity: the text
      // is visible and the LCP from the first paint).
      const press = gsap.fromTo(s,
        { '--st-r': '-8deg', '--st-s': 1.12, '--st-sh': '9px' },
        { '--st-r': '-2deg', '--st-s': 1, '--st-sh': '1px', duration: 0.45, ease: ease.stamp, paused: true,
          onComplete: () => s.classList.add('is-pressed') });
      intro.add(() => press.play(), 0.1);
    });

    // --- the rail and its now-marker (the one clock) ------------------------------------------
    const rail = document.querySelector<HTMLElement>('[data-rail]');
    const shown = { p: 0 };
    let introDone = reduced || !rail;
    let applyRail = (_p: number) => {};
    let frac = () => 0;
    let marks: { el: HTMLElement; ms: number }[] = [];
    const live = rail && rail.querySelector('[data-now]') ? rail : null;
    if (rail && !live) {
      // A finished slate: the rail is simply full. Only the marks are read below.
      marks = [...rail.querySelectorAll<HTMLElement>('[data-mark]')].map((el) => ({ el, ms: Number(el.dataset.mark) }));
      introDone = true;
    }
    if (live) {
      const rail = live;
      const start = Number(rail.dataset.start);
      const end = Number(rail.dataset.end);
      const track = rail.querySelector<HTMLElement>('.rail__track')!;
      const now = rail.querySelector<HTMLElement>('[data-now]')!;
      const done = rail.querySelector<HTMLElement>('[data-rail-done]')!;
      const nowLabel = rail.querySelector<HTMLElement>('.rail__nowlabel')!;
      marks = [...rail.querySelectorAll<HTMLElement>('[data-mark]')].map((el) => ({ el, ms: Number(el.dataset.mark) }));
      frac = () => gsap.utils.clamp(0, 1, (Date.now() - start) / (end - start));
      let width = track.clientWidth;
      applyRail = (p: number) => {
        shown.p = p;
        gsap.set(now, { x: p * width });
        gsap.set(done, { scaleX: p });
        nowLabel.style.translate = `${-p * 100}% 0`; // the label stays inside the rail at both ends
        rail.classList.toggle('is-over', p >= 1);
      };
      const onResize = () => { width = track.clientWidth; applyRail(shown.p); };
      addEventListener('resize', onResize);
      off.push(() => removeEventListener('resize', onResize));
      rail.classList.add('is-live');
      if (reduced || frac() >= 1) { applyRail(frac()); introDone = true; }
      else {
        applyRail(0);
        intro.add(gsap.to(shown, { p: frac(), duration: 0.9, ease: ease.out, onUpdate: () => applyRail(shown.p), onComplete: () => { introDone = true; } }), 0.2);
      }
    }

    // --- per-second text: countdown, game status, passed marks ----------------------------------
    const clock = document.querySelector<HTMLElement>('[data-clock]');
    const clockLabel = clock?.querySelector<HTMLElement>('[data-clock-label]');
    const clockN = clock?.querySelector<HTMLElement>('[data-clock-n]');
    const clockSub = clock?.querySelector<HTMLElement>('[data-clock-sub]');
    const statuses = [...document.querySelectorAll<HTMLElement>('[data-status]')].map((el) => ({ el, tip: Number(el.dataset.tip) }));
    const tips = [...document.querySelectorAll<HTMLElement>('.mark--tip[data-mark]')].map((el) => Number(el.dataset.mark)).sort((a, b) => a - b);
    const nowTime = document.querySelector<HTMLElement>('[data-now-time]');
    const set = (el: HTMLElement, text: string) => { if (el.textContent !== text) el.textContent = text; };

    let lastSecond = -1;
    const tick = () => {
      const t = Date.now();
      const sec = Math.floor(t / 1000);
      if (sec === lastSecond) return;
      lastSecond = sec;
      if (introDone && live) applyRail(frac());
      if (nowTime) set(nowTime, et(t));
      for (const m of marks) m.el.classList.toggle('is-past', m.ms <= t);
      for (const s of statuses) {
        set(s.el,
          t < s.tip ? `Tips in ${hm(s.tip - t)}`
          : t < s.tip + FINAL_AFTER ? 'In play'
          : 'Played, the grade comes with the next update');
      }
      if (clock && clockLabel && clockN && clockSub) {
        const next = tips.find((x) => x > t);
        const last = Number(clock.dataset.lastTip);
        if (next) {
          set(clockLabel, next === tips[0] ? 'First tip in' : 'Next tip in');
          set(clockN, clockText(next - t));
          set(clockSub, `${et(next)}, ${ist(next)}`);
        } else {
          set(clockLabel, t < last + FINAL_AFTER ? 'Every game has tipped off' : 'This slate is over');
          set(clockN, t < last + FINAL_AFTER ? 'In play' : 'Final');
          set(clockSub, t < last + FINAL_AFTER ? `last final due about ${et(last + FINAL_AFTER)}` : 'grades arrive with the next update');
        }
      }
    };
    tick();
    gsap.ticker.add(tick);
    off.push(() => gsap.ticker.remove(tick));

    // --- court lines: markers slide from centre court to where each side put the game ---------
    document.querySelectorAll<HTMLElement>('[data-court]').forEach((court) => {
      if (reduced) return;
      const dot = court.querySelector<HTMLElement>('[data-dot]');
      const ring = court.querySelector<HTMLElement>('[data-ring]');
      const ball = court.querySelector<HTMLElement>('[data-ball]');
      const trail = court.querySelector<HTMLElement>('[data-trail]');
      const mx = parseFloat(getComputedStyle(court).getPropertyValue('--mx')) || 0.5;
      const kx = parseFloat(getComputedStyle(court).getPropertyValue('--kx')) || 0.5;
      const tl = gsap.timeline({ paused: true });
      const w = () => court.clientWidth;
      if (dot) tl.from(dot, { x: () => (0.5 - mx) * w(), duration: 0.9, ease: ease.out }, 0);
      if (ring) tl.from(ring, { x: () => (0.5 - kx) * w(), duration: 0.9, ease: ease.out }, 0.06);
      if (trail) tl.from(trail, { scaleX: 0, duration: 0.9, ease: ease.out }, 0.5);
      if (ball) {
        const bx = ball.style.left ? parseFloat(ball.style.left) / 100 : 0.5;
        tl.from(ball, { x: () => (mx - bx) * w(), autoAlpha: 0, duration: 0.9, ease: ease.out }, 0.5);
      }
      ScrollTrigger.create({ trigger: court, start: 'top 92%', once: true, onEnter: () => tl.play() });
    });

    // --- record bars fill along their rails -------------------------------------------------------
    document.querySelectorAll<HTMLElement>('[data-bar]').forEach((bar, i) => {
      if (reduced) return;
      gsap.from(bar, { scaleX: 0, duration: 0.9, ease: ease.out, delay: i * 0.06, scrollTrigger: { trigger: bar, start: 'top 92%', once: true } });
    });

    return () => off.forEach((f) => f());
  });
}
