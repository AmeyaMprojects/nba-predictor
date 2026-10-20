// Page transitions. The adapter runs leave() while the next page loads.
// Look (DESIGN.md, the concept's signature): the next page is "sealed" on the way in. An ink
// sheet slides across, a stamp carrying the destination's name presses onto it (scale 1.06 → 1,
// back.out), then sheet and stamp lift off together. Under 0.7 s each way; a short fade under
// reduced motion.
import { gsap } from 'gsap';
import { ease, prefersReducedMotion } from './env';

export type Transition = {
  leave: (curtain: HTMLElement) => Promise<unknown> | gsap.core.Animation;
  enter: (curtain: HTMLElement) => Promise<unknown> | gsap.core.Animation;
};

/** Label the stamp with where the visitor is going (called by the adapter with the target URL). */
export function setDestination(url: URL) {
  const base = (import.meta.env.BASE_URL || '/').replace(/\/+$/, '');
  const path = url.pathname.startsWith(base) ? url.pathname.slice(base.length) : url.pathname;
  const day = /^\/day\/(\d{4})-(\d{2})-(\d{2})/.exec(path);
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const label = day ? `${Number(day[3])} ${months[Number(day[2]) - 1]}`
    : path.startsWith('/method') ? 'Method'
    : url.hash === '#ledger' ? 'Ledger'
    : url.hash === '#record' ? 'Record'
    : 'Tonight';
  const el = document.querySelector<HTMLElement>('[data-curtain-label]');
  if (el) el.textContent = label;
}

let current: Transition = {
  // ~0.35 s: the sheet slides in while the stamp presses onto it.
  leave: (c) => {
    const panel = c.querySelector('.curtain__panel');
    const stamp = c.querySelector('.curtain__stamp');
    return gsap.timeline()
      .set(c, { autoAlpha: 1 })
      .set(stamp, { autoAlpha: 0, x: 0, y: 0, yPercent: 0 })
      .fromTo(panel, { xPercent: -100, yPercent: 0 }, { xPercent: 0, duration: 0.24, ease: ease.inOut })
      .fromTo(stamp, { autoAlpha: 0, scale: 1.06, boxShadow: '9px 9px 0 0 rgba(200,70,26,1)' },
        { autoAlpha: 1, scale: 1, boxShadow: '2px 2px 0 0 rgba(200,70,26,1)', duration: 0.18, ease: ease.stamp }, '-=0.1');
  },
  // ~0.35 s: the sheet lifts and the stamp travels into the new page's heading (FLIP), so the
  // label you clicked lands where the page names itself.
  enter: (c) => {
    const panel = c.querySelector<HTMLElement>('.curtain__panel')!;
    const stamp = c.querySelector<HTMLElement>('.curtain__stamp')!;
    const target = document.querySelector<HTMLElement>('#main h1');
    const tl = gsap.timeline();
    if (target) {
      const from = stamp.getBoundingClientRect();
      const to = target.getBoundingClientRect();
      const scale = Math.max(0.2, Math.min(1, to.height / from.height));
      tl.to(stamp, {
        x: to.left - from.left - (from.width * (1 - scale)) / 2,
        y: to.top - from.top - (from.height * (1 - scale)) / 2,
        scale, duration: 0.35, ease: ease.inOut,
      }, 0).to(stamp, { autoAlpha: 0, duration: 0.14 }, 0.21);
    } else {
      tl.to(stamp, { autoAlpha: 0, duration: 0.2 }, 0);
    }
    return tl.to(panel, { yPercent: -100, duration: 0.35, ease: ease.inOut }, 0).set(c, { autoAlpha: 0 });
  },
};

const reduced: Transition = {
  leave: (c) => gsap.timeline().set(c, { autoAlpha: 1 }).set(c.querySelector('.curtain__panel'), { xPercent: 0, yPercent: 0 }).set(c.querySelector('.curtain__stamp'), { autoAlpha: 1, x: 0, y: 0, yPercent: 0, scale: 1 }).fromTo(c, { opacity: 0 }, { opacity: 1, duration: 0.15 }),
  enter: (c) => gsap.to(c, { autoAlpha: 0, duration: 0.15 }),
};

export const setTransition = (t: Transition) => { current = t; };
const pick = () => (prefersReducedMotion() ? reduced : current);
const curtain = () => document.getElementById('curtain') as HTMLElement;

export const playLeave = () => Promise.resolve(pick().leave(curtain()));
export const playEnter = () => Promise.resolve(pick().enter(curtain()));
