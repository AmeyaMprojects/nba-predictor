export const prefersReducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
export const isTouch = () => matchMedia('(hover: none), (pointer: coarse)').matches;
export const isMobile = () => matchMedia('(max-width: 767px)').matches;

// After a page transition, or on a phone, the intro is cut short: the page is ready at once and
// only the signature (the stamp) moves, briefly. Keeps the first screen settled well under 2.6 s.
let navigated = false;
export const markNavigated = () => { navigated = true; };
export const quickIntro = () => navigated || isMobile();

// One easing family for the whole site (DESIGN.md, motion language).
// settle: every marker, rail and entry. stamp: only the seal press.
export const ease = {
  out: 'expo.out',
  inOut: 'expo.inOut',
  soft: 'power3.out',
  stamp: 'back.out(1.4)',
} as const;
