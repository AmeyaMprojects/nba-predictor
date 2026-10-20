export const prefersReducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
export const isTouch = () => matchMedia('(hover: none), (pointer: coarse)').matches;
export const isMobile = () => matchMedia('(max-width: 767px)').matches;

// One easing family for the whole site (DESIGN.md, motion language).
// settle: every marker, rail and entry. stamp: only the seal press.
export const ease = {
  out: 'expo.out',
  inOut: 'expo.inOut',
  soft: 'power3.out',
  stamp: 'back.out(1.4)',
} as const;
