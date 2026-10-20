// Global motion, registered once on the client before the first page initializes.
import { onScroll } from '../engine';
import { startSeal } from './seal';

export function setupMotion() {
  // The header steps aside while reading down and returns on the way up.
  let last = 0;
  onScroll(({ y }) => {
    if (Math.abs(y - last) < 6) return;
    document.documentElement.classList.toggle('frame-away', y > 200 && y > last);
    last = y;
  });
  startSeal();
}
