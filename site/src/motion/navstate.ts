// Marks the current nav item on every page swap (first load, link navigation and back/forward),
// from the URL itself, so a restored page can never keep the previous page's marker.
import { onPage } from '../engine';

export function startNavState() {
  onPage(({ path }) => {
    const base = (import.meta.env.BASE_URL || '/').replace(/\/+$/, '');
    const p = path.startsWith(base) ? path.slice(base.length) || '/' : path;
    const current = p.startsWith('/method') ? 'Method' : p.startsWith('/day/') ? 'Ledger' : p === '/' ? 'Tonight' : '';
    document.querySelectorAll<HTMLAnchorElement>('.frame__nav a').forEach((a) => {
      if (a.textContent?.trim() === current) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
      a.blur();
    });
    const html = document.documentElement;
    html.classList.add('nav-still');
    const moved = () => html.classList.remove('nav-still');
    addEventListener('pointermove', moved, { once: true });
    return () => removeEventListener('pointermove', moved);
  });
}
