// Method page: the "On this page" index marks the section being read. IntersectionObserver, so it
// works the same with or without motion (reduced motion turns off smooth scroll, not the spy).
import { onPage } from '../engine';

export function startToc() {
  onPage(() => {
    const links = [...document.querySelectorAll<HTMLAnchorElement>('.toc a[href^="#"]')];
    if (!links.length) return;
    const mark = (id: string) => links.forEach((a) => {
      const on = a.getAttribute('href') === `#${id}`;
      a.closest('li')?.classList.toggle('is-current', on);
      if (on) a.setAttribute('aria-current', 'location'); else a.removeAttribute('aria-current');
    });
    const sections = links
      .map((a) => document.getElementById(a.getAttribute('href')!.slice(1))?.closest('section'))
      .filter((s): s is HTMLElement => !!s);
    // A thin band at 40% of the viewport height: the section crossing it is the one being read.
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) mark(e.target.querySelector('h2')!.id);
    }, { rootMargin: '-40% 0px -59% 0px' });
    sections.forEach((s) => io.observe(s));
    mark(sections[0]?.querySelector('h2')?.id ?? '');
    return () => io.disconnect();
  });
}
