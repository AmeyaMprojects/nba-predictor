// Method page: the "On this page" index marks the section being read (one clock: ScrollTrigger).
import { ScrollTrigger } from 'gsap/ScrollTrigger';
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
    links.forEach((a) => {
      const id = a.getAttribute('href')!.slice(1);
      const section = document.getElementById(id)?.closest('section');
      if (!section) return;
      ScrollTrigger.create({
        trigger: section, start: 'top 45%', end: 'bottom 45%',
        onToggle: (st) => { if (st.isActive) mark(id); },
      });
    });
    mark(links[0].getAttribute('href')!.slice(1));
  });
}
