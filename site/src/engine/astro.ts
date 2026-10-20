// Astro adapter (ClientRouter). Call start(setup) once from the layout's <script>.
import { boot, cleanup, enter, leave, syncFlags } from './index';

let navigating = false;
let popstate = false;

export function start(setup?: () => void) {
  document.addEventListener('astro:before-preparation', (e) => {
    navigating = true;
    popstate = e.navigationType === 'traverse';
    if (popstate) return; // back/forward: no curtain
    const load = e.loader;
    e.loader = async () => { await Promise.all([leave(), load()]); };
  });
  document.addEventListener('astro:before-swap', () => cleanup());
  document.addEventListener('astro:after-swap', () => syncFlags());
  document.addEventListener('astro:page-load', () => {
    if (!navigating) return void boot(setup);
    navigating = false;
    enter(popstate ? { transition: false, scrollTop: false } : undefined);
  });
  boot(setup);
}
