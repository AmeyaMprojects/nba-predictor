// Base-aware internal URLs. GitHub Pages serves the site under /nba-predictor/.
const BASE = import.meta.env.BASE_URL.replace(/\/+$/, '');

/** `href('/method/')` → `/nba-predictor/method/`. Keeps a trailing slash on page paths. */
export function href(path = '/') {
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${BASE}${p}`;
}

export const dayHref = (date: string) => href(`/day/${date}/`);
