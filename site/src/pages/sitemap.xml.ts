// sitemap.xml: home, method and every day page, from the same data the pages are built from.
import type { APIRoute } from 'astro';
import { getData } from '../lib/data';
import { href } from '../lib/url';

export const GET: APIRoute = ({ site }) => {
  const { days } = getData();
  const last = days.at(-1)?.date;
  const urls = [
    { loc: href('/'), lastmod: last },
    { loc: href('/method/') },
    ...days.map((d) => ({ loc: href(`/day/${d.date}/`), lastmod: d.date })),
  ];
  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map((u) => `  <url><loc>${new URL(u.loc, site).href}</loc>${u.lastmod ? `<lastmod>${u.lastmod}</lastmod>` : ''}</url>`).join('\n')}
</urlset>
`;
  return new Response(body, { headers: { 'Content-Type': 'application/xml' } });
};
