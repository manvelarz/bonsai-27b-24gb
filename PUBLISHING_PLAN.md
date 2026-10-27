# Publishing plan (queued, one by one)

## Facts about the cropper repo (verified 2026-10-26)
- Local clone: `~/cropper` (Astro + Cloudflare adapter; blog = content collection)
- Remote: `https://github.com/Manvelarz/cropper.git` (private; API 404 unauthenticated)
- Branch at time of writing: `feat/best-free-photo-scanner-apps-2026-blog`
- Push from this box: NOT possible (no gh CLI, no credentials). Push via Windows side / SCP or a token.
- HARD RULE: never push or merge to `main`; feature branch only.

## Step 1 — blog post on smartphotocropper.com (canonical home)
- File: `astro/content/blog/bonsai-27b-24gb-vllm-vs-llamacpp.md`
- Front-matter schema (astro/content.config.ts): `title` (string), `date` (string,
  style "August 1, 2026"), `excerpt` (string). Nothing else required.
- Body convention: `# Title` H1 at top (duplicated from front-matter), then content.
- Add near the top or in an author box: one contextual link to
  https://smartphotocropper.com/app/ ("I build SmartPhotoCropper, an on-device AI
  photo cropper — this infra work is what I run locally for model experiments").
  ONE or TWO links max, natural phrasing. Do not stuff anchors.
- Category/nav: it is an engineering note; keep it out of photo-content nav if a
  "notes"/"engineering" tag exists, otherwise rely on the blog index.
- Verify locally: `npm install` (if needed) + `npm run build` (Astro lives in ./astro,
  builds to ./dist/astro). Do not deploy; Cloudflare Pages auto-deploys on main pushes
  only — so a feature branch push is safe.

## Step 2 — GitHub mirror (agent/developer discoverability)
- New repo (suggested name): `bonsai-27b-24gb` under manvelarz, draft branch only.
- Source: `~/bonsai-blog/` (article.md, CREDITS.md, README.md, data/, scripts/).
- Add a canonical note at top of article.md + in README:
  `Canonical: https://smartphotocropper.com/blog/<slug>/`
- GitHub links are effectively nofollow → value is traffic/credibility, not SEO.

## Step 3 — distribution (both point at the BLOG URL, not GitHub)
- HF: post in Community tab of `fraserprice/Ternary-Bonsai-2-27B-vllm`; comment on
  issue/PR #1 (v0.31 bump) with the 24 GB ceiling data (114,688 max, fp8 IMA under
  concurrency, turboquant 'auto' bug) — genuinely useful to that author.
- Reddit r/LocalLLaMA: title "262k context on a 24 GB 3090 with Ternary-Bonsai-27B:
  vLLM vs llama.cpp, measured"; link to blog; include the two tables in the post body.
- Optional later: dev.to/Medium syndication with rel=canonical.

## Open decisions
- License (suggest MIT for scripts/, CC BY 4.0 for article+data).
- Repo name + whether GitHub Pages is wanted.
