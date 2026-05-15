# Demo recordings

Raw clips live here as `.mp4`; static poster frames for tables live under `previews/*.webp`.

For **GitHub README autoplay**, export a **GIF** (full clip length; GIF is
palette‑compressed and typically lower fps than MP4) and host it next to the clip
on the project site (`vlapilot_webpage`: `docs/public/demo/`), same basename as
the MP4. Example matching the deployed assets:

```bash
ffmpeg -y -i cleandesk.mp4 -vf "fps=8,scale=400:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse" cleandesk.gif
```

Raise `fps` for smoother motion (larger file); lower `scale` width to shrink the
GIF. Omit `-t`/`-ss`/`…trim` filters if you want the entire MP4 baked into the GIF.

## Repo size vs hosting

Large binaries bloat git history. Common patterns:

| Approach | Typical use |
|----------|-------------|
| **GitHub Releases** | Attach `.mp4` / assets to a tag; link from README. Keeps the default branch lean. |
| **Git LFS** | Track heavy media in-repo with LFS pointers; needs bandwidth/quota. |
| **Object storage + CDN** | R2, S3, GCS public bucket, or static host; README links to `https://…/demo/foo.mp4`. |
| **YouTube / Vimeo (unlisted)** | Embed or link; good for long demos, less ideal for “download exact clip”. |
| **Hugging Face** | Dataset or Space with video files; common in ML / robotics projects. |

For this repo, MP4s are **H.264 + AAC**, `yuv420p`, `+faststart` for web playback. Re-encode locally, e.g.:

```bash
ffmpeg -y -i input.mp4 -c:v libx264 -crf 26 -preset medium -pix_fmt yuv420p \
  -movflags +faststart -c:a aac -b:a 96k output.mp4
```

Tune `-crf` (lower = larger/better, try 24–28).
