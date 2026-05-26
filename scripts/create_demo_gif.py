from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


OUT = Path("docs/assets/demo-flow.gif")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 720
    colors = ["#f8fafc", "#eef2ff", "#ecfeff", "#f0fdf4"]
    callouts = [
        "upload -> profile",
        "profile -> charts",
        "question -> tool call",
        "answer -> artifacts",
    ]
    steps = [
        (
            "1 Upload dataset",
            "CSV/XLSX file creates a bounded session with preview, TTL, and upload limits.",
        ),
        (
            "2 Dashboard",
            "Profiling produces rows, columns, missing values, suggested insights, and chart recommendations.",
        ),
        (
            "3 Ask in Vietnamese",
            "Router or Gemini fallback selects a whitelisted pandas tool with validated arguments.",
        ),
        (
            "4 Trace and artifacts",
            "The response returns answer, table/chart spec, and a readable tool trace for debugging.",
        ),
    ]

    title_font = _font(46)
    body_font = _font(30)
    small_font = _font(24)
    frames = []
    for index, (title, body) in enumerate(steps):
        image = Image.new("RGB", (width, height), colors[index])
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (50, 45, 1150, 675),
            radius=28,
            fill="white",
            outline="#cbd5e1",
            width=3,
        )
        draw.text((95, 88), "AI Data Analyst Agent", fill="#0f172a", font=small_font)
        draw.rounded_rectangle(
            (95, 145, 370, 600),
            radius=18,
            fill="#f1f5f9",
            outline="#dbe4ef",
            width=2,
        )
        draw.text((125, 185), "Sidebar", fill="#475569", font=small_font)
        draw.rounded_rectangle((125, 235, 340, 295), radius=12, fill="#4f46e5")
        draw.text((155, 250), "Analyze", fill="white", font=small_font)
        draw.rounded_rectangle(
            (410, 145, 1105, 600),
            radius=18,
            fill="#ffffff",
            outline="#dbe4ef",
            width=2,
        )
        draw.text((455, 190), title, fill="#111827", font=title_font)
        y = 275
        for line in textwrap.wrap(body, width=55):
            draw.text((455, y), line, fill="#334155", font=body_font)
            y += 42
        draw.rounded_rectangle((455, 455, 1035, 540), radius=16, fill="#e0f2fe")
        draw.text((485, 482), callouts[index], fill="#1e293b", font=body_font)
        draw.text((970, 625), f"{index + 1}/4", fill="#64748b", font=small_font)
        frames.append(image)

    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=1300,
        loop=0,
    )
    print(OUT)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
