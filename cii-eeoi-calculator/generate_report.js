// Captures the running CII & EEOI Streamlit app as a single A4-landscape PDF.
//
// Usage: node generate_report.js [url] [outputPath]
//   url        defaults to http://localhost:8501
//   outputPath defaults to cii_eeoi_report.pdf
//
// Requires: npm install (see package.json) before first use - this downloads
// Puppeteer's bundled Chromium.

const puppeteer = require("puppeteer");

const url = process.argv[2] || "http://localhost:8501";
const outputPath = process.argv[3] || "cii_eeoi_report.pdf";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function expandAssumptions(page) {
  return page.evaluate(() => {
    const summaries = Array.from(document.querySelectorAll("summary"));
    const target = summaries.find((el) =>
      el.textContent.includes("Assumptions & methodology")
    );
    if (target && target.getAttribute("aria-expanded") !== "true") {
      target.click();
      return true;
    }
    return false;
  });
}

async function waitForImages(page) {
  await page.evaluate(async () => {
    const images = Array.from(document.querySelectorAll("img"));
    await Promise.all(
      images.map((img) => {
        if (img.complete) return Promise.resolve();
        return new Promise((resolve) => {
          img.addEventListener("load", resolve, { once: true });
          img.addEventListener("error", resolve, { once: true });
        });
      })
    );
  });
}

(async () => {
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  try {
    const page = await browser.newPage();

    // A generous desktop viewport so Streamlit's "wide" layout renders at
    // full width rather than a mobile/narrow breakpoint.
    await page.setViewport({ width: 1800, height: 1100, deviceScaleFactor: 2 });

    await page.goto(url, { waitUntil: "networkidle0", timeout: 60000 });

    // Streamlit hydrates over a websocket after the initial HTML load, so
    // network-idle alone isn't enough - wait for a real widget to appear.
    await page.waitForSelector('[data-testid="stMetric"]', { timeout: 60000 });

    // Let data-editor grids and the matplotlib chart finish drawing.
    await sleep(2000);
    await waitForImages(page);

    // Expand the "Assumptions & methodology" section so it's captured too.
    const expanded = await expandAssumptions(page);
    if (expanded) {
      await sleep(1000);
    }

    await page.emulateMediaType("screen");

    await page.pdf({
      path: outputPath,
      format: "A4",
      landscape: true,
      printBackground: true,
      preferCSSPageSize: false,
      margin: { top: "10mm", bottom: "10mm", left: "10mm", right: "10mm" },
    });

    console.log(`Saved report to ${outputPath}`);
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error("PDF generation failed:", err);
  process.exit(1);
});
