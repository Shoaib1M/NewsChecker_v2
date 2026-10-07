/*
FILE PURPOSE:
Load every route in a built client and fail on any runtime error.

WHY THIS EXISTS:
`npm run build` and `eslint` both pass on a component that references an
undefined identifier — Vite does not resolve names at build time, and the
lint config does not flag it in JSX. A missing import therefore ships as a
blank white page with "X is not defined" in the console, and every check in
CI stays green.

That happened: a `<Filter />` icon was used on the How It Works page without
being imported, the build and lint reported success, and the page rendered
nothing at all. This script would have caught it in one second.

Playwright is NOT a dependency of this project — it would pull a few hundred
megabytes of browser for a project that otherwise runs locally for demos. The
script skips itself with instructions when Playwright is absent, so
`npm run smoke` is safe to run in a fresh checkout.

Run against a preview server:
    npm run build
    npx vite preview --port 4173 &
    npm run smoke
*/

let chromium;
try {
  ({ chromium } = await import("playwright"));
} catch {
  console.log(
    "Skipping smoke test: playwright is not installed.\n" +
    "  npm i -D playwright && npx playwright install chromium",
  );
  process.exit(0);
}

const BASE = process.env.SMOKE_BASE_URL || "http://localhost:4173";
const ROUTES = ["", "#/comparison", "#/how-it-works"];

const browser = await chromium.launch();
let failures = 0;

for (const route of ROUTES) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error.message)));
  page.on("console", (message) => {
    // Network noise from unstubbed API calls is expected; script errors are not.
    if (message.type() === "error" && !/Failed to load resource/.test(message.text())) {
      errors.push(message.text());
    }
  });

  // The app calls these on load; stub them so a missing backend isn't a failure.
  await page.route("**/api/**", (r) => r.fulfill({ status: 401, body: "{}" }));

  await page.goto(`${BASE}/${route}`, { waitUntil: "networkidle" });
  await page.waitForTimeout(400);

  const text = (await page.locator("body").innerText()).trim();
  const blank = text.length < 40;

  if (errors.length || blank) {
    failures += 1;
    console.error(`FAIL  ${route || "/"}`);
    if (blank) console.error(`        rendered ${text.length} characters — blank page`);
    for (const error of errors) console.error(`        ${error}`);
  } else {
    console.log(`ok    ${route || "/"}  (${text.length} chars)`);
  }

  await page.close();
}

await browser.close();
if (failures) {
  console.error(`\n${failures} route(s) failed to render.`);
  process.exit(1);
}
console.log("\nall routes rendered without errors");
