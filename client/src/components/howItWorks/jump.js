/*
In-page navigation for the How It Works page.

The app routes on the URL hash (#/how-it-works), so an ordinary
<a href="#stage-nli"> would change the "page" to "stage-nli" and fall back to
Home. Links on this page scroll instead and leave the hash alone.
*/

export function jumpTo(event, id) {
  event.preventDefault();
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}
