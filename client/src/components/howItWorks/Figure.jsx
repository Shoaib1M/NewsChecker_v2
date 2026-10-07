/*
FILE PURPOSE:
A captioned frame for every diagram on the How It Works page.

WHY THIS EXISTS:
Diagrams have a natural width. On a phone, shrinking them to fit makes their
labels unreadable, so the frame scrolls sideways instead and the caption says
what the picture shows in words — which is also what a screen reader gets.
*/

export default function Figure({ caption, children, minWidth = 560 }) {
  return (
    <figure className="hiw-figure">
      <div className="hiw-figure-scroll">
        <div className="hiw-figure-inner" style={{ minWidth }}>
          {children}
        </div>
      </div>
      {caption && <figcaption className="hiw-figure-caption">{caption}</figcaption>}
    </figure>
  );
}
