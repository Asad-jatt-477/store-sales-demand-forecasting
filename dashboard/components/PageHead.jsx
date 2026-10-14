export default function PageHead({ title, intro, meta, children }) {
  return (
    <header className="page-head">
      <div>
        <h1>{title}</h1>
        {intro && <p>{intro}</p>}
      </div>
      {children || (meta && (
        <div className="context">
          <div><b>{meta.n_series.toLocaleString()}</b> series across {meta.n_stores} stores and {meta.n_families} families</div>
          <div>History to <b>{meta.cutoff}</b>, forecast horizon {meta.horizon} days</div>
        </div>
      ))}
    </header>
  );
}
