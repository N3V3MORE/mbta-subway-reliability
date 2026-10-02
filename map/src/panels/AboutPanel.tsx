import { CheckCircle2, CircleAlert } from 'lucide-react'
import { formatDate } from '../data/format'
import { REPO_URL, REPORTS_URL } from '../links'
import type { PreparedNetwork } from '../data/network'
import type { Results } from '../types'

/** Each source, what it gave the project, and its licence as the provider states it. */
const SOURCES: { href: string; licence: string; name: string; what: string }[] = [
  { href: 'https://www.mass.gov/massdot-developers-data-sources', licence: 'MassDOT Developers License Agreement',
    name: 'MBTA performance archive (LAMP)', what: 'Every observed stop with its scheduled time: the delay label.' },
  { href: 'https://www.mass.gov/massdot-developers-data-sources', licence: 'MassDOT Developers License Agreement',
    name: 'MBTA V3 API', what: 'Station locations, track shapes and line colours; the live collector.' },
  { href: 'https://www.mass.gov/massdot-developers-data-sources', licence: 'MassDOT Developers License Agreement',
    name: 'MBTA alerts archive', what: 'Service alerts with their cause, effect and timing.' },
  { href: 'https://creativecommons.org/publicdomain/zero/1.0/', licence: 'CC0 1.0',
    name: 'MBTA gated station entries', what: 'Entries per station per half-hour: the demand signal.' },
  { href: 'https://open-meteo.com/', licence: 'CC BY 4.0',
    name: 'Weather data by Open-Meteo.com', what: 'Hourly Boston weather, aggregated to the hour of each prediction.' },
  { href: 'https://www.openstreetmap.org/copyright', licence: 'ODbL',
    name: 'Base map: OpenFreeMap, © OpenMapTiles, © OpenStreetMap contributors', what: 'Streets, water and place names under the network.' },
]

export function AboutPanel({ network, results }: { network: PreparedNetwork; results: Results }) {
  const spring = results.periods.find((p) => p.id === 'spring')
  const checks = spring?.validation ?? []
  const failed = checks.filter((check) => check.hard && !check.ok)
  return (
    <>
      <header className="panel-head">
        <h1>About</h1>
        <p>
          Mini Boston shows the MBTA subway as it actually ran, from {formatDate(network.window[0], 'short')} to {formatDate(network.window[1], 'short')},
          and how well a train’s next-stop delay can be predicted. A CS 506 project.
        </p>
      </header>

      <section className="panel-section">
        <h2>Code and write-up</h2>
        <dl className="pair-list is-stacked">
          <div>
            <dt><a href={REPO_URL} rel="noreferrer" target="_blank">Source on GitHub</a></dt>
            <dd>The Python pipeline (collection, cleaning, features, models, 260 tests) and this app.</dd>
          </div>
          <div>
            <dt><a href={`${REPO_URL}/blob/main/REPORT.md`} rel="noreferrer" target="_blank">Full report</a></dt>
            <dd>Data, cleaning, models, evaluation and limitations, section by section.</dd>
          </div>
          {REPORTS_URL ? (
            <>
              <div>
                <dt><a href={`${REPORTS_URL}story.html`} rel="noreferrer" target="_blank">The results in plain words</a></dt>
                <dd>Eight interactive charts, one question each.</dd>
              </div>
              <div>
                <dt><a href={`${REPORTS_URL}report.html`} rel="noreferrer" target="_blank">Every table and figure</a></dt>
                <dd>
                  The generated report for spring. Also for{' '}
                  <a href={`${REPORTS_URL}winter/report.html`} rel="noreferrer" target="_blank">winter</a> and the{' '}
                  <a href={`${REPORTS_URL}holdout/report.html`} rel="noreferrer" target="_blank">July–September holdout</a>.
                </dd>
              </div>
            </>
          ) : null}
        </dl>
      </section>

      <section className="panel-section prose">
        <h2>What “late” means</h2>
        <p>
          Late is judged the way riders feel it: how much longer than planned they waited for a train, the gap behind the train in
          front minus the scheduled gap. Delay against the timetable is used for the predictions, but not to call a train late:
          on frequent lines trains are matched to timetable slots in order, so a line running slightly sparse drifts far
          “behind” while riders see near-normal service.
        </p>
        <h2>What the map draws</h2>
        <p>
          Trains are placed only between stops where the records observed them, never guessed beyond. Station columns and train
          colours use the same measure of lateness, so a tall column is a place riders wait.
        </p>
      </section>

      <section className="panel-section">
        <h2>Data and licences</h2>
        <p className="hint">
          Data provided by MassDOT and the MBTA, used under the MassDOT Developers License Agreement; it remains theirs and comes
          as is. This project is not affiliated with or endorsed by the MBTA or MassDOT.
        </p>
        <dl className="pair-list is-stacked">
          {SOURCES.map((source) => (
            <div key={source.name}>
              <dt><a href={source.href} rel="noreferrer" target="_blank">{source.name}</a></dt>
              <dd>{source.what} <span className="licence">{source.licence}</span></dd>
            </div>
          ))}
        </dl>
      </section>

      {spring?.cleaning ? (
        <section className="panel-section">
          <h2>Cleaning, spring window</h2>
          <table className="plain-table">
            <tbody>
              {spring.cleaning.filter((row) => row.removed > 0).map((row) => (
                <tr key={row.step}><td>{row.step}</td><td className="is-numeric">{row.removed.toLocaleString('en-US')}</td></tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}

      {checks.length ? (
        <section className="panel-section">
          <h2>Data checks</h2>
          <p className="status-line">
            {failed.length
              ? <><CircleAlert size={15} /> {failed.length} hard check{failed.length > 1 ? 's' : ''} failed</>
              : <><CheckCircle2 size={15} /> All {checks.filter((c) => c.hard).length} hard checks pass</>}
          </p>
          <ul className="check-list">
            {checks.map((check) => (
              <li className={check.ok ? '' : 'is-failed'} key={`${check.source}-${check.check}`}>
                {check.ok ? <CheckCircle2 aria-label="Passed" size={13} /> : <CircleAlert aria-label="Failed" size={13} />}
                <span>{check.check}</span>
                <small>{check.source}</small>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </>
  )
}
