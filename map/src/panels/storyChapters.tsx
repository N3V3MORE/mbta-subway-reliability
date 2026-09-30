import type { ReactNode } from 'react'
import { BarList, Dumbbell, StackedBars } from '../components/charts'
import { LineTag } from '../components/LineTag'
import { formatDate } from '../data/format'
import { lineName, SERIES, seconds } from '../data/labels'
import { lostPlaces, MOVING, placeName, STANDING } from '../data/lost'
import type { MetricId, PreparedNetwork } from '../data/network'
import { DELAY_BANDS } from '../data/replay'
import type { LostMetric } from '../data/lost'
import type { Mode } from '../map/NetworkMap'
import type { Period, Results } from '../types'

/** What the map beside the story shows while a chapter is being read. */
export type StoryView = { lostMetric?: LostMetric; metric?: MetricId; mode: Mode; replayDate?: string; seconds?: number }

export type Chapter = { answer: ReactNode; body?: ReactNode; explore: { href: string; label: string }; question: string; view: StoryView }

const pct = (value: number) => `${Math.round(value * 100)}%`
/** Small counts read better as words beside a percentage ("6% five stops ahead"). */
const WORDS: Record<number, string> = { 2: 'two', 3: 'three', 4: 'four', 5: 'five', 10: 'ten' }
const score = (period: Period, id: string) => period.regression.find((row) => row.id === id)?.mae

/** The chapters, in reading order, from the exported results. A chapter whose data is missing is left out. */
export function chapters(network: PreparedNetwork, results: Results): Chapter[] {
  const spring = results.periods.find((p) => p.id === 'spring') ?? results.periods[0]
  if (!spring) return []
  const out: Chapter[] = []
  const days = network.window

  out.push({
    answer: (
      <>
        Every train the MBTA’s records caught on the subway from {formatDate(days[0], 'short')} to {formatDate(days[1], 'short')}:
        {spring.arrivals ? <> {(spring.arrivals / 1e6).toFixed(1)} million arrivals</> : null} at {network.stations.length} stations.
        This page asks four things of them: how late trains run, whether it can be predicted, where the time goes, and what
        cannot be foreseen.
      </>
    ),
    explore: { href: '#/network', label: 'Explore the stations' },
    question: 'The T, as it actually ran',
    view: { metric: 'entries', mode: 'network' },
  })

  if (spring.bandsByLine && spring.lateShare !== undefined) {
    const worst = spring.bandsByLine[spring.bandsByLine.length - 1]
    const onTime = spring.bandsByLine.reduce((sum, row) => sum + row.shares[0], 0) / spring.bandsByLine.length
    out.push({
      answer: (
        <>
          Mostly close to plan: on a typical line {pct(onTime)} of arrivals keep riders waiting less than a minute longer than
          planned. But <b>{pct(spring.lateShare)}</b> keep them waiting 5 minutes or more, most on the{' '}
          {lineName(network.lineById.get(worst.line), worst.line)}{worst.line.startsWith('Green-') ? ' branch' : ' Line'}.
        </>
      ),
      body: (
        <StackedBars
          colors={DELAY_BANDS.map((band, i) => (i === 0 ? 'var(--surface-3)' : band.color))}
          labels={DELAY_BANDS.map((band) => band.label)}
          rows={[...spring.bandsByLine].reverse().map((row) => ({ key: row.line, label: <LineTag id={row.line} line={network.lineById.get(row.line)} />, shares: row.shares }))}
        />
      ),
      explore: { href: '#/network', label: 'See every station' },
      question: 'How late do trains run?',
      view: { metric: 'late', mode: 'network' },
    })
  }

  const model = score(spring, 'hist_gradient_boosting_change')
  const guess = score(spring, 'baseline_persistence')
  const lookup = score(spring, 'baseline_run_time')
  if (model && guess && lookup) {
    const far = spring.horizons.find((h) => h.stops === 5)
    out.push({
      answer: (
        <>
          Yes. One stop ahead, the model is off by <b>{seconds(model)}</b> on average; assuming a train “stays as late as it is”
          is off by {seconds(guess)}. A simple rule, when it left plus the usual travel time, does about as well one stop ahead
          ({seconds(lookup)}){far ? <>, but five stops ahead the model is ahead: {seconds(far.model)} against {seconds(far.runTime)}</> : null}.
        </>
      ),
      body: (
        <BarList
          bars={[
            { color: SERIES.persistence, key: 'guess', label: 'Stays as late as it is', value: guess },
            { color: SERIES.lookup, key: 'lookup', label: 'Left the last stop + usual travel time', value: lookup },
            { color: SERIES.model, key: 'model', label: 'Our model', value: model },
          ]}
          format={seconds}
        />
      ),
      explore: { href: `#/replay/${network.replays[0]?.date ?? ''}`, label: 'Watch a day, train by train' },
      question: 'Can we predict how late the next train will be?',
      view: { mode: 'replay', replayDate: network.replays[0]?.date, seconds: 8 * 3_600 },
    })
  }

  if (spring.lost) {
    const s = spring.lost.summary
    const worst = lostPlaces(spring).sort((a, b) => b.mean - a.mean).slice(0, 5)
    out.push({
      answer: (
        <>
          About <b>{Math.round(s.lost_minutes_per_day).toLocaleString('en-US')} train-minutes a day</b> go beyond what a good run
          takes: {pct(s.running_share)} of it moving, the rest standing at platforms. It is lost in the same places every season,
          and rush hour barely changes it.
        </>
      ),
      body: (
        <BarList
          bars={worst.map((place) => ({ color: place.kind === 'stretch' ? MOVING : STANDING, key: place.id, label: placeName(network, place), value: place.mean }))}
          format={seconds}
        />
      ),
      explore: { href: '#/lost', label: 'See every stretch and platform' },
      question: 'Where does the T lose time?',
      view: { lostMetric: 'train', mode: 'lost' },
    })
  }

  const steps = spring.ablation
  if (steps.length > 2) {
    const gain = (label: string) => {
      const i = steps.findIndex((s) => s.features === label)
      return i > 0 ? steps[i - 1].mae - steps[i].mae : null
    }
    const sources = [['+ delay propagation', 'How late it already is'], ['+ other trains', 'The trains around it'], ['+ weather', 'Weather'],
      ['+ alerts', 'MBTA service alerts'], ['+ demand', 'How busy the station is']] as const
    const bars = sources.flatMap(([step, label]) => { const g = gain(step); return g === null ? [] : [{ key: step, label, value: Math.max(0, g) }] })
    const minor = Math.max(...['+ weather', '+ alerts', '+ demand'].map((step) => Math.abs(gain(step) ?? 0)))
    const lead = gain('+ delay propagation')
    out.push({
      answer: (
        <>
          Almost everything we tried. One fact does the predicting: how late the train already is
          {lead !== null ? <>, which cuts the error by {Math.round(lead)} s</> : null}. Weather, service alerts and how busy the
          station is change it by {minor < 0.1 ? 'less than a tenth of a second' : `at most ${seconds(minor)}`}.
          {spring.crossTrack ? <> And busy stations are not the unreliable ones: the two are unrelated
            (p = {spring.crossTrack.pValue.toFixed(2)}).</> : null}
        </>
      ),
      body: <BarList bars={bars.map((b) => ({ ...b, color: SERIES.model }))} format={(v) => `${seconds(v)} better`} />,
      explore: { href: '#/results', label: 'See every result' },
      question: 'What doesn’t help?',
      view: { metric: 'late', mode: 'network' },
    })
  }

  const onsets = spring.warning?.filter((w) => w.method === 'model' && w.subset === 'onsets' && w.recall !== null)
  if (onsets?.length) {
    const storm = network.replays.find((d) => d.date === '2026-02-23') ?? network.replays[1]
    out.push({
      answer: (
        <>
          Sudden big delays. Of trains that are on time now but end up 10+ minutes late, the model flags{' '}
          <b>{pct(onsets[0].recall!)}</b> one stop ahead while keeping half its alarms right
          {onsets[1] ? <>, and only {pct(onsets[1].recall!)} {WORDS[onsets[1].stops] ?? onsets[1].stops} stops ahead</> : null}. Even the MBTA’s own alerts
          arrive too late to help.
        </>
      ),
      body: (
        <BarList
          bars={onsets.map((w) => ({ color: SERIES.model, key: String(w.stops), label: `${w.stops} stop${w.stops > 1 ? 's' : ''} ahead`, value: w.recall! }))}
          format={pct}
          max={1}
        />
      ),
      explore: { href: storm ? `#/replay/${storm.date}` : '#/replay', label: 'Replay the February snowstorm' },
      question: 'What can’t be predicted?',
      view: { mode: 'replay', replayDate: storm?.date, seconds: 8 * 3_600 },
    })
  }

  const periods = results.periods.flatMap((p) => {
    const m = score(p, 'hist_gradient_boosting_change')
    const g = score(p, 'baseline_persistence')
    return m && g ? [{ from: g, key: p.id, label: p.label, to: m }] : []
  })
  if (periods.length > 1) {
    out.push({
      answer: (
        <>
          Yes. In a snowy winter and on a summer the model never saw during its design, its error stays about a third of the rule
          of thumb’s.
        </>
      ),
      body: <Dumbbell format={seconds} rows={periods} />,
      explore: { href: '#/results', label: 'Compare the periods' },
      question: 'Does it hold up?',
      view: { metric: 'late', mode: 'network' },
    })
  }
  return out
}

