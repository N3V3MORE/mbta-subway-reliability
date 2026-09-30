/** Shapes of the JSON written by `python -m mbta_ds.cli export-map`. */

export type LonLat = [longitude: number, latitude: number]

export type Line = { color: string; id: string; name: string; short: string }

/** One branch of track: its geometry and its stations in order along it. */
export type Pattern = {
  /** Metres along the track of each station in `stations`. */
  at: number[]
  coords: LonLat[]
  /** Metres along the track at each vertex of `coords`. */
  dist: number[]
  id: string
  line: string
  stations: string[]
}

export type Station = {
  coords: LonLat
  demand: string | null
  entries: number | null
  id: string
  lines: string[]
  median: number | null
  name: string
  onTime: number | null
  p90: number | null
  /** Share of the day's entries in each half-hour, midnight first. */
  profile: number[] | null
  reliability: string | null
}

export type ReplaySummary = { incidents?: number; late: number; maeModel: number; maePersistence: number; stops: number; trips: number }

export type ReplayDay = ReplaySummary & { date: string; file: string; label: string; note: string }

export type Network = {
  bounds: [LonLat, LonLat]
  lines: Line[]
  patterns: Pattern[]
  replays: ReplayDay[]
  stations: Station[]
  window: [start: string, end: string]
}

/** One observed trip. Arrays are per stop, in order. */
export type Trip = {
  /** Arrival, in seconds after the start of the service day. */
  a: number[]
  /** Delay against the timetable at arrival, in seconds. */
  d: number[]
  dest: string
  id: string
  /** How much longer than planned riders waited for the train there, in seconds. */
  l: number[]
  line: string
  /** The model's prediction of `d`, made before the train reached the stop. */
  p: (number | null)[]
  pattern: number
  /** Station index into `Network.stations`. */
  s: number[]
  /** Seconds spent at the stop before leaving. */
  w: number[]
}

/** A service incident the MBTA reported that day (its alerts archive). */
export type Incident = {
  cause: string | null
  /** MBTA effect code, such as DETOUR or REDUCED_SERVICE. */
  effect: string
  /** When the alert was closed, in seconds after the start of the service day; null if never recorded. */
  end: number | null
  id: string
  lines: string[]
  /** When the alert was raised, in seconds after the start of the service day. */
  start: number
  /** Station indices into `Network.stations` the alert names; often none. */
  stations: number[]
  text: string
}

export type Replay = { date: string; incidents?: Incident[]; label: string; summary: ReplaySummary; trips: Trip[] }

/** Shapes of `results.json`: each built period's results (see `export_map.build_results`). */
export type ModelScore = { bias: number | null; id: string; label: string; mae: number; r2: number; rmse: number }
export type ClassifierScore = { f1: number; id: string; label: string; precision: number; prAuc: number; recall: number; rocAuc: number }
export type HorizonScore = { model: number; n: number; ownTrain: number; persistence: number; runTime: number; stops: number }
export type BacktestWindow = { candidate: number; from: string; model: number; n: number; persistence: number; runTime: number; to: string; trainDays: number }
export type LineError = { line: string; mae: number; median: number; n: number }
export type Warning = { method: string; n: number; positives: number; prAuc: number; recall: number | null; stops: number; subset: string }
export type ValidationCheck = { check: string; hard: boolean; limit: number | string; ok: boolean; op: string; source: string; value: number | string }

export type Period = {
  ablation: { features: string; mae: number; n: number; spread: number }[]
  /** Share of each line's arrivals per lateness band (`bands`), least late line first. */
  bands?: string[]
  bandsByLine?: { line: string; shares: number[] }[]
  arrivals?: number
  backtest: BacktestWindow[]
  byDayType: { dayType: string; mae: number; median: number; n: number }[]
  byLine: LineError[]
  classification: ClassifierScore[]
  cleaning?: { removed: number; step: string }[]
  horizons: HorizonScore[]
  hours?: number[]
  id: string
  importance?: { feature: string; label: string; value: number }[]
  label: string
  lateByHour?: { line: string; shares: (number | null)[] }[]
  lateShare?: number
  lost?: TimeLost
  note: string
  ranges?: { coverage: number; dayType: string; stops: number; width: number }[]
  regression: ModelScore[]
  split: { cutoff_service_date: string; test_dates: [string, string]; train_dates: [string, string] }
  validation?: ValidationCheck[]
  warning?: Warning[]
  window: [string | null, string | null]
}

export type Results = {
  crossSeason: { trained_on: string; spring_test_mae: number; winter_test_mae: number }[] | null
  periods: Period[]
}

/** Where a period's trains lost time (`segments` stage): per stretch of track and per platform. */
export type LostPlace = { dir: number; good: number; line: string; mean: number; median: number; p90: number; perDay: number; trains: number }
export type LostStretch = LostPlace & { from: string; terminal: boolean; to: string }
export type LostPlatform = LostPlace & { station: string }
export type TimeLost = {
  platforms: LostPlatform[]
  stretches: LostStretch[]
  summary: {
    by_hour: { hour: number; platform_mean: number; running_mean: number }[]
    by_line: { line: string; platform_mean: number; running_mean: number; running_share: number }[]
    days: number
    into_terminal: { elsewhere_median_lost: number; median_lost: number; stretches: number }
    lost_minutes_per_day: number
    per_visit: { offpeak_mean: number; offpeak_median: number; peak_mean: number; peak_median: number }
    platforms: number
    running_share: number
    stretches: number
    top_tenth_share: number
  }
}
