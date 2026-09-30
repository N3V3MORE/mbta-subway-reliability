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

export type ReplaySummary = { late: number; maeModel: number; maePersistence: number; stops: number; trips: number }

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

export type Replay = { date: string; label: string; summary: ReplaySummary; trips: Trip[] }
