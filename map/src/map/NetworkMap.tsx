import { type ReactNode, useEffect, useRef, useState } from 'react'
import {
  AttributionControl, type ExpressionSpecification, type GeoJSONSource, Map as MapLibreMap, NavigationControl, setWorkerUrl, type StyleSpecification,
} from 'maplibre-gl'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import type { Feature, FeatureCollection } from 'geojson'
import 'maplibre-gl/dist/maplibre-gl.css'
import { basemap } from './basemap'
import { columnRing } from '../data/geometry'
import { isInterchange, METRICS, type MetricId, type PreparedNetwork, RELIABILITY_COLORS, UNCLUSTERED_COLOR } from '../data/network'
import { bandOf, DELAY_BANDS, type Vehicle } from '../data/replay'
import type { LonLat } from '../types'

// MapLibre computes its worker's URL at run time, which bundlers cannot see. Let
// Vite bundle the worker (with the chunk it imports) and hand MapLibre the URL.
setWorkerUrl(workerUrl)

export type Mode = 'network' | 'replay' | 'lost'
export type Selection = { id: string; kind: 'station' | 'trip' | 'place' }

/** The "time lost" view's places, each with its share `t` (0-1) of the worst one. */
export type LostLayer = {
  platforms: { coords: LonLat; id: string; t: number }[]
  stretches: { coords: LonLat[]; id: string; t: number }[]
}

type Props = {
  /** What to show on hover for a station, train or place; nothing when it returns null. */
  describe?: (target: Selection) => ReactNode
  /** Changes when the space around the map changes (a new view): the network is framed again. */
  frame?: string
  /** Scroll-wheel zoom; off where the map illustrates a page being scrolled. */
  scrollZoom?: boolean
  /** Fly here: a station, or any point (a stretch of track's middle). */
  focus?: { center?: LonLat; stationId?: string }
  /** A route to bring forward, dimming the rest (from hovering a results chart). */
  highlightLine?: string
  /** Stations named by an MBTA alert in effect at the replay's time. */
  incidentStations: string[]
  lost?: LostLayer
  metric: MetricId
  mode: Mode
  network: PreparedNetwork
  onSelect: (selection: Selection | null) => void
  rideTripId?: string
  selection: Selection | null
  vehicles: Vehicle[]
}

const COLUMN_MAX_METRES = 1_600
/** Stations under an MBTA alert: distinct from the lateness ramp and the selection amber. */
const INCIDENT_COLOR = '#ff7a59'
const PITCH: Record<Mode, number> = { network: 52, replay: 40, lost: 48 }
const CLICKABLE = ['stations', 'interchanges', 'station-columns', 'trains', 'train-columns', 'lost-stretches', 'lost-platforms']
/** Time lost moving: one blue ramp, dim to bright (validated against the basemap as the panel's pair). */
const LOST_MOVING = ['interpolate', ['linear'], ['get', 't'], 0, '#1f3656', 0.25, '#2a78d6', 0.6, '#6da7ec', 1, '#cde2fb'] as ExpressionSpecification
const LOST_STANDING = '#c98500'
const collection = (features: Feature[]): FeatureCollection => ({ features, type: 'FeatureCollection' })
const EMPTY = collection([])
const width = (low: number, high: number): ExpressionSpecification => ['interpolate', ['linear'], ['zoom'], 10, low, 14, high]
const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

function stationColumns(network: PreparedNetwork, metric: MetricId) {
  const { value } = METRICS[metric]
  const max = Math.max(...network.stations.map((station) => value(station) ?? 0))
  return collection(network.stations.flatMap((station) => {
    const v = value(station)
    if (v === null || max <= 0) return []
    return [{
      geometry: { coordinates: [columnRing(station.coords, 110)], type: 'Polygon' },
      properties: {
        color: RELIABILITY_COLORS[station.reliability ?? ''] ?? UNCLUSTERED_COLOR,
        height: 15 + (Math.max(0, v) / max) * COLUMN_MAX_METRES,
        id: station.id,
        kind: 'station',
      },
      type: 'Feature',
    }]
  }))
}

function trainCollections(network: PreparedNetwork, vehicles: Vehicle[], selectedTripId?: string) {
  const points: Feature[] = []
  const columns: Feature[] = []
  for (const { coords, lateness, trip } of vehicles) {
    const properties = { band: DELAY_BANDS[bandOf(lateness)].color, id: trip.id, kind: 'trip' }
    points.push({
      geometry: { coordinates: coords, type: 'Point' },
      properties: { ...properties, line: network.lineById.get(trip.line)?.color, selected: trip.id === selectedTripId },
      type: 'Feature',
    })
    // Columns rise with lateness, so the colour is never the only cue.
    columns.push({
      geometry: { coordinates: [columnRing(coords, 140)], type: 'Polygon' },
      properties: { ...properties, height: 40 + (Math.min(Math.max(lateness, 0), 1_200) / 1_200) * 1_100 },
      type: 'Feature',
    })
  }
  return { columns: collection(columns), points: collection(points) }
}

function addLayers(map: MapLibreMap, network: PreparedNetwork) {
  map.addSource('lines', {
    data: collection(network.patterns.map((pattern) => ({
      geometry: { coordinates: pattern.coords, type: 'LineString' },
      properties: { color: network.lineById.get(pattern.line)?.color, line: pattern.line },
      type: 'Feature',
    }))),
    type: 'geojson',
  })
  map.addSource('stations', {
    data: collection(network.stations.map((station) => ({
      geometry: { coordinates: station.coords, type: 'Point' },
      properties: { id: station.id, interchange: isInterchange(station), kind: 'station', name: station.name },
      type: 'Feature',
    }))),
    type: 'geojson',
  })
  for (const id of ['station-columns', 'trains', 'train-columns', 'selection', 'incidents', 'lost-stretches', 'lost-platforms']) {
    map.addSource(id, { data: EMPTY, type: 'geojson' })
  }

  const interchange: ExpressionSpecification = ['==', ['get', 'interchange'], true]
  const layers: Parameters<MapLibreMap['addLayer']>[0][] = [
    {
      id: 'buildings', minzoom: 12, source: 'openmaptiles', 'source-layer': 'building', type: 'fill-extrusion',
      paint: {
        'fill-extrusion-base': ['coalesce', ['get', 'render_min_height'], 0],
        'fill-extrusion-color': '#302d26',
        'fill-extrusion-height': ['coalesce', ['get', 'render_height'], ['*', ['coalesce', ['get', 'levels'], 2], 3]],
        'fill-extrusion-opacity': 0,
      },
    },
    { id: 'line-casing', layout: { 'line-cap': 'round', 'line-join': 'round' }, paint: { 'line-color': '#100f0d', 'line-width': width(5, 10) }, source: 'lines', type: 'line' },
    { id: 'line-core', layout: { 'line-cap': 'round', 'line-join': 'round' }, paint: { 'line-color': ['get', 'color'], 'line-width': width(2.6, 6) }, source: 'lines', type: 'line' },
    {
      filter: ['==', ['get', 'selected'], true], id: 'lost-stretch-selected', layout: { 'line-cap': 'round', 'line-join': 'round' }, source: 'lost-stretches', type: 'line',
      paint: { 'line-color': '#f0b64e', 'line-width': ['interpolate', ['linear'], ['zoom'], 10, ['+', 6, ['*', ['get', 't'], 6]], 14, ['+', 10, ['*', ['get', 't'], 12]]] },
    },
    {
      id: 'lost-stretches', layout: { 'line-cap': 'round', 'line-join': 'round' }, source: 'lost-stretches', type: 'line',
      paint: { 'line-color': LOST_MOVING, 'line-width': ['interpolate', ['linear'], ['zoom'], 10, ['+', 2, ['*', ['get', 't'], 6]], 14, ['+', 4, ['*', ['get', 't'], 12]]] },
    },
    { filter: ['!', interchange], id: 'stations', paint: { 'circle-color': '#f5eee3', 'circle-radius': width(2, 5), 'circle-stroke-color': '#171613', 'circle-stroke-width': 1 }, source: 'stations', type: 'circle' },
    { filter: interchange, id: 'interchanges', paint: { 'circle-color': '#f5eee3', 'circle-radius': width(3.2, 8), 'circle-stroke-color': '#171613', 'circle-stroke-width': 2.4 }, source: 'stations', type: 'circle' },
    { id: 'incident-stations', paint: { 'circle-color': 'rgba(255, 122, 89, 0.18)', 'circle-radius': width(7, 16), 'circle-stroke-color': INCIDENT_COLOR, 'circle-stroke-width': width(1.5, 2.5) }, source: 'incidents', type: 'circle' },
    { id: 'station-columns', paint: { 'fill-extrusion-color': ['get', 'color'], 'fill-extrusion-height': ['get', 'height'], 'fill-extrusion-opacity': 0.86 }, source: 'station-columns', type: 'fill-extrusion' },
    { id: 'lost-platforms', paint: { 'fill-extrusion-color': LOST_STANDING, 'fill-extrusion-height': ['get', 'height'], 'fill-extrusion-opacity': 0.88 }, source: 'lost-platforms', type: 'fill-extrusion' },
    { id: 'train-columns', paint: { 'fill-extrusion-color': ['get', 'band'], 'fill-extrusion-height': ['get', 'height'], 'fill-extrusion-opacity': 0.8 }, source: 'train-columns', type: 'fill-extrusion' },
    { id: 'trains', paint: { 'circle-color': ['get', 'band'], 'circle-radius': width(3.2, 6.5), 'circle-stroke-color': ['get', 'line'], 'circle-stroke-width': width(1.6, 3) }, source: 'trains', type: 'circle' },
    { filter: ['==', ['get', 'selected'], true], id: 'train-selection', paint: { 'circle-color': 'rgba(0,0,0,0)', 'circle-radius': width(8, 13), 'circle-stroke-color': '#f0b64e', 'circle-stroke-width': 2 }, source: 'trains', type: 'circle' },
    { id: 'station-selection', paint: { 'circle-color': 'rgba(0,0,0,0)', 'circle-radius': width(8, 14), 'circle-stroke-color': '#f0b64e', 'circle-stroke-width': 2 }, source: 'selection', type: 'circle' },
    {
      filter: interchange, id: 'interchange-labels', minzoom: 10.5, source: 'stations', type: 'symbol',
      layout: { 'text-field': ['get', 'name'], 'text-font': ['Noto Sans Regular'], 'text-radial-offset': 0.9, 'text-size': width(10, 12), 'text-variable-anchor': ['right', 'left', 'top', 'bottom'] },
      paint: { 'text-color': '#ded4c6', 'text-halo-color': '#171613', 'text-halo-width': 1.4 },
    },
    {
      filter: ['!', interchange], id: 'station-labels', minzoom: 13, source: 'stations', type: 'symbol',
      layout: { 'text-field': ['get', 'name'], 'text-font': ['Noto Sans Regular'], 'text-radial-offset': 0.8, 'text-size': 11, 'text-variable-anchor': ['right', 'left', 'top', 'bottom'] },
      paint: { 'text-color': '#b9ae9f', 'text-halo-color': '#171613', 'text-halo-width': 1.2 },
    },
  ]
  for (const layer of layers) map.addLayer(layer)
}

function overview(map: MapLibreMap, network: PreparedNetwork, mode: Mode) {
  const compact = map.getContainer().clientWidth <= 560
  const pitch = PITCH[mode]
  // The panel is docked beside the map, so only the replay's transport bar overlaps it.
  // Tilting pushes the near (southern) end of the network down the frame, so the
  // bottom keeps extra room: Braintree must stay in view. On wide screens the
  // replay's transport bar also lies over the map; stacked, it sits below it.
  const overlaid = mode === 'replay' && !window.matchMedia('(max-width: 820px)').matches
  const bottom = overlaid ? 170 : compact ? 40 : 90
  const camera = map.cameraForBounds(network.bounds, {
    bearing: -14,
    padding: compact ? { bottom, left: 16, right: 16, top: 16 } : { bottom, left: 40, right: 64, top: 24 },
  })
  if (camera?.zoom === undefined) return
  // The fit is computed flat; tilting foreshortens the tall network, so win back a little of it.
  const zoom = camera.zoom + 0.25 * Math.log2(1 / Math.cos((pitch * Math.PI) / 180))
  // Zooming out further than a little past the whole network only shows empty country.
  map.setMinZoom(Math.max(8, zoom - 0.8))
  map.easeTo({ ...camera, duration: reducedMotion() ? 0 : 900, pitch, zoom })
}

/** The network's bounds with room around them: panning stops before the map runs out of subway. */
function panLimits([[west, south], [east, north]]: [LonLat, LonLat]): [LonLat, LonLat] {
  const [dx, dy] = [(east - west) * 0.35, (north - south) * 0.35]
  return [[west - dx, south - dy], [east + dx, north + dy]]
}

const source = (map: MapLibreMap, id: string) => map.getSource(id) as GeoJSONSource

export function NetworkMap({ describe, focus, frame, highlightLine, incidentStations, lost, metric, mode, network, onSelect, rideTripId, scrollZoom = true, selection, vehicles }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [hover, setHover] = useState<{ target: Selection; x: number; y: number } | null>(null)
  const onSelectRef = useRef(onSelect)
  const networkRef = useRef(network)
  const rodeRef = useRef<string | undefined>(undefined)
  onSelectRef.current = onSelect

  useEffect(() => {
    const container = containerRef.current!
    const initialNetwork = networkRef.current
    const map = new MapLibreMap({
      attributionControl: false,
      bounds: initialNetwork.bounds,
      container,
      maxBounds: panLimits(initialNetwork.bounds),
      maxPitch: 70,
      maxZoom: 17,
      minZoom: 9,
      renderWorldCopies: false,
      style: basemap as StyleSpecification,
    })
    mapRef.current = map
    map.addControl(new NavigationControl({ visualizePitch: true }), 'bottom-right')
    map.addControl(new AttributionControl({ compact: true, customAttribution: 'Data: MBTA · Basemap © OpenStreetMap contributors' }), 'bottom-right')

    // Hit-test a padded box so small markers are easy to click and tap.
    const hit = ({ x, y }: { x: number; y: number }) => map.queryRenderedFeatures([[x - 6, y - 6], [x + 6, y + 6]], { layers: CLICKABLE })[0]
    map.on('click', (event) => {
      const feature = hit(event.point)
      onSelectRef.current(feature ? { id: String(feature.properties.id), kind: feature.properties.kind } : null)
    })
    map.on('mousemove', (event) => {
      const feature = hit(event.point)
      map.getCanvas().style.cursor = feature ? 'pointer' : ''
      setHover(feature ? { target: { id: String(feature.properties.id), kind: feature.properties.kind }, x: event.point.x, y: event.point.y } : null)
    })
    map.on('mouseout', () => setHover(null))
    map.on('movestart', () => setHover(null))
    // 'style.load', not 'load': the network needs the style, not every basemap tile.
    map.once('style.load', () => {
      addLayers(map, initialNetwork)
      setLoaded(true)
    })
    // The compact attribution opens expanded on wide maps; start it folded behind its (i).
    map.once('load', () => container.querySelector('.maplibregl-ctrl-attrib')?.classList.remove('maplibregl-compact-show'))
    // The panel beside the map changes width between views; keep the canvas in step.
    const resize = new ResizeObserver(() => map.resize())
    resize.observe(container)
    return () => {
      resize.disconnect()
      map.remove()
      mapRef.current = null
    }
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map) return
    source(map, 'station-columns').setData(mode === 'network' ? stationColumns(network, metric) : EMPTY)
    // In a replay the trains are the subject, and in "time lost" the track is:
    // recede the stations so neither is mistaken for one.
    for (const id of ['stations', 'interchanges']) {
      map.setPaintProperty(id, 'circle-opacity', mode === 'network' ? 1 : 0.35)
      map.setPaintProperty(id, 'circle-stroke-opacity', mode === 'network' ? 1 : 0.35)
    }
  }, [loaded, metric, mode, network])

  const selectedPlace = selection?.kind === 'place' ? selection.id : undefined
  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map) return
    const show = mode === 'lost' && lost
    source(map, 'lost-stretches').setData(show ? collection(lost.stretches.map(({ coords, id, t }) => ({
      geometry: { coordinates: coords, type: 'LineString' }, properties: { id, kind: 'place', selected: id === selectedPlace, t }, type: 'Feature',
    }))) : EMPTY)
    source(map, 'lost-platforms').setData(show ? collection(lost.platforms.map(({ coords, id, t }) => ({
      geometry: { coordinates: [columnRing(coords, 110)], type: 'Polygon' }, properties: { height: 15 + t * COLUMN_MAX_METRES, id, kind: 'place' }, type: 'Feature',
    }))) : EMPTY)
  }, [loaded, lost, mode, selectedPlace])

  const selectedTripId = selection?.kind === 'trip' ? selection.id : undefined
  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map) return
    const { columns, points } = trainCollections(network, vehicles, selectedTripId)
    source(map, 'trains').setData(points)
    source(map, 'train-columns').setData(columns)
    map.getContainer().dataset.trains = String(vehicles.length)
  }, [loaded, network, selectedTripId, vehicles])

  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map) return
    source(map, 'incidents').setData(collection(incidentStations.flatMap((id): Feature[] => {
      const station = network.stationById.get(id)
      return station ? [{ geometry: { coordinates: station.coords, type: 'Point' }, properties: {}, type: 'Feature' }] : []
    })))
  }, [incidentStations, loaded, network])

  const selectedStation = selection?.kind === 'station' ? network.stationById.get(selection.id) : undefined
  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map) return
    source(map, 'selection').setData(selectedStation
      ? collection([{ geometry: { coordinates: selectedStation.coords, type: 'Point' }, properties: {}, type: 'Feature' }])
      : EMPTY)
  }, [loaded, selectedStation])

  // Frame the whole network on first load and whenever the view changes. A new
  // view can move the divider, so wait for the slide to finish before framing.
  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map || rodeRef.current) return undefined
    const timer = window.setTimeout(() => overview(map, network, mode), 300)
    return () => window.clearTimeout(timer)
  }, [frame, loaded, mode, network])

  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map) return
    if (scrollZoom) map.scrollZoom.enable()
    else map.scrollZoom.disable()
  }, [loaded, scrollZoom])

  useEffect(() => {
    const map = mapRef.current
    const center = focus?.center ?? (focus?.stationId ? network.stationById.get(focus.stationId)?.coords : undefined)
    if (!loaded || !map || !center) return
    map.easeTo({ center, duration: reducedMotion() ? 0 : 700, zoom: Math.max(13.5, map.getZoom()) })
  }, [focus, loaded, network])

  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map || rideTripId) return
    // In "time lost" the coloured track replaces the line colours.
    map.setPaintProperty('line-core', 'line-opacity', mode === 'lost' ? 0.12
      : highlightLine ? ['case', ['==', ['get', 'line'], highlightLine], 1, 0.18] : 1)
  }, [highlightLine, loaded, mode, rideTripId])

  // Ride: fly down to the train once, then keep it centred as it moves.
  const ridden = rideTripId ? vehicles.find((vehicle) => vehicle.trip.id === rideTripId) : undefined
  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map) return
    const started = rideTripId !== rodeRef.current
    rodeRef.current = rideTripId
    if (started) {
      const line = ridden?.trip.line
      map.setPaintProperty('line-core', 'line-opacity', line ? ['case', ['==', ['get', 'line'], line], 1, 0.2] : 1)
      map.setPaintProperty('buildings', 'fill-extrusion-opacity', rideTripId ? 0.5 : 0)
      if (!rideTripId) overview(map, network, mode)
      else if (ridden) map.easeTo({ center: ridden.coords, duration: reducedMotion() ? 0 : 900, pitch: 62, zoom: 14.5 })
    } else if (ridden && !map.isMoving()) {
      map.jumpTo({ center: ridden.coords })
    }
  }, [loaded, mode, network, rideTripId, ridden])

  const tip = hover && describe ? describe(hover.target) : null
  return (
    <section aria-label="Map of the MBTA subway" className="metro-map">
      <div className="metro-map-canvas" data-state={loaded ? 'ready' : 'loading'} ref={containerRef} />
      {tip && hover ? (
        <div className="map-tooltip" role="status" style={{
          // Keep the card inside the map: flip it to the pointer's left near the right edge.
          left: hover.x + 260 > (containerRef.current?.clientWidth ?? Infinity) ? hover.x - 254 : hover.x + 14,
          top: hover.y + 14,
        }}>{tip}</div>
      ) : null}
    </section>
  )
}
