import { useEffect, useRef, useState } from 'react'
import {
  AttributionControl, type ExpressionSpecification, type GeoJSONSource, Map as MapLibreMap, NavigationControl, setWorkerUrl, type StyleSpecification,
} from 'maplibre-gl'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import type { Feature, FeatureCollection } from 'geojson'
import 'maplibre-gl/dist/maplibre-gl.css'
import basemap from './basemap.json'
import { columnRing } from '../data/geometry'
import { isInterchange, METRICS, type MetricId, type PreparedNetwork, RELIABILITY_COLORS, UNCLUSTERED_COLOR } from '../data/network'
import { bandOf, DELAY_BANDS, type Vehicle } from '../data/replay'

// MapLibre computes its worker's URL at run time, which bundlers cannot see. Let
// Vite bundle the worker (with the chunk it imports) and hand MapLibre the URL.
setWorkerUrl(workerUrl)

export type Mode = 'network' | 'replay'
export type Selection = { id: string; kind: 'station' | 'trip' }

type Props = {
  focus?: { stationId: string }
  /** Stations named by an MBTA alert in effect at the replay's time. */
  incidentStations: string[]
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
const PITCH: Record<Mode, number> = { network: 52, replay: 40 }
const CLICKABLE = ['stations', 'interchanges', 'station-columns', 'trains', 'train-columns']
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
  for (const id of ['station-columns', 'trains', 'train-columns', 'selection', 'incidents']) map.addSource(id, { data: EMPTY, type: 'geojson' })

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
    { filter: ['!', interchange], id: 'stations', paint: { 'circle-color': '#f5eee3', 'circle-radius': width(2, 5), 'circle-stroke-color': '#171613', 'circle-stroke-width': 1 }, source: 'stations', type: 'circle' },
    { filter: interchange, id: 'interchanges', paint: { 'circle-color': '#f5eee3', 'circle-radius': width(3.2, 8), 'circle-stroke-color': '#171613', 'circle-stroke-width': 2.4 }, source: 'stations', type: 'circle' },
    { id: 'incident-stations', paint: { 'circle-color': 'rgba(255, 122, 89, 0.18)', 'circle-radius': width(7, 16), 'circle-stroke-color': INCIDENT_COLOR, 'circle-stroke-width': width(1.5, 2.5) }, source: 'incidents', type: 'circle' },
    { id: 'station-columns', paint: { 'fill-extrusion-color': ['get', 'color'], 'fill-extrusion-height': ['get', 'height'], 'fill-extrusion-opacity': 0.86 }, source: 'station-columns', type: 'fill-extrusion' },
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
  const compact = map.getContainer().clientWidth <= 760
  const pitch = PITCH[mode]
  const camera = map.cameraForBounds(network.bounds, {
    bearing: -14,
    padding: compact ? { bottom: 190, left: 20, right: 20, top: 150 } : { bottom: 150, left: 380, right: 120, top: 110 },
  })
  if (camera?.zoom === undefined) return
  // The fit is computed flat; tilting foreshortens the (tall) network, so zoom in
  // by most of that foreshortening to keep it filling the frame.
  const zoom = camera.zoom + 0.8 * Math.log2(1 / Math.cos((pitch * Math.PI) / 180))
  map.easeTo({ ...camera, duration: reducedMotion() ? 0 : 900, pitch, zoom })
}

const source = (map: MapLibreMap, id: string) => map.getSource(id) as GeoJSONSource

export function NetworkMap({ focus, incidentStations, metric, mode, network, onSelect, rideTripId, selection, vehicles }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const [loaded, setLoaded] = useState(false)
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
      maxPitch: 70,
      maxZoom: 17,
      minZoom: 9,
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
    map.on('mousemove', (event) => { map.getCanvas().style.cursor = hit(event.point) ? 'pointer' : '' })
    // 'style.load', not 'load': the network needs the style, not every basemap tile.
    map.once('style.load', () => {
      addLayers(map, initialNetwork)
      setLoaded(true)
    })
    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!loaded || !map) return
    source(map, 'station-columns').setData(mode === 'network' ? stationColumns(network, metric) : EMPTY)
    // In a replay the trains are the subject; recede the stations so an on-time
    // (near-white) train is not mistaken for one.
    for (const id of ['stations', 'interchanges']) {
      map.setPaintProperty(id, 'circle-opacity', mode === 'replay' ? 0.35 : 1)
      map.setPaintProperty(id, 'circle-stroke-opacity', mode === 'replay' ? 0.35 : 1)
    }
  }, [loaded, metric, mode, network])

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

  // Frame the whole network on first load and whenever the view changes.
  useEffect(() => {
    const map = mapRef.current
    if (loaded && map && !rodeRef.current) overview(map, network, mode)
  }, [loaded, mode, network])

  useEffect(() => {
    const map = mapRef.current
    const station = focus && network.stationById.get(focus.stationId)
    if (!loaded || !map || !station) return
    const compact = map.getContainer().clientWidth <= 760
    map.easeTo({ center: station.coords, duration: reducedMotion() ? 0 : 700, offset: compact ? [0, -90] : [-120, 0], zoom: Math.max(13.5, map.getZoom()) })
  }, [focus, loaded, network])

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

  return <section aria-label="Map of the MBTA subway" className="metro-map"><div className="metro-map-canvas" data-state={loaded ? 'ready' : 'loading'} ref={containerRef} /></section>
}
