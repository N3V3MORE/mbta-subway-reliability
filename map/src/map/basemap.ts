import type { ExpressionSpecification, LayerSpecification, StyleSpecification } from 'maplibre-gl'
import base from './basemap.json'

/**
 * The dark basemap: water, parks and place names from `basemap.json`, plus the
 * streets, rail, runways and building footprints a real city map has. Streets
 * stay a few steps off the ground colour so the subway lines remain the subject.
 */
const INK = {
  building: '#1e1c19',
  minor: '#23211d',
  primary: '#2f2b25',
  rail: '#2b2823',
  roadLabel: '#7a7266',
  runway: '#24221e',
  secondary: '#29261f',
  motorway: '#38332b',
}

const byZoom = (...stops: number[]): ExpressionSpecification => ['interpolate', ['exponential', 1.5], ['zoom'], ...stops]
const roadClass = (...classes: string[]): ExpressionSpecification => ['match', ['get', 'class'], classes, true, false]
const notTunnel: ExpressionSpecification = ['!=', ['get', 'brunnel'], 'tunnel']

const road = (id: string, classes: string[], color: string, width: ExpressionSpecification, minzoom = 0): LayerSpecification => ({
  filter: ['all', roadClass(...classes), notTunnel],
  id,
  layout: { 'line-cap': 'round', 'line-join': 'round' },
  minzoom,
  paint: { 'line-color': color, 'line-width': width },
  source: 'openmaptiles',
  'source-layer': 'transportation',
  type: 'line',
})

const streets: LayerSpecification[] = [
  {
    id: 'aeroway-runway', filter: ['==', ['get', 'class'], 'runway'], minzoom: 10, source: 'openmaptiles', 'source-layer': 'aeroway', type: 'line',
    paint: { 'line-color': INK.runway, 'line-width': byZoom(10, 2, 15, 40) },
  },
  {
    id: 'building', minzoom: 14, source: 'openmaptiles', 'source-layer': 'building', type: 'fill',
    paint: { 'fill-color': INK.building, 'fill-opacity': ['interpolate', ['linear'], ['zoom'], 14, 0, 15, 1] },
  },
  road('road-minor', ['minor', 'service'], INK.minor, byZoom(12, 0.4, 17, 9), 12),
  road('road-secondary', ['secondary', 'tertiary'], INK.secondary, byZoom(9, 0.4, 17, 12), 9),
  road('road-primary', ['primary', 'trunk'], INK.primary, byZoom(8, 0.5, 17, 16), 8),
  road('road-motorway', ['motorway'], INK.motorway, byZoom(6, 0.6, 17, 18), 6),
  {
    filter: ['all', roadClass('rail'), notTunnel], id: 'rail', minzoom: 10, source: 'openmaptiles', 'source-layer': 'transportation', type: 'line',
    paint: { 'line-color': INK.rail, 'line-dasharray': [3, 3], 'line-width': byZoom(10, 0.5, 17, 2) },
  },
  {
    id: 'road-label', minzoom: 14, source: 'openmaptiles', 'source-layer': 'transportation_name', type: 'symbol',
    filter: roadClass('primary', 'secondary', 'tertiary', 'trunk', 'minor'),
    layout: { 'symbol-placement': 'line', 'text-field': ['get', 'name'], 'text-font': ['Noto Sans Regular'], 'text-size': 10 },
    paint: { 'text-color': INK.roadLabel, 'text-halo-color': '#171613', 'text-halo-width': 1.2 },
  },
]

/** Small places appear only when zoomed in, so the overview is not a field of town names. */
const PLACE_MINZOOM: Record<string, number> = { place_other: 12.5, place_suburb: 12, place_town: 10.5, place_village: 12 }

const layers = (base.layers as LayerSpecification[]).map((layer) =>
  layer.id in PLACE_MINZOOM ? { ...layer, minzoom: PLACE_MINZOOM[layer.id] } : layer)
// Streets sit above water and parks and below every label.
const firstLabel = layers.findIndex((layer) => layer.type === 'symbol')

export const basemap: StyleSpecification = {
  ...(base as StyleSpecification),
  layers: [...layers.slice(0, firstLabel), ...streets, ...layers.slice(firstLabel)],
}
