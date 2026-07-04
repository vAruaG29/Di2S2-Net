import maplibregl, { LngLatBoundsLike, Map as MLMap } from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";

import { api, DatasetDetail, LayerInfo } from "../api/client";
import { BASE_STYLES, inferGeom, LAYER_STYLES } from "../constants";
import type { LayerState } from "../types";

export interface CameraInfo {
  lat: number;
  lng: number;
  zoom: number;
}

interface Props {
  dataset: DatasetDetail | null;
  layerState: Record<string, LayerState>;
  baseStyle?: keyof typeof BASE_STYLES;

  /** Hide basemap layers (used when stacking two maps for swipe). */
  showBasemap?: boolean;
  /** Hide the aerial COG raster. */
  showImagery?: boolean;
  /** Master toggle for vector predictions. */
  showPredictions?: boolean;
  /** Master toggle for the ground-truth overlay. Pulls per-class
   *  GeoJSON from `/api/datasets/{name}/gt/{layer}` and styles every
   *  class with the GT-green palette. Independent of `showPredictions`
   *  so the two stacks can coexist when swipe is off. */
  showGroundTruth?: boolean;
  /** Per-class GT visibility/opacity (same shape as `layerState` but
   *  for the green overlay). Optional — when omitted every GT class
   *  defaults to visible at 0.7 opacity. */
  gtLayerState?: Record<string, LayerState>;

  /** Called once when the MapLibre instance is ready. */
  onMapReady?: (m: MLMap) => void;
  /** Camera updates (used by FooterBar). */
  onCameraChange?: (c: CameraInfo) => void;
  /** Tile/layer loading status (used by the loading bar). */
  onLoadingChange?: (s: LoadStatus) => void;
  /** Notifies how many prediction-vector GeoJSON fetches are in flight
   *  + how many total are wanted. Used by App to surface a "loading
   *  N/M layers" hint while predictions stream in. */
  onVectorLoadingChange?: (inFlight: number, total: number) => void;

  /** Suppress the built-in navigation control (used by the swipe pair). */
  noControls?: boolean;
  /** When false, MapLibre's pan/zoom/wheel handlers are all disabled
   *  (used for the passive mirror in SwipeMap). */
  interactive?: boolean;
}

export interface LoadStatus {
  imageryLoading: boolean;
  tilesInFlight: number;
  layersInFlight: number;
  layersTotal: number;
}

const IMG_RASTER_SRC = "dinov3:img";
const IMG_RASTER_LYR = "dinov3:img:raster";
const PREVIEW_SRC    = "dinov3:img:preview";
const PREVIEW_LYR    = "dinov3:img:preview:layer";

/**
 * Headless MapView — renders the MapLibre canvas + all of its sources
 * and layers; reports loading/camera state via callbacks. Floats no
 * chrome (legend / coord chip / etc) of its own; the parent decides
 * what overlays to put around it.
 */
export function MapView({
  dataset, layerState, baseStyle = "dark",
  showBasemap = true, showImagery = true, showPredictions = true,
  showGroundTruth = false, gtLayerState,
  onMapReady, onCameraChange, onLoadingChange, onVectorLoadingChange,
  noControls = false, interactive = true,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MLMap | null>(null);
  const [styleReady, setStyleReady] = useState(false);
  // Track which dataset the camera was last fit to so a style swap
  // doesn't yank the user back to the dataset extent every time.
  const lastFitRef = useRef<string | null>(null);

  // ── Map lifecycle ───────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: BASE_STYLES[baseStyle],
      center: [78.9629, 20.5937],
      zoom: 4.5,
      // 20 is plenty for any aerial COG (sub-cm GSD already at z18-19).
      // Cap protects against the "stops zooming + central rectangle"
      // artefact users hit past the COG's last overview level — once
      // we're past native resolution the tile pyramid runs out and
      // MapLibre falls back to weird intermediate states.
      maxZoom: 20,
      minZoom: 0,
      attributionControl: { compact: true },
      // Hard-disable interaction for the passive mirror in SwipeMap.
      interactive,
    });
    if (!noControls) {
      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
      map.addControl(new maplibregl.ScaleControl({ maxWidth: 100, unit: "metric" }), "bottom-left");
    }
    map.on("load", () => setStyleReady(true));

    const emitCam = () => {
      const c = map.getCenter();
      onCameraChange?.({ lat: c.lat, lng: c.lng, zoom: map.getZoom() });
    };
    map.on("mousemove", (e) => {
      onCameraChange?.({ lat: e.lngLat.lat, lng: e.lngLat.lng, zoom: map.getZoom() });
    });
    map.on("zoom", emitCam);
    map.on("move", emitCam);

    // Reliable raster-tile load tracking via polling.
    const poll = setInterval(() => {
      if (!mapRef.current) return;
      const m = mapRef.current;
      try {
        const allLoaded = m.areTilesLoaded();
        let pending = 0;
        // @ts-ignore — internal but stable
        const caches = (m.style as any)?.sourceCaches ?? {};
        for (const id of Object.keys(caches)) {
          if (!id.startsWith("dinov3:img")) continue;
          const cache = caches[id];
          const tiles = cache?._tiles ?? cache?.tiles ?? {};
          for (const tk of Object.keys(tiles)) {
            const t = tiles[tk];
            if (!t) continue;
            if (t.state !== "loaded" && t.state !== "errored") pending += 1;
          }
        }
        onLoadingChange?.({
          imageryLoading: !allLoaded || pending > 0,
          tilesInFlight: pending,
          layersInFlight: 0, // VectorLayers reports separately via its own bump
          layersTotal: 0,
        });
      } catch { /* swap race */ }
    }, 250);
    (map as any)._loadPollInterval = poll;
    mapRef.current = map;
    onMapReady?.(map);

    return () => {
      const iv = (map as any)._loadPollInterval;
      if (iv) clearInterval(iv);
      map.remove();
      mapRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Base style swap ─────────────────────────────────────────────────
  //
  // Two non-obvious things have to be right here, otherwise our COG +
  // vector layers vanish after the swap:
  //
  //  (a) `setStyle(..., { diff: false })` — by default MapLibre tries
  //      to *diff* the old and new styles and migrate user-added
  //      sources/layers across. The migration is brittle and we've
  //      seen it leave our COG + GPKG sources in a half-state where
  //      the source still appears in `map.getSource()` but renders
  //      nothing. Forcing `diff: false` gives us a clean slate that
  //      our re-add effects below can fully recreate.
  //
  //  (b) Wait for `isStyleLoaded()`, not the first `styledata` event.
  //      Big basemap styles fire `styledata` many times during a swap
  //      (once per sub-layer); flipping `styleReady = true` on the
  //      first one races against the rest of the style. We re-arm
  //      `styledata` until `isStyleLoaded()` is finally true. A 2 s
  //      timeout protects us if the event never fires (e.g. a network
  //      hiccup on the basemap fetch).
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    setStyleReady(false);
    map.setStyle(BASE_STYLES[baseStyle], { diff: false });

    let done = false;
    const finalize = () => {
      if (done || !mapRef.current) return;
      if (map.isStyleLoaded()) {
        done = true;
        setStyleReady(true);
      } else {
        map.once("styledata", finalize);
      }
    };
    finalize();
    map.once("idle", finalize);
    const watchdog = setTimeout(() => {
      if (!done) { done = true; setStyleReady(true); }
    }, 2000);

    return () => { done = true; clearTimeout(watchdog); };
  }, [baseStyle]);

  // ── Force-reorder our layers above the basemap after every swap.
  //    Even with diff:false some basemaps continue to add symbol /
  //    label layers a few ticks after isStyleLoaded() returns true, so
  //    we sweep a couple of times.
  useEffect(() => {
    if (!styleReady) return;
    const map = mapRef.current;
    if (!map) return;
    const reorder = () => {
      try {
        // Desired stack (bottom → top):
        //   basemap → dinov3:img:* (COG) → dinov3:gt:* (ground truth)
        //                              → dinov3:v:* (predictions, topmost)
        // We move each band in that order — every moveLayer() with no
        // `beforeId` pushes the layer to the TOP of the stack, so the
        // LAST band moved is the one that ends up on top.
        const ours = map.getStyle().layers ?? [];
        const order = ["dinov3:img:", "dinov3:gt:", "dinov3:v:"];
        for (const prefix of order) {
          for (const lyr of ours) {
            if (lyr.id.startsWith(prefix)) {
              try { map.moveLayer(lyr.id); } catch { /* race */ }
            }
          }
        }
      } catch { /* style swap race */ }
    };
    reorder();
    const t1 = setTimeout(reorder, 200);
    const t2 = setTimeout(reorder, 800);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [styleReady, baseStyle]);

  // ── Toggle basemap visibility via opacity sweep ─────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !styleReady) return;
    try {
      for (const lyr of map.getStyle().layers ?? []) {
        if (lyr.id.startsWith("dinov3:")) continue;       // ours, skip
        const t = lyr.type;
        try {
          if (t === "raster") {
            map.setPaintProperty(lyr.id, "raster-opacity", showBasemap ? 1 : 0);
          } else if (t === "fill") {
            map.setPaintProperty(lyr.id, "fill-opacity", showBasemap ? 1 : 0);
          } else if (t === "line") {
            map.setPaintProperty(lyr.id, "line-opacity", showBasemap ? 1 : 0);
          } else if (t === "symbol") {
            map.setPaintProperty(lyr.id, "text-opacity", showBasemap ? 1 : 0);
            map.setPaintProperty(lyr.id, "icon-opacity", showBasemap ? 1 : 0);
          } else if (t === "background") {
            map.setPaintProperty(lyr.id, "background-opacity", showBasemap ? 1 : 0);
          } else if (t === "circle") {
            map.setPaintProperty(lyr.id, "circle-opacity", showBasemap ? 1 : 0);
          }
        } catch { /* layer type may not support that paint key */ }
      }
    } catch { /* style swap race */ }
  }, [showBasemap, styleReady, baseStyle]);

  // ── Aerial COG (raster preview + tile pyramid) ──────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !styleReady) return;

    [PREVIEW_LYR, IMG_RASTER_LYR].forEach((id) => {
      if (map.getLayer(id)) map.removeLayer(id);
    });
    [PREVIEW_SRC, IMG_RASTER_SRC].forEach((id) => {
      if (map.getSource(id)) map.removeSource(id);
    });

    if (!dataset?.source_cog || !dataset.bounds_wgs84) return;
    const [w, s, e, n] = dataset.bounds_wgs84;

    if (dataset.preview_url) {
      map.addSource(PREVIEW_SRC, {
        type: "image",
        url: dataset.preview_url,
        coordinates: [[w, n], [e, n], [e, s], [w, s]],
      });
      map.addLayer({
        id: PREVIEW_LYR,
        type: "raster",
        source: PREVIEW_SRC,
        paint: {
          // Fade out earlier and hard-clamp to zero at z13 — past
          // that, the preview is just a low-res rectangle floating
          // inside the high-res raster and reads as a visual
          // artefact. Real tile pyramid carries from there.
          "raster-opacity": [
            "interpolate", ["linear"], ["zoom"],
            10, 1.0, 12, 1.0, 13, 0.0,
          ] as any,
          "raster-resampling": "linear",
        },
      });
    }

    const tilesTemplate =
      dataset.tiles_url_template ??
      dataset.tilejson_url ??
      api.tilejsonUrl(dataset.source_cog);
    map.addSource(IMG_RASTER_SRC, {
      type: "raster",
      tiles: [tilesTemplate],
      tileSize: 256,
      bounds: [w, s, e, n],
      minzoom: 0,
      // Match the map's overall maxZoom — past z20 we just oversample,
      // and the dataset's overview pyramid almost certainly bottoms
      // out before then. Capping here lets MapLibre stop asking for
      // ever-deeper tiles that come back as black/empty pixels.
      maxzoom: 20,
    });
    map.addLayer({
      id: IMG_RASTER_LYR,
      type: "raster",
      source: IMG_RASTER_SRC,
      paint: { "raster-opacity": showImagery ? 1 : 0, "raster-resampling": "linear" },
    });

    // Only fit-bounds when the dataset actually changes — a basemap
    // swap shouldn't yank the user back to the full extent.
    //
    // Skip fitBounds entirely on non-interactive maps (the SwipeMap
    // mirror): they're camera-slaved to the primary, so any auto-fit
    // they do races against the sync and leaves them at a different
    // extent until the user nudges the zoom.
    if (interactive && dataset?.name && lastFitRef.current !== dataset.name) {
      map.fitBounds([[w, s], [e, n]] as LngLatBoundsLike, {
        padding: 40, duration: 600,
      });
      lastFitRef.current = dataset.name;
    }

    try {
      const layers = map.getStyle().layers ?? [];
      for (const lyr of layers) {
        if (lyr.id.startsWith("dinov3:v:")) map.moveLayer(lyr.id);
      }
    } catch { /* race */ }
  }, [dataset?.name, dataset?.source_cog, dataset?.preview_url, dataset?.tilejson_url, styleReady]);

  // ── Toggle aerial COG visibility ────────────────────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !styleReady) return;
    try {
      if (map.getLayer(IMG_RASTER_LYR))
        map.setPaintProperty(IMG_RASTER_LYR, "raster-opacity", showImagery ? 1 : 0);
      if (map.getLayer(PREVIEW_LYR))
        map.setPaintProperty(PREVIEW_LYR, "raster-opacity", showImagery ? 1 : 0);
    } catch { /* race */ }
  }, [showImagery, styleReady, dataset?.name]);

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="absolute inset-0" />
      {dataset && (
        <VectorLayers
          map={mapRef}
          layers={dataset.layers}
          datasetName={dataset.name}
          gpkgMtime={dataset.gpkg_mtime ?? null}
          state={layerState}
          styleReady={styleReady}
          enabled={showPredictions}
          onLoadingChange={onVectorLoadingChange}
        />
      )}
      {dataset && (
        <GroundTruthLayers
          map={mapRef}
          gtLayers={dataset.gt_layers ?? []}
          gtState={gtLayerState ?? {}}
          datasetName={dataset.name}
          styleReady={styleReady}
          // Empty gt_layers (e.g. Test datasets) → effectively disabled,
          // but we keep the component mounted so its cleanup runs when
          // the dataset changes or GT is toggled off.
          enabled={showGroundTruth && (dataset.gt_layers?.length ?? 0) > 0}
        />
      )}
    </div>
  );
}

/** Six "headline" classes that come on by default — kept in sync with
 *  the seed in App.tsx so the fallback below behaves the same. */
const DEFAULT_VISIBLE_LAYERS = new Set([
  "Built_Up_Area_type", "Road", "Water_Body",
  "Bridge", "Railway", "Utility_Poly",
]);

/**
 * Lazy-load GPKG layers as the user toggles them. When `enabled` is
 * false the whole stack is hidden (master toggle from the LeftPanel).
 */
function VectorLayers({
  map, layers, datasetName, gpkgMtime, state, styleReady, enabled, onLoadingChange,
}: {
  map: React.MutableRefObject<MLMap | null>;
  layers: LayerInfo[];
  datasetName: string;
  /** Unix mtime of the resolved GPKG. Included in the in-memory cache
   *  key so a re-run on the SAME dataset name (force=True, fresh GPKG
   *  on disk) does not serve the previous run's stale GeoJSON. */
  gpkgMtime: number | null;
  state: Record<string, LayerState>;
  styleReady: boolean;
  enabled: boolean;
  onLoadingChange?: (inFlight: number, total: number) => void;
}) {
  const cacheRef = useRef<Map<string, GeoJSON.FeatureCollection>>(new Map());
  const inFlightRef = useRef(0);

  const bump = (delta: number, total: number) => {
    inFlightRef.current = Math.max(0, inFlightRef.current + delta);
    onLoadingChange?.(inFlightRef.current, total);
  };

  // `state` may not yet have entries for the new dataset's layers when
  // the user switches between datasets — App.tsx's seed runs *after*
  // the dataset prop has changed, so for one render the lookup is
  // undefined. Without the fallback the cleanup at the bottom of the
  // effect would nuke every source on the map. The fallback treats an
  // unseeded layer as "default-visible if it's in the headline set",
  // matching App's seed, so the transition is invisible to the user.
  const visible = useMemo(
    () => layers.filter((l) => {
      if (!enabled) return false;
      const s = state[l.name];
      if (s !== undefined) return s.visible;
      return DEFAULT_VISIBLE_LAYERS.has(l.name);
    }),
    [layers, state, enabled],
  );

  // STABLE key keyed only on which layer names are visible — opacity
  // slider moves and other state churn don't change this. Without
  // this the source-management effect would re-fire on every
  // opacity change and cancel its own in-flight Promise.all,
  // leaving the map empty until the user toggled something.
  const visibleKey = useMemo(
    () => visible.map((l) => l.name).sort().join("|"),
    [visible],
  );

  // ── Source / layer add (idempotent) + dataset-switch cleanup ─────
  useEffect(() => {
    const m = map.current;
    if (!m || !styleReady) return;
    let cancelled = false;

    (async () => {
      // Parallel fetch across every currently-visible layer. Track
      // in-flight count so the LoadingBar can show "loading N/M
      // layers".
      const fetches = visible.map(async (l) => {
        // Cache key includes the GPKG mtime so a re-run that regenerates
        // the GeoPackage at the same path serves fresh features instead
        // of the previous run's cached FeatureCollection.
        const cacheKey = `${datasetName}::${l.name}::${gpkgMtime ?? 0}`;
        let fc = cacheRef.current.get(cacheKey);
        if (!fc) {
          bump(+1, visible.length);
          try {
            fc = await api.layerGeoJSON(datasetName, l.name);
            cacheRef.current.set(cacheKey, fc);
          } catch (e) {
            console.warn("Failed to load layer", l.name, e);
            bump(-1, visible.length);
            return { l, fc: null as GeoJSON.FeatureCollection | null };
          }
          bump(-1, visible.length);
        }
        return { l, fc };
      });
      const fetched = await Promise.all(fetches);
      if (cancelled || !m.getStyle()) return;

      // Hand each result to MapLibre. addSource / addLayer are
      // idempotent (guarded by `getSource` / `getLayer`); paint
      // updates from opacity changes live in the separate effect
      // below. When a source already exists (re-run on same dataset)
      // we `setData` to swap in the fresh FeatureCollection so the
      // map reflects the NEW inference, not the previous run.
      for (const { l, fc } of fetched) {
        if (!fc) continue;
        const srcId = sourceId(datasetName, l.name);
        const existing = m.getSource(srcId);
        if (!existing) {
          m.addSource(srcId, { type: "geojson", data: fc });
        } else if ("setData" in existing && typeof (existing as maplibregl.GeoJSONSource).setData === "function") {
          (existing as maplibregl.GeoJSONSource).setData(fc);
        }
        const style = LAYER_STYLES[l.name];
        const colour = style?.colour ?? l.colour ?? "#ff5600";
        const geom = style?.geom ?? inferGeom(l.name);
        const op = state[l.name]?.opacity ?? 0.7;
        for (const sub of subLayerSpecs(l.name, geom)) {
          if (!m.getLayer(sub.id)) {
            m.addLayer({
              id: sub.id,
              source: srcId,
              ...sub.spec(colour, op),
            } as maplibregl.LayerSpecification);
          }
          try { m.moveLayer(sub.id); } catch { /* race */ }
        }
      }

      // Drop sources that no longer belong (dataset switch or user
      // hid a class).
      const wantedSrcIds = new Set(visible.map((l) => sourceId(datasetName, l.name)));
      const allSrcIds = Object.keys(m.getStyle().sources).filter((s) => s.startsWith("dinov3:v:"));
      for (const sid of allSrcIds) {
        if (!wantedSrcIds.has(sid)) {
          for (const lid of layerIdsForSource(m, sid)) m.removeLayer(lid);
          m.removeSource(sid);
        }
      }
    })();

    return () => { cancelled = true; };
    // NOTE: `state` is intentionally absent from deps. Opacity changes
    // go through the separate effect below; including `state` here
    // makes the effect cancel its own in-flight fetches every time a
    // slider moves. `gpkgMtime` IS included — re-running inference
    // bumps the GPKG mtime, which re-fires this effect and swaps the
    // map sources to the freshly-fetched features.
  }, [visibleKey, datasetName, gpkgMtime, styleReady, map]);     // eslint-disable-line

  // ── Opacity updater (cheap; fires on every state change) ─────────
  useEffect(() => {
    const m = map.current;
    if (!m || !styleReady) return;
    for (const l of visible) {
      const style = LAYER_STYLES[l.name];
      const colour = style?.colour ?? l.colour ?? "#ff5600";
      const geom = style?.geom ?? inferGeom(l.name);
      const op = state[l.name]?.opacity ?? 0.7;
      for (const sub of subLayerSpecs(l.name, geom)) {
        if (!m.getLayer(sub.id)) continue;
        for (const [k, v] of Object.entries(sub.spec(colour, op).paint ?? {})) {
          try { m.setPaintProperty(sub.id, k, v as never); } catch { /* race */ }
        }
      }
    }
  }, [state, visible, styleReady, map]);              // eslint-disable-line

  return null;
}

function sourceId(ds: string, layer: string) {
  return `dinov3:v:${ds}:${layer}`;
}

const GT_COLOUR = "#00B894";   // green palette dedicated to ground truth.

/** Ground-truth overlay. Same pattern as VectorLayers, but pulls from
 *  `/api/datasets/.../gt/...` (which vectorises masks on demand) and
 *  paints every class in the GT-green palette so the user can tell GT
 *  apart from predictions at a glance.
 *
 *  `enabled` is the master switch (from LeftFloat's GT row). When on,
 *  per-class visibility comes from `gtState` — each class name is
 *  toggled independently by the GtClassesFloat panel.  When off, every
 *  GT source is removed from the map. */
function GroundTruthLayers({
  map, gtLayers, gtState, datasetName, styleReady, enabled,
}: {
  map: React.MutableRefObject<MLMap | null>;
  gtLayers: string[];
  gtState: Record<string, LayerState>;
  datasetName: string;
  styleReady: boolean;
  enabled: boolean;
}) {
  const cacheRef = useRef<Map<string, GeoJSON.FeatureCollection>>(new Map());

  // Classes the user wants visible right now. Defaults each class to
  // visible (matching the App-level seed) so a still-unseeded class
  // during a dataset switch doesn't briefly disappear.
  const visible = useMemo(
    () => (enabled
      ? gtLayers.filter((n) => gtState[n]?.visible ?? true)
      : []),
    [enabled, gtLayers, gtState],
  );

  // Stable key — see VectorLayers above. Opacity changes shouldn't
  // re-fire the fetch effect (and cancel in-flight requests).
  const visibleKey = useMemo(() => visible.slice().sort().join("|"), [visible]);

  // ── Source / layer add (idempotent) + master-toggle cleanup ──────
  useEffect(() => {
    const m = map.current;
    if (!m || !styleReady) return;
    let cancelled = false;

    (async () => {
      const fetches = visible.map(async (layerName) => {
        const cacheKey = `${datasetName}::${layerName}`;
        let fc = cacheRef.current.get(cacheKey);
        if (!fc) {
          try {
            fc = await api.gtLayerGeoJSON(datasetName, layerName);
            cacheRef.current.set(cacheKey, fc);
          } catch (e) {
            console.warn("GT fetch failed", layerName, e);
            return { layerName, fc: null as GeoJSON.FeatureCollection | null };
          }
        }
        return { layerName, fc };
      });
      const fetched = await Promise.all(fetches);
      if (cancelled || !m.getStyle()) return;

      for (const { layerName, fc } of fetched) {
        if (!fc) continue;
        const srcId = gtSourceId(datasetName, layerName);
        if (!m.getSource(srcId)) {
          m.addSource(srcId, { type: "geojson", data: fc });
        }
        const geom = inferGeom(layerName);
        const op = gtState[layerName]?.opacity ?? 0.7;
        for (const sub of gtSubLayerSpecs(layerName, geom)) {
          if (!m.getLayer(sub.id)) {
            m.addLayer({
              id: sub.id,
              source: srcId,
              ...sub.spec(op),
            } as maplibregl.LayerSpecification);
          }
          try { m.moveLayer(sub.id); } catch { /* swap race */ }
        }
      }

      // Cleanup GT sources that no longer belong.
      const wantedSrcIds = new Set(visible.map((l) => gtSourceId(datasetName, l)));
      const allSrcIds = Object.keys(m.getStyle().sources)
        .filter((s) => s.startsWith("dinov3:gt:"));
      for (const sid of allSrcIds) {
        if (!wantedSrcIds.has(sid)) {
          for (const lid of layerIdsForSource(m, sid)) m.removeLayer(lid);
          m.removeSource(sid);
        }
      }
    })();

    return () => { cancelled = true; };
  }, [visibleKey, datasetName, styleReady, map]);     // eslint-disable-line

  // ── Opacity updater (cheap; fires on every gtState change) ───────
  useEffect(() => {
    const m = map.current;
    if (!m || !styleReady) return;
    for (const layerName of visible) {
      const geom = inferGeom(layerName);
      const op = gtState[layerName]?.opacity ?? 0.7;
      for (const sub of gtSubLayerSpecs(layerName, geom)) {
        if (!m.getLayer(sub.id)) continue;
        for (const [k, v] of Object.entries(sub.spec(op).paint ?? {})) {
          try { m.setPaintProperty(sub.id, k, v as never); } catch { /* race */ }
        }
      }
    }
  }, [gtState, visible, styleReady, map]);            // eslint-disable-line

  return null;
}

function gtSourceId(ds: string, layer: string) {
  return `dinov3:gt:${ds}:${layer}`;
}

/** GT styling: green fill at ~35% × opacity factor + a dashed green
 *  outline. The opacity factor is the per-class slider value from
 *  GtClassesFloat (0..1) — same shape as the predictions side. */
function gtSubLayerSpecs(name: string, geom: "fill" | "line" | "circle") {
  if (geom === "fill") {
    return [
      {
        id: `dinov3:gt:${name}:fill`,
        spec: (op: number) => ({
          type: "fill",
          paint: { "fill-color": GT_COLOUR, "fill-opacity": op * 0.5 },
        }),
      },
      {
        id: `dinov3:gt:${name}:outline`,
        spec: (op: number) => ({
          type: "line",
          paint: {
            "line-color": GT_COLOUR,
            "line-width": 1.5,
            "line-opacity": Math.min(1, op + 0.2),
            "line-dasharray": [2, 1.5],
          },
        }),
      },
    ];
  }
  if (geom === "line") {
    return [{
      id: `dinov3:gt:${name}:line`,
      spec: (op: number) => ({
        type: "line",
        paint: {
          "line-color": GT_COLOUR,
          "line-width": 2,
          "line-opacity": op,
          "line-dasharray": [2, 1.5],
        },
      }),
    }];
  }
  return [{
    id: `dinov3:gt:${name}:circle`,
    spec: (op: number) => ({
      type: "circle",
      paint: {
        "circle-color": GT_COLOUR,
        "circle-radius": 4,
        "circle-opacity": op,
        "circle-stroke-color": "#151515",
        "circle-stroke-width": 1,
      },
    }),
  }];
}

function layerIdsForSource(map: MLMap, sid: string): string[] {
  return map.getStyle().layers
    .filter((l) => "source" in l && l.source === sid)
    .map((l) => l.id);
}

function subLayerSpecs(name: string, geom: "fill" | "line" | "circle") {
  if (geom === "fill") {
    return [
      {
        id: `dinov3:v:${name}:fill`,
        spec: (c: string, op: number) => ({
          type: "fill",
          paint: { "fill-color": c, "fill-opacity": op * 0.55 },
        }),
      },
      {
        id: `dinov3:v:${name}:outline`,
        spec: (c: string, op: number) => ({
          type: "line",
          paint: {
            "line-color": c,
            "line-width": 1.2,
            "line-opacity": Math.min(1, op + 0.15),
          },
        }),
      },
    ];
  }
  if (geom === "line") {
    return [{
      id: `dinov3:v:${name}:line`,
      spec: (c: string, op: number) => ({
        type: "line",
        paint: { "line-color": c, "line-width": 2, "line-opacity": op },
      }),
    }];
  }
  return [{
    id: `dinov3:v:${name}:circle`,
    spec: (c: string, op: number) => ({
      type: "circle",
      paint: {
        "circle-color": c,
        "circle-radius": 4,
        "circle-opacity": op,
        "circle-stroke-color": "#151515",
        "circle-stroke-width": 1,
      },
    }),
  }];
}
