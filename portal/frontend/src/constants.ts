/**
 * Visual constants for layers and base maps.
 *
 * The colour map is a fallback — at runtime we fetch /api/config and
 * prefer whatever colours the bundle's data_prep.yaml carries, so the
 * frontend and CLI outputs stay in sync.
 */

export type GeomKind = "fill" | "line" | "circle";

export interface LayerStyleSpec {
  label: string;
  colour: string;
  geom: GeomKind;
}

/** Default per-layer styling. Matches data_prep.yaml visualization.class_colors. */
export const LAYER_STYLES: Record<string, LayerStyleSpec> = {
  Built_Up_Area_type: { label: "Built-Up Area",    colour: "#FF0000", geom: "fill"   },
  Road:                { label: "Road",            colour: "#FFFF00", geom: "fill"   },
  Road_Centre_Line:    { label: "Road centre line",colour: "#FFA500", geom: "line"   },
  Water_Body:          { label: "Water body",      colour: "#0000FF", geom: "fill"   },
  Water_Body_Line:     { label: "Water body (line)", colour: "#00BFFF", geom: "line" },
  Waterbody_Point:     { label: "Water body (pt)", colour: "#87CEEB", geom: "circle" },
  Utility_Poly:        { label: "Utility (area)",  colour: "#00FF00", geom: "fill"   },
  Utility:             { label: "Utility (pt)",    colour: "#00FF00", geom: "circle" },
  Bridge:              { label: "Bridge",          colour: "#00FFFF", geom: "fill"   },
  Railway:             { label: "Railway",         colour: "#FF00FF", geom: "line"   },
};

/**
 * Heuristic — pick a geometry kind from a layer name if we have no
 * other info. Polygons are the safest default.
 */
export function inferGeom(layer: string): GeomKind {
  if (layer.endsWith("_Line") || layer === "Railway") return "line";
  if (layer.endsWith("_Point") || layer === "Utility") return "circle";
  return "fill";
}

export const FALLBACK_COLOUR = "#22d3ee";

/** Open-licensed base maps. Voyager + dark variants from CARTO. */
export const BASE_STYLES = {
  positron: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
  dark:     "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  voyager:  "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
};
