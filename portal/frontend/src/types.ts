/** Shared types used across the portal frontend. */

/** Per-class GPKG layer state (drives MapView's vector layers). */
export interface LayerState {
  visible: boolean;
  opacity: number;       // 0..1
}

/** Which Train/Test/Upload tab is open. `null` = files panel collapsed. */
export type LeftTab = "train" | "test" | "upload";

/** Visibility flags for the four numbered layers in the layer stack. */
export interface StackState {
  basemap: boolean;
  imagery: boolean;
  predictions: boolean;
  groundTruth: boolean;
}
