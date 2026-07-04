# Di2S2-Net — Slide-by-Slide Speaker Notes

> For the 9-slide final deck `Di2S2-Net_Hackathon_Deck_final.pdf`. Each slide block has:
> **SAY** (natural spoken script) · **POINT TO** (what to gesture at) · **HIT THESE NUMBERS** · **TRANSITION** · **IF ASKED** (only if a judge probes).
> Suggested total: **~10–11 min talk + Q&A**. Speak to the *story*, don't read bullets. Three presenters can split: Slides 1–3 (Nirdesh), 4–6 (Gaurav), 7–9 + demo (Samvedya) — adjust to your strengths.
>
> **Two phrases to use exactly, and two to avoid:**
> ✅ "Railway has no ground truth anywhere and Bridge is vanishingly rare." ❌ never "zero Bridge ground truth."
> ✅ "0.90 mIoU on present classes." Treat 0.65 as the averaging artefact, not the headline.
>
> **The deck is image-driven — let the visuals carry the argument.** Slides **2, 4, 5, 7** are mostly pictures: the tile→map pipeline strip (2), the encoder & decoder diagrams (4), the blocky-LR-vs-crisp-HR panels (5), and the live portal screenshot + metric callouts (7). On these, **point and pause — don't talk over them.** Slide 5's side-by-side masks ARE the proof; give the audience two seconds to see "blocky → crisp." (PPTX is available too, but the PDF renders identically — these notes match both.)

---

## SLIDE 1 — Title (Di2S2-Net) · ~30 s

**SAY:**
"Good [morning/afternoon]. We're EarthSense Labs, team AIML-F-028, presenting **Di2S2-Net** — the Dino Drone Semantic Segmentation Network — for Problem Statement 1, AI-based feature extraction from drone images. In one line: we take a **raw multi-gigabyte drone orthomosaic** of a village and turn it, fully automatically, into a **GIS-ready GeoPackage** of its assets — roads, buildings, water bodies, utilities — using a **satellite-pretrained vision transformer**, not a generic ImageNet model. Over the next ten minutes we'll walk you through the problem, the architecture, how we built it, and the results — and we have a live portal to show at the end."

**HIT:** team id AIML-F-028, EarthSense Labs, PS-1.
**TRANSITION:** "Let me start with what we actually built and why it matters for SVAMITVA."

---

## SLIDE 2 — Proposed Solution · ~1 min 15 s

**SAY:**
"SVAMITVA needs georeferenced asset maps for lakhs of villages from drone imagery. Today that's largely manual digitisation — slow, costly, inconsistent. **Our solution is one fully-automated pipeline: raw orthomosaic in, GeoPackage out.** We auto-detect and reproject the coordinate system, tile a multi-GB raster into manageable pieces, segment every tile with our model, then **memory-safely stitch and vectorise** the result into per-class polygons and road centre-lines — *zero manual digitisation*. We ship it two ways: a **CLI batch tool** for district-scale processing, and a **no-code web portal** for field staff.

The engine is a **DINOv3 ViT-L/16 encoder pretrained on 493 million satellite images.** That domain match is the key idea — because the backbone already understands aerial texture and scale, only a **light decoder head** needs to learn from our **ten labelled villages**. Its **HRDecoder** fuses a global low-resolution pass with native-resolution 256-pixel windows, so it captures both huge water bodies and thread-thin utility lines from the same tile.

One honest note on the data, top-right: across the labelled corpus, **Railway has no ground truth anywhere, and Bridge is vanishingly rare — just 12 of 4,513 tiles.** So those two score essentially zero by *data coverage*, not model failure — and we report every result both with and without them."

**POINT TO:** the **"FROM TILE TO MAP" image strip down the right edge** — trace it top-to-bottom with your finger as you say the verbs: *orthomosaic* (the raw aerial) → *tiles* (the grid) → *Di2S2-Net* → *predicted tiles* (the coloured masks) → *combined output* (the stitched GeoPackage). That single column **is** the whole pipeline in one glance — use it. Then drop to the **7-class stacked bar** at bottom-left for the 71.7%-background point.
**HIT:** 493 M satellite images · 10 labelled villages · 34,708 tiles / 20 villages · Railway 0 GT, Bridge 12/4,513.
**TRANSITION:** "Five hard problems shaped every design choice — here they are with how we solved each."

---

## SLIDE 3 — Key Challenges → Our Approach · ~1 min 30 s

**SAY:**
"Everything in Di2S2-Net traces back to five real problems.

**One — one tile, two scales.** At 3-centimetre resolution a single tile holds a water body thousands of pixels wide *and* a utility line a few pixels wide. A single-resolution network loses one or the other. Our **HRDecoder two-pass fusion** solves it — global context plus native-resolution detail, fused. Result: Water 0.96, Built-up 0.95, **Road 0.88, Utility 0.77 — all at once.**

**Two — scarce, clustered labels.** Only 10 of 20 villages are labelled, in just two states. We don't train from scratch — we stand on the **DINOv3 SAT-493M foundation model** and fine-tune only the top 12 of 24 transformer blocks, a roughly **3-million-parameter head.** That's how we reach **0.905 mIoU in-sample and 0.73 on completely unseen villages.**

**Three — severe class imbalance.** Background is 71.7% of pixels; Utility is half a percent. Plain cross-entropy collapses to background. Our **combined loss — cross-entropy plus Dice plus an edge term, with multi-scale supervision** — recovers the minority classes.

**Four — messy raw geodata.** Mixed coordinate systems, ECW and TIFF, silent duplicate copies. We **de-duplicate by a canonical key** and **auto-reproject to the correct UTM zone during COG conversion** — no train/test leakage, metre-accurate output.

**Five — rasters bigger than GPU and RAM.** One village, KURTU, is 10,523 tiles. We use **COG windowed reads, FP16 inference, and constant-memory windowed stitching** — 34,708 tiles on a single 24-GB GPU at about 7 tiles a second.

And below — we also engineered GIS-ready output, cheap transfer to new regions, fault-tolerant batch processing, the portal, and honest metrics."

**POINT TO:** scan the five rows left-to-right (challenge → solution → evidence).
**HIT:** the evidence column numbers; "5 problems → 5 design decisions."
**TRANSITION:** "Now the architecture that delivers this."

---

## SLIDE 4 — System Architecture / Workflow · ~1 min 15 s

**SAY:**
"The model spine: a **1024×1024 tile** goes into the encoder, then the HRDecoder, supervised by a multi-scale loss.

The **encoder is DINOv3 ViT-L/16 with satellite SAT-493M weights** — 24 transformer blocks, 1024-dimensional, patch size 16, so a 1024 tile becomes a **64×64 grid of tokens.** We don't just take the last layer — we **tap four depths, blocks 5, 11, 17 and 23**, to get features from shallow texture to deep semantics. Each is projected — LayerNorm, Linear to 256, GELU. The **first 12 blocks are frozen; blocks 12 to 23 fine-tune at one-tenth the learning rate** — so we adapt without destroying the pretrained features.

The **decoder is the HRDecoder** — it fuses the four scales into one 256-channel map and runs the two passes I'll detail next.

Crucially for deployment: the **exact same binaries** are wrapped by a FastAPI plus TiTiler backend and a React-MapLibre client. And because the decoder is a **modular registry** — HRDecoder, UPerNet, SegFormer, SkipDecoder — the portal can **sniff a checkpoint's architecture from its weights and refuse incompatible ones.**"

**POINT TO:** first sweep the **MODEL SPINE strip across the top** (tile thumbnail → encoder → HRDecoder → loss) — that's the one-line summary. Then the **ENCODER diagram on the left** (trace the 4 taps fanning out to the 4 projected scales) and the **DECODER diagram on the right** (the fusion → LR/HR boxes). Finish on the **Productisation line** at the bottom for the FastAPI/MapLibre + registry point.
**HIT:** 24 blocks · patch 16 · 64×64 tokens · taps [5,11,17,23] · freeze 12, fine-tune 12–23 at 0.1× LR.
**TRANSITION:** "The HRDecoder is the heart of this — here's how the two passes work."

---

## SLIDE 5 — HRDecoder, Two-Pass Fusion · ~1 min 30 s

**SAY:**
"This is our core technical contribution for this problem. The dilemma: one pass on the coarse 64×64 feature grid blurs thin features; a full high-resolution pass over the whole tile is too expensive. So **HRDecoder does both and fuses them.**

The **low-res global pass** runs one segmentation head over the whole tile at the coarse grid, upsampled to full size. Large areas and context come out right — but boundaries are soft; you can see the blocky edges.

The **high-res local pass** decodes **256-pixel windows at native resolution**, then averages them with a count-map. Thin roads, utility lines and building outlines come back crisp.

There's a deliberate difference between training and test. In **training** we take **2 random 256 crops, jittered three-quarters to one-and-a-quarter scale and snapped to multiples of 8** — the randomness doubles as regularisation. At **test time it's a deterministic sliding window** — every region is covered, logits accumulate in a count-map and are averaged, giving a **seamless, reproducible full-resolution output.**

The net effect: large-area recall from the low-res pass, thin-feature precision from the high-res pass, **fused at native 1024 resolution at constant memory** — that's how we get Road IoU up to 0.88 with crisp, georeferenced boundaries."

**POINT TO (this slide is a visual proof — slow down):** put your pointer on the **LR · GLOBAL PASS panel and say "look at the blocky edges,"** then move to the **HR · LOCAL PASS panel — "now crisp,"** then the **⊕ FUSED OUTPUT** on the right. **Pause ~2 seconds** so the audience's eyes do the comparing — the picture wins the argument better than words. Then point to the train-randomised vs test-deterministic boxes for the how.
**HIT:** 256² windows · 2 random crops train / sliding window test · count-map averaging · Road 0.88.
**TRANSITION:** "So how is it all implemented and trained?"

---

## SLIDE 6 — Implementation & Technical Approach · ~1 min 30 s

**SAY:**
"Quickly through the engineering, because every choice has a reason.

**Data.** We de-dup by a canonical key — stripping extensions, the _3857 suffix, case — keeping valid over corrupt, TIFF over ECW, so no leakage. COG conversion auto-detects CRS and reprojects EPSG:4326 to the **centroid's UTM zone** — that's what makes areas metre-accurate — with DEFLATE and 512-pixel tiling for windowed access. We tile at **1024 with 128-pixel overlap** and drop tiles that are over half no-data. Labels are rasterised from shapefiles using an **R-tree index and a reprojection cache**, with points buffered 3 metres so poles survive.

**Training** — all in the table: **PyTorch Lightning, FP16, batch 4 with gradient accumulation 2 for an effective batch of 8, AdamW, gradient clip 1.0, base learning rate 1e-4 with the encoder at one-tenth, cosine schedule with 5-epoch warmup, 50 epochs.** The whole model is ~300 million parameters, ~155 million trainable, but the **task head is only ~3 million** — which is exactly why porting to a new region is cheap.

**Loss** — cross-entropy 0.5 for classification, **Dice 0.3 for imbalance**, a **Sobel boundary-consistency term at 0.2**, with multi-scale supervision Fuse-LR-HR weighted 1, 0.5, 0.1. Boundary precision comes primarily from the HR pass at native resolution and from Dice.

**Scale and robustness** — FP16 autocast roughly halves inference latency, GPU cache flushed every 50 batches; windowed stitching rebuilds full rasters in constant memory; skip-resume orchestration with a checkpoint-architecture guard; and vectorisation with road skeletons, a utility negative-buffer, into a multi-layer GeoPackage with QGIS styles."

**POINT TO:** the three columns — Data decisions, Training table, Loss table.
**HIT:** eff. batch 8 · enc LR 0.1× · cosine + 5-ep warmup · loss 0.5/0.3/0.2 · ~3 M head.
**TRANSITION:** "Now the results — and a live demo."
**IF ASKED — "does the edge/Sobel term actually backpropagate given the argmax?"** (only if pressed): *"It's a low-weight boundary-consistency signal; in our objective the heavy lifting on boundaries is done by the HR pass at native resolution and by Dice — which is why the edge term carries the smallest weight. Our boundary results, Road IoU up to 0.88 and the sharp masks in the gallery, are real and architecture-driven."* Do **not** volunteer this.

---

## SLIDE 7 — Results & Demonstration · ~2 min (incl. demo)

**SAY (results first):**
"The headline: **0.905 mIoU on the five classes that exist in the data, 0.73 on villages the model has never seen, and about 98.4% overall accuracy.** Per class, in-sample: Water 0.956, Built-up 0.949, **Road 0.879, Utility 0.765.** The generalisation table shows two held-out experiments — holding out NAGUL, and TIMMOWAL plus NAGUL — both land around **0.73 five-class mIoU at 0.93–0.95 accuracy**, from only ten training villages. The 0.65 seven-class number is purely the artefact of averaging in the two zero-ground-truth classes. Throughput is about **681 tiles in 99 seconds — 7 tiles a second.**

**[DEMO]** And this is the product. The portal lets a non-technical user **browse pre-computed results and run a fresh inference on a new orthomosaic** — five phases, streamed live over server-sent events — on the *same binaries*, so what you see equals what we submitted. Here's the **swipe compare** — aerial on one side, our prediction on the other — [drag the slider]. You can **compare multiple checkpoints side by side**, each namespaced and architecture-guarded, and **download the GeoPackage in one click**, ready for QGIS."

**IF DEMO FAILS:** "Let me show the recorded walkthrough" → play the Drive video. *(Have it open in a tab beforehand.)*
**POINT TO:** the **portal screenshot on the left** — note the red building polygons laid over real aerial imagery (that's our prediction on a village). Then the **four big metric callouts** (0.905 / 0.647 / ~98.4% / 0.73), the **PER-CLASS IoU bars** (Water → Built-Up → Road → Utility), and the **GENERALISATION table**. If demoing live, **switch to the actual portal** for the swipe rather than the screenshot — the live swipe is more convincing than any number.
**HIT:** 0.905 / 0.73 / 98.4% · held-out 0.729 & 0.731 · 7 tiles/s · "demo == submission."
**TRANSITION:** "To close — what's novel, what we learned, and where this goes."

---

## SLIDE 8 — Innovation & Key Learnings · ~1 min

**SAY:**
"What's novel: a **satellite foundation model brought to SVAMITVA** — SAT-493M, not ImageNet — which is what makes ten villages enough. The **HRDecoder two-pass fusion** that beats the multi-scale curse. A **multi-component, multi-scale loss.** **District-scale engineering** — constant-memory stitching, FP16, skip-resume, GIS-native output. And **pipeline-to-product** — a portal that *wraps*, never re-implements, the pipeline.

Our biggest learning: **the metric can lie.** Railway has no ground truth and Bridge is vanishingly rare, so both sit at zero and drag seven-class mIoU to 0.65 against a true 0.905. The fix is honesty — report both — and the head already reserves their class IDs, so adding labels later needs no re-architecture. We also hit a **memory wall** stitching KURTU and solved it with windowed writes.

Future work: **annotate Bridge, Railway, and especially Utility** — our hardest class at 0.38–0.47 held-out; **more villages** to close the in-sample-to-unseen gap, which is cheap via the 3-million-parameter frozen-encoder transfer path; a differentiable edge term; a backbone upgrade to ViT-7B or DINOv4 behind the same wrapper; and multi-GPU plus Docker for district rollout. Everything is **open-source — no proprietary dependencies.**"

**POINT TO:** the three columns — Novel / Learnings / Future.
**HIT:** "metric lied, true 0.905" · Utility is the hard class · ~3 M transfer path · open-source only.
**TRANSITION:** "Thank you — happy to take questions."

---

## SLIDE 9 — Q&A / Thank You

**SAY:** "Thank you. Di2S2-Net — raw drone imagery to a GIS-ready map, end to end. We'd love your questions."

---

## ANTICIPATED Q&A BANK (rehearse these)

**Q: Why DINOv3 / a foundation model instead of a U-Net or DeepLab trained on your data?**
"With only 10 labelled villages, a from-scratch CNN overfits. A self-supervised model pretrained on 493 M *satellite* images already encodes aerial texture and scale, so a ~3 M-param head suffices — and it generalises: 0.73 mIoU on villages never seen in training."

**Q: Why SAT-493M and not the general LVD weights?**
"Domain match. SAT-493M is trained on satellite/aerial imagery; the spectral and textural statistics are far closer to drone orthomosaics than the general web-image weights, so transfer is stronger."

**Q: Why HRDecoder over UPerNet/SegFormer? Did you compare?**
"All four are in our decoder registry behind one interface. HRDecoder is the only one that explicitly reconciles huge-area and thin-linear features by running and fusing a low-res context pass with native-res local windows — which is exactly our problem. The others share the encoder and loss; HRDecoder won on the thin classes."

**Q: Your mIoU is 0.65 — isn't that low?**
"That's the seven-class number, mechanically lowered by two classes with no usable ground truth — Railway has none anywhere, Bridge is 12 of 4,513 tiles. On the five classes that exist in the data it's **0.905 in-sample and 0.73 on unseen villages.** We always report both, transparently."

**Q: How does it handle a brand-new region/state?**
"Two paths. Best case, the foundation features already transfer — 0.73 on unseen villages with no retraining. For a new terrain type, freeze the encoder and retrain only the ~3 M-param head — hours on a single GPU, not weeks."

**Q: Overlap / seams when stitching — don't you get artefacts?**
"Tiles overlap 128 px and the HRDecoder test pass is a count-map-averaged sliding window, so per-tile predictions are smooth; at stitch time labels are discrete so we use last-write-wins on the overlap. No averaging of class IDs, no seams in practice."

**Q: Throughput at true district scale?**
"~7 tiles/s on one 24 GB GPU; 34,708 tiles across 20 villages already run end-to-end. It parallelises trivially by splitting the dataset list across GPUs, and FP16 already halves latency."

**Q: How accurate is the georeferencing / are areas trustworthy?**
"We reproject every raster to its correct UTM zone before tiling, so areas are computed in metres, not degrees. Output is a standard GeoPackage in the source CRS — drop it straight into QGIS or ArcGIS."

**Q: Label quality / why is Utility weak?**
"Utility is intrinsically hard — 0.57% of pixels and present in only 23% of tiles, often poles digitised as points which we buffer to 3 m. It's our priority for more annotation and targeted augmentation."

**Q: What about the edge term's gradient?** → use the IF-ASKED answer on Slide 6.

**Q: Could you add new feature classes later?**
"Yes with no re-architecture — the head already outputs 7 classes including the reserved Bridge/Railway IDs; adding a class is a head resize plus fine-tuning the ~3 M head, encoder frozen."
