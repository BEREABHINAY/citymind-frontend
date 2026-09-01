# CityMind — ML/DL/DIP/RL Smart City Project
## Implementation notes for all 6 requirements

This folder contains **working, runnable code** for the two ML-heavy pieces
(traffic/fire classification, DQN routing). Points 3–6 are lighter-weight
(state machines / geolocation / naming), so they're specified precisely below
as drop-in JavaScript for your existing `city__11_.html`, ready to wire into
the React app's state.

---

### 1. Traffic & Fire classification — `classifier.py`
**DIP (classical CV) + ML pipeline**, no camera hardware needed, trained on a
**real public dataset** (not synthetic):

- **Traffic images**: `dense_traffic` / `sparse_traffic` classes from
  [OlafenwaMoses/Traffic-Net](https://github.com/OlafenwaMoses/Traffic-Net) (a
  published dataset of real road-camera photos).
- **Fire images**: the `fire` class from Traffic-Net, plus extra real fire
  photos from
  [cair/Fire-Detection-Image-Dataset](https://github.com/cair/Fire-Detection-Image-Dataset).
  Negatives (`no_fire`) use Traffic-Net's `accident` class (a deliberately
  hard negative — red brake lights and red car bodies look fire-colored to a
  naive classifier, which is exactly the case a real system must not
  misfire on) plus extra road scenes.
- Feature extraction: Canny edge density + adaptive-threshold vehicle-blob
  contour counting + color-std clutter measure for traffic → RandomForest;
  HSV fire-color masking + connected-component **area ratio** for fire →
  RandomForest.
- **Honest, held-out accuracy on real photos: 66.7% traffic / 82.5% fire.**
  (Real-world images are much harder than a synthetic set — this is the
  number to report, and it's a legitimate empirical result you can discuss
  and try to improve in your paper, e.g. by adding more training images or
  moving to a CNN.)
- Run `python3 generate_dataset.py` once if `real_dataset/` isn't already
  present (it re-downloads Traffic-Net + cair fire images from GitHub), then
  `python3 classifier.py` — it trains both models and picks 4 random real
  images to classify, exactly as you asked.
- The web app's **Vision AI** tab embeds a curated sample of these same real
  photos and runs genuine client-side DIP (Sobel edge density, HSV
  fire-color-band pixel ratio) directly on the pixel data in the browser —
  it shows the predicted label next to the real ground-truth label so you
  can see where it's right and where it's wrong, which is far more credible
  in a demo/viva than a "black box always says yes."

**For an even stronger dataset for your final submission:** UA-DETRAC or the
Kaggle "Traffic Vehicles Object Detection" set gives more traffic images and
finer density levels (light/medium/heavy instead of just light/heavy); the
D-Fire dataset (21,000 images) or FireNet give a much larger fire set. Swap
`TRAFFIC_DIR`/`FIRE_DIR` in `classifier.py` to point at them — nothing else
needs to change. If you want a stronger DL angle for the journal paper
(recommended), swap the RandomForest head for a small CNN (transfer learning
on MobileNetV2) fed the raw image instead of hand-engineered features — keep
the DIP preprocessing (HSV mask / edge map) as an *input channel* to the
CNN so you can claim a genuine DIP+DL hybrid, which is a stronger paper
contribution than either alone.

### 2. Per-traveller DQN routing — `dqn_routing.py`
Every user gets a start point + destination. The trained agent walks the
learned policy and returns the path, factoring in **distance, time, and live
traffic** (traffic weighted heaviest in the reward: `-(0.4·dist + 0.4·time +
6.0·traffic)` per road segment, +20 bonus on arrival).
Run `python3 dqn_routing.py` — trains on 11 named intersections / 26 road
segments in ~15s and prints 3 sample trips with full route, distance, and
traffic-adjusted ETA. In production the `live_traffic` dict is populated
live from `classifier.py`'s traffic-level output per road-camera feed.

### 3. Garbage truck tracker (available / not available)
Simple 2-state finite state machine per truck, broadcast to all users:

```js
// each truck object in React state
{ id: "GT-01", status: "available" | "en_route", lat, lng,
  currentStop: "Industrial Dump Yard Road", nextStop: "Market Street",
  etaMinutes: 6 }

// user-facing: "nearest truck" = the AVAILABLE truck with the
// shortest DQN-computed route (reuse dqn_routing.find_route) from the
// truck's location to the user's location.
function nearestTruck(userLoc, trucks) {
  return trucks
    .filter(t => t.status === "available")
    .map(t => ({ ...t, route: findRoute(t.location, userLoc) }))
    .sort((a, b) => a.route.estimated_time_min - b.route.estimated_time_min)[0];
}
```
Toggle status to `"en_route"` the moment a truck is dispatched to a
collection point; flip back to `"available"` on completion. Show a live ETA
countdown on the user's map pin — this reuses the *same* DQN router from
point 2, which is a nice unifying detail for your report ("one routing
engine, three use-cases: commuters, garbage trucks, emergency responders").

### 4. Named locations (drop into your city grid / dataset)
Intersections / roads: **MG Road Junction, Tech Park Circle, Riverside
Bridge, Market Street, Old Town Square, Industrial Dump Yard Road**
Hospitals: **Central Hospital, Sunrise Multispecialty Hospital**
IT / commercial: **Silicon Heights IT Park, Tech Park Circle Business Hub**
Apartments: **Lakeview Apartments, Greenfield Apartments, Sunrise
Apartments, Riverside Residency**
Utility: **Industrial Dump Yard Road** (garbage depot), **Riverside Bridge**
(chokepoint — good for blockage demo), **Old Town Square** (high-traffic hub)

These exact names are already wired into `dqn_routing.py`'s graph so the
route-planning demo and your UI can share one source of truth.

### 5. Emergency button — location tracking
```js
function onEmergencyClick() {
  if (!navigator.geolocation) { alert("Geolocation not supported"); return; }
  navigator.geolocation.getCurrentPosition(
    pos => {
      const emergency = {
        id: crypto.randomUUID(),
        lat: pos.coords.latitude,
        lng: pos.coords.longitude,
        timestamp: Date.now(),
        status: "dispatched"
      };
      // 1. drop a pulsing marker on the 3D map at this location
      // 2. find nearest available responder using the same DQN router
      //    (reuse nearestTruck-style logic against an "emergency units" list)
      // 3. push to Event Log: "🚨 Emergency reported near <nearest named location>"
      dispatchEmergencyUnit(emergency);
    },
    err => alert("Location permission denied — cannot dispatch help without it."),
    { enableHighAccuracy: true, timeout: 8000 }
  );
}
```
Reverse-map `lat/lng` to the nearest named location (simple nearest-neighbor
over your location list) so the Event Log reads "Emergency near Old Town
Square" instead of raw coordinates — much better for your demo video.

### 6. Blockage detection (combines points 1 + 2)
A road segment is flagged **BLOCKED** when the classifier pipeline reports
either: `fire == "Yes"` on that segment's camera feed, OR `traffic ==
"heavy"` sustained for N consecutive frames/ticks. Blocked segments get their
edge weight set to `∞` (or a very large traffic penalty) in the routing
graph, so the DQN router **automatically reroutes** every in-progress
traveller and every garbage truck around it — no separate "blockage" system
needed, it's just a live edge-weight update feeding the same graph:

```python
def update_blockages(graph_edges, classifier_results):
    for (a, b), result in classifier_results.items():
        if result.get("fire") == "Yes" or result.get("traffic_level") == "heavy":
            live_traffic[(a, b)] = 1.0   # max penalty -> DQN routes around it
            blocked_segments.add((a, b))
        else:
            blocked_segments.discard((a, b))
```
This is a good "systems integration" paragraph for your paper: show a
before/after route diagram when a segment gets blocked mid-simulation.

---

## Suggested project narrative for the journal paper
1. **Perception layer** — DIP + CNN classify traffic density and fire from
   camera frames (points 1).
2. **Decision layer** — DQN agent consumes perception output as live edge
   weights and computes optimal routes for commuters, garbage trucks, and
   emergency responders (points 2, 3, 5, 6) — **one shared routing engine,
   three applications** is your novelty claim.
3. **Deployment layer** — the existing 3D CityMind dashboard visualizes all
   of the above live. This is Figure/Demo material, not the contribution.

Report: classifier accuracy/precision/recall tables, DQN training reward
curve, and a comparison of DQN-routed ETA vs. plain-shortest-path (Dijkstra,
traffic-blind) ETA under 3 traffic scenarios — that comparison table is your
strongest quantitative result.

## Running the real backend (true integration)

The web app now talks to an actual Python backend (`server.py`) that serves
the real trained models — not just a JS approximation:

```bash
pip install flask flask-cors joblib opencv-python scikit-learn --break-system-packages
python3 server.py
```

Then open `citymind_ai_enhanced.html` in your browser as normal. The AI/ML
Systems panel auto-detects the backend at `http://localhost:5000` and shows
a green "Connected to Python backend" status. From that point on:
- **Vision AI tab** calls `/api/samples` — real RandomForest predictions on
  real photos, served fresh from `classifier.py`'s trained model.
- **Route Planner / Garbage Tracker tabs** call `/api/route` and
  `/api/garbage/nearest` — the actual trained DQN policy from
  `dqn_routing.py`, with a Dijkstra safety net so a blocked road always
  triggers a real reroute (the raw DQN rollout doesn't always find the
  detour on its own — this hybrid is honest, standard practice for
  deploying RL, and worth a paragraph in your report).

If the server isn't running, the page still works standalone (JS
fallback) — it just tells you it's using the approximation instead of the
real model, rather than failing silently.
