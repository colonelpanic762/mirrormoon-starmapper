# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Star Mapper is a scientific visualization tool for locating anomaly stars in the game MirrorMoon EP. It helps triangulate anomaly positions by simulating observatory skybox projections and computing cone intersections from multiple observations.

## Tech Stack

- **Python 3.11+**
- **numpy** - Coordinate transforms, linear algebra
- **scipy** - Geometric computations, spatial algorithms
- **matplotlib** - Quick prototyping, static 3D plots
- **PyOpenGL + pyglet** - Interactive 3D visualization

## Commands

```bash
# Create virtual environment (first time)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
source .venv/bin/activate  # if not already active
python main.py

# Run with specific star data
python main.py --data data/stars.json

# Visualize star map (interactive 3D)
python main.py --visualize

# Save visualization to file
python main.py --save data/star_map.png

# First-person observatory view (by-eye workflow, no screenshots).
# --visible lists the stars you can see in-game: known ones are highlighted and
# framed, unknown ones are recorded in stars.json as "seen_from" this observatory.
python main.py --observatory OBS1 --visible GIOVAN AUY/14 DAV21 AUT56

# Same, but hide the known anomaly star and report how far each mark was from it
python main.py --observatory OBS1 --visible GIOVAN AUY/14 DAV21 --check-against ROBERT

# One-time camera settings (saved in stars.json under "observatory_camera")
python main.py --tilt 13 --turn-time 4.5 --aspect 16:10

# Run tests
python -m pytest tests

# Multi-image stitching: open the marking tool on a saved session
# (markings are loaded from the file and saved after every click)
python main.py --stitch-session data/stitch-OBSERV.json

# Start a new stitching session from screenshots
python main.py --stitch IMG1 IMG2 --screenshot-observatory OBSERV --stitch-session data/stitch-NAME.json

# Solve a saved session without opening the GUI (repeatable; use this for debugging)
python main.py --stitch-session data/stitch-OBSERV.json --stitch-solve

# Solve and save the direction to a marked label as an observation for --triangulate
# (replaces the observation previously saved from the same session and label)
python main.py --stitch-session data/stitch-OBSERV.json --stitch-solve --stitch-save-observation ANOMALY

# Triangulate the anomaly from all stored observations
python main.py --triangulate
```

## Development Environment (Distrobox)

Claude Code runs from the Claude desktop app inside an **Ubuntu 24.04 distrobox** (Python 3.12), and the project's `.venv` is built there. Commands run directly in that container; there is no need for `distrobox enter`.

- Run Python with `.venv/bin/python` (or activate the venv first).
- System packages are installed with `sudo apt-get install` (passwordless inside the container).
- The interactive matplotlib tools need `python3-tk` from apt. `tkinter` cannot be installed with pip; without it matplotlib falls back to the non-interactive `agg` backend and no window opens.
- A venv is tied to the interpreter and path it was created with. If the repo moves or a different container is used, recreate it: `rm -rf .venv && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`

## Project Structure

```
starmapper/
├── coords.py           # Coordinate parsing, 9-digit ↔ (x,y,z) transforms, Star dataclass
├── visualization.py    # matplotlib-based 3D star map rendering
├── camera.py           # MoonCamera: first-person camera logic (turn/walk/twist, projection)
├── observatory_view.py # Interactive first-person observatory view (uses MoonCamera)
├── triangulation.py    # Observation cones, intersection calculations
├── navigation.py       # Ship navigation guidance and orientation calibration
├── calibration.py      # Camera model, single-screenshot calibration, projection helpers
├── screenshot_tool.py  # Interactive single-screenshot calibration tool
├── stitching.py        # Multi-image stitching solver, session save/load
└── stitching_tool.py   # Interactive multi-image marking tool
```

Stitching sessions are JSON files in `data/` (e.g. `data/stitch-OBSERV.json`) holding image paths and marked pixel positions. Known-star coordinates are looked up from `stars.json` on load.

## Coordinate System

Stars use a 9-digit coordinate string that encodes X, Y, Z positions:
- `123456789` → X=123, Y=456, Z=789
- Each axis ranges from 000-999

Star naming uses `XXX/##` format where letters map to coordinate ranges:
- A=000-099, E=100-199, I=200-299, U=300-399, Y=400-499
- H=500-599, J=600-699, L=700-799, T=800-899, X=900-999

## Planned Architecture

### Core Components
- **3D Star Map**: Renders all stars in the coordinate space with color coding
- **Observatory Mode**: First-person simulation matching in-game WASD controls (W/S orbit planet surface, A/D rotate view)
- **Triangulation Engine**: Projects uncertainty cones from observations, calculates intersection volumes

### Observation Input Workflow (M3/M4)
1. **Match the sky**: Recreate observatory view with reference stars and constellation lines, rotate to match in-game orientation
2. **First click**: Mark best guess of anomaly position (sets cone center/direction)
3. **Second click**: Set radius of uncertainty cone (confidence interval)
4. **Repeat** from other observatories to generate intersecting cones

### Star Data Model
```python
@dataclass
class Star:
    coordinates: str       # 9-digit location "XXXYYYZZZ"
    original_name: str     # XXX/## format
    player_name: str | None = None  # max 6 chars, uppercase
    discovered_by_user: bool = False
    has_observatory: bool = False
    observatory_active: bool = False
    anomaly_visible: bool = False
```

## Development Milestones

1. **M1: Data + Math Core** - Load stars, parse coordinates, compute distances (CLI)
2. **M2: Static 3D View** - Visualize star map with rotation/zoom
3. **M3: Observatory Projection** - Skybox projection from any observer star
4. **M4: Triangulation** - Input observations, compute cone intersections
5. **M5: Polish** - UI improvements, save/load observations

## Ship Mechanics

### Units
- **Distance**: PC (parsecs) — the 9-digit coordinates map directly to the nav computer's XYZ display
- **Velocity**: PCS (PC/second)

### Fuel System
- Capacity: 0-1000 units
- Refills slowly in real time
- Only consumed when **starting** travel (not during transit)
- Displayed via cylindrical gauge with glowing liquid + numerical readout

### Engine Power
- 3 settings: **1EP**, **2EP**, **3EP** (selected via 3-position knob)
- Higher EP = more fuel consumed = faster travel
- Display shows: current EP, fuel cost for next trip, resulting velocity

### Navigation Computer
Located next to star display, shows (unlabeled):
- Ship XYZ position
- Selected star XYZ position
- Distance to selected star (PC)
- Estimated travel time at current EP + fuel
- Current destination XYZ (if in transit)
- Time remaining (if in transit)

*Note: Like all MirrorMoon EP interfaces, nothing is labeled—players discover meanings through experimentation.*

### Ship Star Map Display
- Flat rectangular screen with center crosshair for star selection
- Ship position shown as **diamond** marker
- Selected star shown with **circle** outline
- Star size indicates distance (larger = closer, smaller = farther)
- Can view any star in the cluster via pan/rotate combinations

**Controls:**
1. **Panning** — moves the camera/cursor position (not the ship itself)
2. **Rotation** — physically rotates the ship, which rotates the view
   - Up (↑) = pitch up (raise head)
   - Down (↓) = pitch down (lower head)
   - Right (→) = yaw LEFT (inverted, like flight controls)
   - Left (←) = yaw RIGHT (inverted)
   - Rotation pivot is always the ship's position
3. **Forward-only toggle** — unlocked after finding first observatory
   - OFF: All stars visible; rotating causes stars to move in different directions (front vs. behind)
   - ON: Only stars in front visible; all visible stars move together when rotating; stars appear/disappear at edges

**Orientation:**
- No angle readouts available — must estimate or use coordinate math
- Orientation likely persists between sessions (unconfirmed)
- "Forward" probably defined by a plane through ship's position

## Mirror Moon & Manipulator

Each moon has a **mirror moon**—a smaller copy that orbits it, functioning as a map/overview. The manipulator primarily interacts with the mirror moon.

### Identifying Structures from Mirror Moon
- **Inactive observatory**: Concentric circles WITHOUT a large inner circle, in an alternate color
- Certain manipulator parts make mirror moon navigation easier

### Manipulator
- 3 modular slots for parts
- Parts are found scattered on the moon's surface
- Some puzzles require specific parts; some observatories are hidden behind full 3-slot puzzles

## Key Constraints

- Observatory views have no compass/orientation cues - matching in-game perspective is the core challenge
- Single observations give direction only, not distance - triangulation requires multiple observatories
- The game likely projects stars onto a spherical skybox - must replicate this projection math
- Player names limited to 6 uppercase characters with numbers and basic symbols

## Future Ideas (Backlog)

These are explicitly deferred—not part of current milestones:

- **Travel path visualization**: Show a path tracing visited stars in order (mirrors in-game feature)
- **Mapping progress indicator**: Show percentage of stars catalogued (in-game shows cluster-wide naming progress)
- **Map orientation helper**: Utility to determine viewing direction by comparing star positions before/after moving (left/right/up/down), then guide user to find missing stars
- **Coordinate prediction**: Estimate likely coordinates of player-named stars based on their relative angular positions to known-coordinate stars from observatory views
- **Auto-alignment**: Given an observatory and a list of reference star names, automatically compute azimuth/elevation to center those stars in the view. User would then only need to determine flip settings and rotation to match their in-game orientation.
- **GUI**: Unified graphical interface bringing together the 3D star map, observatory views, and triangulation workflow.
- **Auto-generated star names**: Generate systematic names for discovered stars using their exact coordinates encoded as three pairs of base-36 digits (0-9, A-Z). Example: star at (123, 456, 789) → "3FCOLX". Fits exactly within the game's 6-character player name limit (by design!). Include a decoder to convert base-36 names back to coordinates for stars not yet in the database.
- **Directional indicators on observatory view**: Show compass-like arrows indicating which direction +/- azimuth and +/- elevation adjustments will move the view. Essential when rotation and flip settings are applied, since the mapping becomes non-intuitive.
- **Navigate to known stars by name**: Allow `--navigate BENCHI` to look up coordinates from the star database, in addition to original names and raw coordinates.
- **Adaptive navigation calibration**: Enhance interactive navigation (`--navigate-interactive`) to refine the ship orientation calibration as you travel. Each position update provides new constraint data that can improve accuracy. Requires: (1) overshoot detection to avoid corrupting calibration when you pass the target, (2) divergence detection to recognize when calibration error is growing rather than shrinking and trigger a recalibration prompt. The goal is progressively more accurate "turn left/right" guidance as you home in on the target.

## Multi-Image Stitching Design (Coordinate Prediction)

This section documents the geometric constraints for estimating unknown star positions from observatory screenshots when only limited reference stars are available.

### The Calibration Problem

Each observatory screenshot has 6 unknowns:
- **Continuous**: FOV, azimuth, elevation, rotation (4 parameters)
- **Discrete**: horizontal flip, vertical flip (4 combinations)

Each known-position star visible in an image provides 2 constraints (its pixel x,y must match its projected position). Therefore:

| Known Stars | Constraints | Result |
|-------------|-------------|--------|
| 1 star | 2 | Underdetermined → cone of possible directions |
| 2 stars | 4 | Determined but flip-ambiguous → 4 candidate rays |
| 3 stars | 6 | Overdetermined, some flips distinguishable → 2 candidate rays |
| 4+ stars | 8+ | Fully constrained → 1 direction ray |

### Multi-Image Stitching Approach

When known-position stars are split across multiple images (e.g., IAY/53 in Image 2, UTA/99 in Image 3), we can link them via constellation correspondences:

**Constraint sources:**
1. **Known-position stars**: Anchor the combined view to real 3D coordinates
2. **Cross-image correspondences**: Same unnamed star marked in two images constrains their relative orientation
3. **Constellation geometry**: The angular pattern is rigid—provides implicit constraints

**Minimum requirements for 4 effective reference points:**
- 2 known-position stars (in any images)
- 2+ unnamed constellation stars marked across image boundaries
- Enough correspondences to chain all images together

**Example for OBSERV screenshots:**
```
Image 2          Image 1              Image 3
─────────        ─────────            ─────────
IAY/53 ✓    ←──  VOID6      ──→      DERVOT
VOID6            DERVOT               KLORIA
anomaly          KLORIA               UTA/99 ✓
                 UMA
                 anomaly
```

Mark correspondences:
1. IAY/53 in Image 2 (known position)
2. UTA/99 in Image 3 (known position)
3. VOID6 in Images 1 & 2 (links those images)
4. DERVOT in Images 1 & 3 (links those images)
5. KLORIA in Images 1 & 3 (resolves flip ambiguity)

### Output

With 2 known-position stars and constellation links, the result is **4 candidate direction rays** for any unknown star (one per flip combination). These rays can be:
1. Visualized in the 3D star map
2. Exported for use with ship navigation (`--navigate-interactive`)
3. Further narrowed if the star has an original name (coordinate bounds from name decoding)

### Key Geometric Insight

Constellation patterns act as a **rigid angular reference frame**. Even though unnamed constellation stars don't have known absolute positions, their fixed angular relationships to each other propagate calibration information across images. This is analogous to surveying: a few known reference points plus angular measurements let you extend your coordinate system to new locations.

### Spherical Moon Geometry Model (v2)

The original calibration model treated azimuth, elevation, and rotation as independent parameters. This was incorrect—the game's actual camera system has strong geometric constraints.

#### The "Roomba" Camera Model

The player walks on the surface of a **tiny spherical moon** with these constraints:

| Action | Effect |
|--------|--------|
| Rotate (A/D keys) | Yaw around local "up" vector |
| Walk (W/S keys) | Move along moon surface at **10°/sec** |
| Tilt up/down | **Not possible** (fixed horizon) |
| Roll | **Not possible** (always level) |
| Zoom | **Not possible** (fixed FOV) |

**Key insight:** "Up" = local gravity = radial direction from moon center. As you walk, your local reference frame rotates because the moon surface curves. Two screenshots taken meters apart see stars from essentially the same 3D position but with completely different "up" directions.

#### Measured Constants

- **Walking angular velocity**: 10°/sec (360° / 36.03s circumnavigation)
- **Moon physical size**: Irrelevant—only angular relationships matter
- **FOV**: ~84° horizontal on 1280×800 screenshots (fitted: 83.9° on the OBSERV set; each image independently prefers 83–84.5°). Assuming 90° leaves every star 2–3° off.
- **Handedness**: the game's frame is left-handed (`right = up × forward`). The mirrored alternative fits 8–23° RMS worse, so there is no flip ambiguity to resolve.

#### New Parameterization

**Per-image parameters (3 DOF):**
```
up_theta, up_phi  — spherical coords defining local "up" vector (2 DOF)
yaw               — horizontal rotation around up vector (1 DOF)
```

**Derived camera orientation:**
```python
up = spherical_to_cartesian(up_theta, up_phi)  # Local gravity direction
right = normalize(cross([0,0,1], up))          # Or other reference
forward = cross(up, right)                      # Look direction (horizon)
forward = rotate_around_axis(forward, up, yaw) # Apply yaw
right = cross(forward, up)                      # Recompute right
```

**Implicit constraints (vs. old model):**
- Elevation = 0 (always looking at local horizon)
- Roll = 0 (camera always level relative to local gravity)
- Flip settings: Unnecessary—confirmed by the fit (see Handedness above)

**Shared parameters:**
- FOV (same game session)
- Observer position (same moon = same star cluster position)

#### Why This Model Is Better

The old model had 4 continuous DOF per image (FOV, azimuth, elevation, rotation) plus flip ambiguity. The solver could find many local minima because:
1. Different (elevation, rotation) combinations could produce similar projections
2. Flip settings were trying to compensate for reference frame differences

The new model has only 3 DOF per image and directly encodes the game's physics:
1. The up vector captures the reference frame difference between screenshots
2. Yaw is the only rotational freedom (matching actual player control)
3. No elevation/roll to optimize—they're implicitly zero

#### Skybox Geometry (Confirmed: Spherical)

**Evidence for spherical skybox:**
1. When walking forward, stars at screen edges curve inward slightly (center stars move straight)
2. **Mirror moon parallax**: The apparent distance between nearby stars and the mirror moon changes with viewing angle—this is definitive proof of a 3D spherical projection, not a flat cubemap

The game renders stars onto a **spherical skybox** that the player views from inside. This explains:
- Why edge stars show higher projection error (~13-16°) vs center stars (~2-4 pixels)
- The characteristic curving motion of edge stars during walking
- Why rectilinear projection is only an approximation

**Practical implications:**
- Rectilinear projection works well for stars near center of frame
- For highest accuracy, mark reference stars away from screen edges
- Edge accuracy matters less for triangulation since we primarily use centered stars

### First-Person Observatory View (by-eye workflow)

The user considers solving from screenshots "a bit cheaty"; the intended workflow is a 3D version of mapping the puzzle on graph paper. `--observatory` opens a view drawn the way the game draws it, which the user lines up by eye against the running game, then marks the anomaly with two clicks (best guess, then the radius they are confident it lies within). The screenshot tools remain as a way to validate the math.

**Controls:** A/D turn, W/S walk (10°/s), Shift = slow, `[` `]` change the tilt, Esc clears a mark. Q/E twist the sky around the view center; this is the only non-game control and is physically valid—it is the same as standing elsewhere on the moon while looking at the same point (in-game it takes a wiggle of walking and turning).

**Design notes:**
- `camera.py` holds all movement and projection logic with no drawing code, so the matplotlib view could later be swapped for a real-time 3D one (the user's lost original prototype used Ursina).
- Stars never move; only the camera's three axes change. Local "up" = the player's position on the moon.
- The old stereographic chart with azimuth/elevation/rotation/flip settings was removed. It did not match the game's projection and its December observations were 6–8° off.

**Camera constants and how sure they are:**
- **Vertical FOV 58.7°** (`camera.py`). Derived from the fitted ~84° horizontal at 16:10. Assuming the game fixes the vertical FOV, this gives exactly 90° horizontal at 16:9, which would explain the old "~90°" note. Only verified at 1280×800.
- **Moon's edge 20° below the view center** (measured in the OBS1 and OBS007 screenshots). This equals tilt + horizon dip; on a tiny moon the visible edge is below the true horizon.
- **Tilt 13° (provisional).** Fitted from the curve of the moon's edge in two screenshots (12–14°, dip ≈ 7°). The earlier 30° assumption would need a negative dip and fits worst. Tilt cannot be seen in a still view; it sets the axis that A/D turns around, so tune it by comparing a turn in-game with a turn in the tool.
- **Turn rate 80°/s** (a full A/D spin takes about 4.5 s, hand-timed by the user; rough). D turns right.
- **Tilt check by the user:** 13° looks about right by eye when comparing turns; 0° is definitely wrong.

### Stitching Solver (v3, current)

The solver in `stitching.py` no longer optimizes up/yaw/pitch iteratively. The earlier L-BFGS-B version got stuck in mirrored and bound-pinned local minima (yaw is periodic but was given hard ±180° bounds) and never converged.

**How it works now:**
1. For a candidate FOV, convert each marked pixel to a unit ray in camera coordinates.
2. Per image, find the rotation that best maps the known stars' world directions onto those rays. This is Wahba's problem and has an exact SVD solution—no starting guess, no local minima.
3. Search the single shared FOV (coarse scan, then 1D refinement) to minimize the summed squared angular residuals.
4. Images with fewer than 2 known stars are fitted afterwards, using correspondences to already-fitted images as extra reference directions.

**Consequences for the models above:**
- Local "up" (2 DOF) plus yaw (1 DOF) already spans every 3D rotation, so each image is simply a free rotation. A shared camera pitch therefore **cannot be measured** from the star fit; `fixed_pitch` (30° in the tool) only changes how the fitted orientation is reported as an up vector and yaw.
- With handedness known, 2 known stars in an image give a unique orientation (the "4 candidate rays" table under The Calibration Problem describes the old flip-ambiguous model).
- Errors are reported as RMS angles in degrees. Cross-image correspondences are weighted lightly in the FOV search and not used for orientation when an image has 2+ known stars, so their agreement is a near-independent check.

**Result on the OBSERV screenshots:** 0.34° RMS, reference stars within 2–8 px, correspondences agreeing to 0.3–0.8°. Anomaly direction from OBSERV: (-0.5983, 0.2791, 0.7511).
