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
```

## Development Environment (Distrobox)

This project uses a **Fedora 43 distrobox** container for development. The container provides:
- Full dnf access for installing system packages (e.g., `python3-tkinter`)
- Consistent Python environment with GUI support (matplotlib interactive)
- VS Code with proper Python interpreter detection

### What runs where

| Task | Environment | Command |
|------|-------------|---------|
| Run Python/tests | **Distrobox** | `distrobox enter dev` then run commands |
| Edit code | **Distrobox VS Code** | Launch "VS Code (on dev)" from app menu |
| Git operations | **Host or Distrobox** | Both work (home directory is shared) |
| Claude Code | **Host** | Uses `distrobox enter dev -- <cmd>` to run Python |

### Quick reference
```bash
# Enter the dev container
distrobox enter dev

# Run command inside container without entering
distrobox enter dev -- python3 --version

# Run starmapper from container
distrobox enter dev -- bash -c "cd ~/Documents/starmapper && source .venv/bin/activate && python main.py"
```

## Project Structure

```
starmapper/
├── coords.py        # Coordinate parsing, 9-digit ↔ (x,y,z) transforms, Star dataclass
├── visualization.py # matplotlib-based 3D star map rendering
├── projection.py    # (planned) Skybox projection math (observer → sky coordinates)
└── triangulation.py # (planned) Observation cones, intersection calculations
```

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
