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
```

## Project Structure

```
starmapper/
├── coords.py        # Coordinate parsing, 9-digit ↔ (x,y,z) transforms
├── projection.py    # Skybox projection math (observer → sky coordinates)
├── triangulation.py # Observation cones, intersection calculations
└── visualization.py # vispy-based 3D rendering
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

## Key Constraints

- Observatory views have no compass/orientation cues - matching in-game perspective is the core challenge
- Single observations give direction only, not distance - triangulation requires multiple observatories
- The game likely projects stars onto a spherical skybox - must replicate this projection math
- Player names limited to 6 uppercase characters with numbers and basic symbols
