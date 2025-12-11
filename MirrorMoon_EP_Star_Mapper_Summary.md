# MirrorMoon EP Star Mapper - Project Summary

## Project Goal

Create a tool to help locate **anomaly stars** in MirrorMoon EP by:

1. **Visualizing** the game's 3D star map (1000×1000×1000 coordinate space)
2. **Simulating observatory views** to match what you see in-game from a planet's surface
3. **Triangulating** the anomaly's position using observations from multiple observatories

The core challenge: when you spot the anomaly from an observatory, you can see where it appears relative to named stars, but you need multiple observations from different locations to pinpoint its actual 3D coordinates.

---

## About MirrorMoon EP

### Core Concept

MirrorMoon EP is a space exploration and puzzle game where players pilot a spacecraft called the **Boatswain**, traveling between star systems to explore planets. The first player to land on and complete a planet's puzzle gets to name the star system permanently for all players.

### Key Terminology

- **Star / Planet / Moon**: These terms are essentially interchangeable in MirrorMoon EP—each star has exactly one associated planet/moon
- **Galaxy**: Contains ~1000 star systems in a finite 3D space
- **Anomaly**: A single special star per galaxy that players collaborate to find
- **Observatory**: A structure on some planets that, when activated, displays star names and constellation lines in the sky

---

## Coordinate System

### 9-Digit Coordinates

Each star's location is represented by a **9-digit number**, which breaks down into three 3-digit values representing X, Y, and Z coordinates:

```
Example: 123456789
- X = 123
- Y = 456  
- Z = 789
```

Each axis ranges from 000 to 999, creating a 1000×1000×1000 cube of possible star positions.

### Star Naming Convention

Stars have an **original name** in the format `XXX/##` where:
- The three letters correspond to coordinate ranges:
  - A = 000-099
  - E = 100-199
  - I = 200-299
  - U = 300-399
  - Y = 400-499
  - H = 500-599
  - J = 600-699
  - L = 700-799
  - T = 800-899
  - X = 900-999

### Player Names

- Maximum **6 characters**
- **Uppercase letters, numbers, and basic symbols** only
- The game auto-converts lowercase to uppercase (uses 16-segment display style)

---

## Observatory Mechanics

### What Observatories Show

When standing on a planet with an **active observatory**, you can see:
- **Star names** (white text labels in the sky)
- **Constellation lines** connecting certain stars
- **The anomaly** (if visible from that location) highlighted with a large circle

### What Observatories Don't Show

- No angular measurements or coordinates
- No compass directions or orientation cues
- No indication of which direction is "up" relative to the galaxy

### Camera Controls (In-Game)

The first-person observatory view uses only **W/A/S/D** controls:
- **W/S**: Move forward/backward along the planet's surface (orbiting the planet at ground level)
- **A/D**: Turn left/right (rotate view direction)
- Mouse is not used for camera movement

---

## Anomaly Hunting Process

### The Challenge

1. Some planets have observatories that can **see the anomaly** (marked with a circle in the sky)
2. You can estimate the anomaly's position **relative to named stars** (e.g., "halfway between Star A and Star B")
3. Constellation lines help gauge fractional positions
4. A single observation only gives you a **direction**, not a distance
5. Multiple observations from different observatories allow **triangulation**

### Triangulation Method

From each observatory observation:
1. Identify the named stars near the anomaly
2. Estimate where the anomaly appears relative to those stars
3. Project a **cone** (accounting for estimation uncertainty) from the observatory toward the anomaly
4. The **intersection of multiple cones** from different observatories narrows down the anomaly's true position

---

## Data to Track Per Star

| Field | Description |
|-------|-------------|
| `coordinates` | 9-digit location string |
| `originalName` | Game-assigned name (XXX/## format) |
| `playerName` | 6-character player-assigned name |
| `discoveredByUser` | Whether you personally discovered this star |
| `hasObservatory` | Whether this planet has an observatory |
| `observatoryActive` | Whether the observatory has been activated |
| `anomalyVisible` | Whether the anomaly can be seen from this observatory |

---

## Previous Approaches Attempted

### Desmos 3D Graphing

- Created a 1000×1000×1000 cube to represent the game space
- Added stars as individual points
- Drew lines from observatory planets to nearby named stars
- Attempted to create cones pointing toward estimated anomaly positions

**Problems encountered:**
- Nearly impossible to position the camera to match the in-game observatory view
- Difficult to match the correct orientation/perspective
- Manual and time-consuming

---

## Planned Star Mapper Features

### Core Visualization
- 3D star map with the full coordinate space
- Color-coded stars (discovered, observatory, anomaly-visible, etc.)
- Wireframe boundary box
- Toggle-able grid and labels

### Observatory Mode
- First-person view simulation from any observatory
- Display nearby stars as they would appear in the sky
- Draw constellation lines between selected stars
- Mark the estimated anomaly position with an error circle

### Triangulation Tools
- Project cones from observatory sightings
- Calculate and visualize intersection volumes
- Identify candidate anomaly locations
- Planning mode to identify useful observatories to visit

### Data Management
- Save/load star data as JSON
- Track all star properties
- Community data sharing potential

---

## Previous Python Prototype (now lost)

### Tech Stack
- **Language**: Python
- **Game Engine**: Ursina (built on Panda3D)

### Formerly Implemented
- Wireframe cube representing the 1000×1000×1000 game space
- Camera that looks at the cube
- Example spheres representing stars

### Known Implementation Differences from Web Version
The web prototype (HTML/Three.js) we worked on previously had some differences:

| Aspect | Python Version | Web Version |
|--------|---------------|-------------|
| Coordinates | 3 separate integers (X, Y, Z) | Single 9-digit string |

---

## Technical Notes

- Distances between stars use **Euclidean distance** in 3D space
- Travel in-game occurs in real time (distant trips can take 15+ minutes)
- 71% cataloging of stars in a cluster triggers formation of new clusters
- The game is intentionally cryptic—no tutorials, players must discover mechanics themselves
