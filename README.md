# Super_Advanced_Ballistics_Tool.Beta_V.1
Advanced Ballistics Tool (Beta)
advanced_ballistics.py is a menu-driven external ballistics simulator with:

large built-in caliber preset library
saved custom calibers
saved rifle profiles
saved optic profiles
single-run trajectory simulation
reverse target solving
DOPE / range card generation
full simulation by range increments
compare mode
Monte Carlo uncertainty mode
terminal trajectory animation
optional self-contained 3D trajectory animation export
Quick Start
Run:

python advanced_ballistics.py
The SQLite database is stored in:

ballistics_results.db
The code creates the required SQLite tables automatically on first run, so a fresh machine does not need a prebuilt database file.

Release Status
This project should currently be treated as:

Beta
That means:

core features are implemented and usable
the tool has been smoke-tested
edge cases may still exist, especially for unusual reverse-solve targets or extreme shot setups
the solver is still an advanced external-ballistics point-mass model, not a rigid-body 6DOF research solver
Dependencies
Required:

Python 3.10+ recommended
Optional but supported:

numpy
numba
Notes:

the tool can run without full GPU acceleration
CUDA-backed full-simulation acceleration is only used when the environment supports it
3D trajectory export is self-contained HTML and does not need extra web packages
Main Flow
When the tool starts, it opens a main menu:

Single Run / Fun Shot
Compare Two Setups
Exit
For a normal shot, the usual order is:

Choose environment values
Choose or create an optic profile
Choose or create a rifle profile
Choose a caliber preset, saved caliber, or custom projectile
Choose an optional barrel-length velocity preset
Enter launch conditions
Enter solver settings
Review the shot result
Open feature categories for deeper analysis
Feature Categories
After a single run, the feature menu provides:

Shooter Tools: DOPE / Range Card
Builds a range card from the recorded trajectory.

Shows:

range
line-of-sight drop
elevation in MOA and mil
wind in MOA
elevation clicks with direction
windage clicks with direction
speed
time of flight
energy
Shooter Tools: Reverse Calculation
Solves the inverse problem:

target forward distance
target height
target lateral offset
It returns:

solved launch elevation
solved launch azimuth
MOA / mil correction
optic clicks with direction
bullet values at target:
time of flight
speed
Mach
energy
drop from muzzle
It can also:

show the solved shot in terminal animation
export a full-flight 3D animation HTML with the target marker
Batch Analysis: Full Simulation
Uses the single shot as the source trajectory and samples it at user-selected range increments up to the run's sample distance.

Outputs:

one result row per range increment
rows stored in SQLite
CUDA path when available
CPU fallback when CUDA is not available
Visuals & Export
Provides:

terminal side-view trajectory animation
CSV trajectory export
optional full-time 3D trajectory animation HTML export
Comparison
Runs two complete setups under the same environment and prints a direct summary.

Useful for:

caliber vs caliber
load vs load
barrel length vs barrel length
Uncertainty / Monte Carlo
Runs repeated varied shots with randomized:

muzzle velocity
elevation
crosswind X
Outputs summary statistics for:

range
drop
drift
impact speed
Profiles and Libraries
Caliber Presets
The built-in preset library is grouped into categories:

Rimfire
Pistols
Revolvers
Intermediate Rifle
Full-Power Rifle
Precision / Long Range
Anti-Materiel
Each preset includes:

representative projectile values
drag model type
barrel-length muzzle velocity suggestions
Saved Calibers
Custom projectile setups can be saved and loaded later from SQLite.

Table:

calibers
Saved Rifle Profiles
Rifle profiles store:

name
sight height
zero range
twist direction
scope click unit
scope click value
Table:

rifle_profiles
Saved Optic Profiles
Optic profiles store:

name
click unit
click value
Table:

optic_profiles
Database Tables
The database currently uses these tables:

calibers
full_simulations
rifle_profiles
optic_profiles
Git / Repo Notes
For a clean repository:

do not commit your local ballistics_results.db
do not commit generated trajectory exports unless you want them as examples
the tool recreates its SQLite tables automatically on a new machine
Notes on Inputs
Elevation
Elevation input is in degrees, not fractions.

Examples:

0 = flat shot
2 = 2 degrees
15 = steep arc
Lateral Offset in Reverse Calculation
This is the target's left/right offset from your current straight-ahead reference line.

0 = target straight ahead
positive = target to the right
negative = target to the left
Sample Distance
Sample distance forward (m) is used for:

sample-point reporting in a single shot
full simulation maximum distance
3D Animation Export
The 3D exporter creates a self-contained HTML file.

It shows:

full flight path over time
bullet position playback
optional target marker
time, range, height, drift, speed, energy
No external web dependencies are required.

Current Model Scope
This tool is an advanced external-ballistics point-mass solver with added effects. It is not yet a true rigid-body 6DOF research solver.

Current major effects include:

drag
gravity
Coriolis
optional Magnus/spin effects
atmosphere variation
spin decay
Troubleshooting
Reverse calculation gives huge click values
Check:

rifle zero range
optic click unit/value
target height input
target lateral offset sign
Recent fixes made reverse corrections relative to the rifle zero, which is the intended behavior.

Simulation ends in the air
If the shot does not hit the ground before max_time_s, the report shows the state at simulation end instead of true impact.

Increase:

Maximum time (s)
Full simulation does not run
Full simulation requires:

trajectory recording enabled
a valid sample distance
the sample distance being reached during the shot
Suggested Typical Workflow
For a practical rifle session:

Load caliber preset
Load optic profile
Load rifle profile
Pick a barrel-length velocity suggestion
Run a single shot
Open DOPE / Range Card
Use Reverse Calculation for exact target-solving
Export 3D animation only when needed
