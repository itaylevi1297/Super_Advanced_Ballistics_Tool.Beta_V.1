"""Advanced external ballistics simulator.

This is an upgraded single-projectile simulator with:
- validated dataclasses for inputs
- ISA atmosphere model with variable speed of sound
- interpolated G1/G7 drag tables
- RK4 integration with event interpolation for exact ground/sample crossing
- Coriolis, Magnus, and spin decay support
- optional trajectory recording and CSV export

The model is still an external-ballistics solver, not a true rigid-body 6DOF
attitude simulation. It tracks point-mass motion plus spin-related effects.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    from numba import cuda
except ImportError:  # pragma: no cover - optional dependency
    cuda = None


PROGRAM_NAME = "BALLISTICS CALCULATOR 4.0 - Advanced External Ballistics Simulator"
EARTH_ROTATION_RATE = 7.2921159e-5
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi
MOA_PER_RAD = 3437.7467707849396
MILS_PER_RAD = 1000.0
EPSILON = 1e-12
DATABASE_PATH = Path(__file__).with_name("ballistics_results.db")


Vector3 = tuple[float, float, float]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def lerp(a: float, b: float, ratio: float) -> float:
    return a + (b - a) * ratio


def dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(v: Vector3) -> float:
    return math.sqrt(dot(v, v))


def cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(v: Vector3, factor: float) -> Vector3:
    return (v[0] * factor, v[1] * factor, v[2] * factor)


def unit(v: Vector3) -> Vector3:
    length = norm(v)
    if length < EPSILON:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def cross_section_area(radius_m: float) -> float:
    return math.pi * radius_m * radius_m


def basis_from_azimuth_deg(azimuth_deg: float) -> tuple[Vector3, Vector3, Vector3]:
    azimuth_rad = math.radians(azimuth_deg)
    forward = (math.sin(azimuth_rad), 0.0, math.cos(azimuth_rad))
    right = (math.cos(azimuth_rad), 0.0, -math.sin(azimuth_rad))
    up = (0.0, 1.0, 0.0)
    return right, up, forward


def local_to_global(v_local: Vector3, azimuth_deg: float) -> Vector3:
    right, up, forward = basis_from_azimuth_deg(azimuth_deg)
    return add(add(scale(right, v_local[0]), scale(up, v_local[1])), scale(forward, v_local[2]))


def global_to_local(v_global: Vector3, azimuth_deg: float) -> Vector3:
    right, up, forward = basis_from_azimuth_deg(azimuth_deg)
    return dot(v_global, right), dot(v_global, up), dot(v_global, forward)


def interpolate_1d(table: Iterable[tuple[float, float]], x: float) -> float:
    points = list(table)
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]

    for index in range(1, len(points)):
        x1, y1 = points[index - 1]
        x2, y2 = points[index]
        if x <= x2:
            ratio = (x - x1) / (x2 - x1)
            return lerp(y1, y2, ratio)
    return points[-1][1]


G1_CD_TABLE: tuple[tuple[float, float], ...] = (
    (0.0, 0.2629),
    (0.5, 0.2500),
    (0.7, 0.2558),
    (0.8, 0.2700),
    (0.9, 0.3100),
    (0.95, 0.3800),
    (1.00, 0.4800),
    (1.05, 0.5300),
    (1.10, 0.5400),
    (1.20, 0.5250),
    (1.30, 0.5000),
    (1.40, 0.4850),
    (1.60, 0.4600),
    (1.80, 0.4400),
    (2.00, 0.4200),
    (2.50, 0.3900),
    (3.00, 0.3650),
    (4.00, 0.3300),
)


G7_CD_TABLE: tuple[tuple[float, float], ...] = (
    (0.0, 0.1198),
    (0.5, 0.1190),
    (0.7, 0.1200),
    (0.8, 0.1240),
    (0.9, 0.1400),
    (0.95, 0.1700),
    (1.00, 0.2150),
    (1.05, 0.2450),
    (1.10, 0.2550),
    (1.20, 0.2450),
    (1.30, 0.2330),
    (1.40, 0.2200),
    (1.60, 0.2050),
    (1.80, 0.1950),
    (2.00, 0.1900),
    (2.50, 0.1820),
    (3.00, 0.1760),
    (4.00, 0.1700),
)


@dataclass(slots=True)
class Environment:
    latitude_deg: float = 32.0
    wind_x: float = 0.0
    wind_y: float = 0.0
    wind_z: float = 0.0
    gravity: float = 9.80665
    sea_level_pressure_pa: float = 101325.0
    sea_level_temperature_k: float = 288.15
    relative_humidity: float = 0.0
    temperature_lapse_k_per_m: float = 0.0065

    def validate(self) -> None:
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("Latitude must be between -90 and 90 degrees.")
        if self.gravity <= 0.0:
            raise ValueError("Gravity must be positive.")
        if self.sea_level_pressure_pa <= 0.0:
            raise ValueError("Sea-level pressure must be positive.")
        if self.sea_level_temperature_k <= 0.0:
            raise ValueError("Sea-level temperature must be positive.")
        if not 0.0 <= self.relative_humidity <= 1.0:
            raise ValueError("Relative humidity must be in the range [0, 1].")


@dataclass(slots=True)
class Projectile:
    mass_kg: float
    diameter_m: float
    ballistic_coefficient: float
    bc_model: str = "G7"
    magnus_coefficient: float = 0.0
    spin_decay_rate: float = 0.015
    twist_rate_inches: float = 8.0
    bullet_length_m: float = 0.03

    def validate(self) -> None:
        if self.mass_kg <= 0.0:
            raise ValueError("Projectile mass must be positive.")
        if self.diameter_m <= 0.0:
            raise ValueError("Projectile diameter must be positive.")
        if self.ballistic_coefficient <= 0.0:
            raise ValueError("Ballistic coefficient must be positive.")
        if self.twist_rate_inches <= 0.0:
            raise ValueError("Twist rate must be positive.")
        if self.bullet_length_m <= 0.0:
            raise ValueError("Bullet length must be positive.")
        if self.bc_model.upper() not in {"G1", "G7"}:
            raise ValueError("BC model must be G1 or G7.")

    @property
    def radius_m(self) -> float:
        return self.diameter_m / 2.0

    @property
    def area_m2(self) -> float:
        return cross_section_area(self.radius_m)

    @property
    def twist_m_per_turn(self) -> float:
        return self.twist_rate_inches * 0.0254

    @property
    def sectional_density_lb_per_in2(self) -> float:
        mass_lb = self.mass_kg * 2.20462262185
        diameter_in = self.diameter_m / 0.0254
        if diameter_in <= 0.0:
            return 0.0
        return mass_lb / (diameter_in * diameter_in)


@dataclass(slots=True)
class BarrelPreset:
    label: str
    barrel_length_in: float
    muzzle_velocity_mps: float


@dataclass(slots=True)
class CaliberPreset:
    category: str
    name: str
    description: str
    projectile: Projectile
    barrel_presets: list[BarrelPreset]


def bp(label: str, barrel_length_in: float, muzzle_velocity_mps: float) -> BarrelPreset:
    return BarrelPreset(label, barrel_length_in, muzzle_velocity_mps)


def cp(
    category: str,
    name: str,
    description: str,
    mass_kg: float,
    diameter_m: float,
    ballistic_coefficient: float,
    bc_model: str,
    twist_rate_inches: float,
    bullet_length_m: float,
    barrel_presets: list[BarrelPreset],
) -> CaliberPreset:
    return CaliberPreset(
        category=category,
        name=name,
        description=description,
        projectile=Projectile(mass_kg, diameter_m, ballistic_coefficient, bc_model, 0.0, 0.015, twist_rate_inches, bullet_length_m),
        barrel_presets=barrel_presets,
    )


POPULAR_CALIBER_PRESETS: tuple[CaliberPreset, ...] = (
    cp("Rimfire", ".17 HMR", "17 gr V-MAX", 0.00110, 0.00450, 0.125, "G1", 9.0, 0.0090, [bp("Rifle 18 in", 18.0, 775.0), bp("Rifle 22 in", 22.0, 790.0)]),
    cp("Rimfire", ".22 LR", "40 gr round nose", 0.00259, 0.00570, 0.130, "G1", 16.0, 0.0120, [bp("Rifle 16 in", 16.0, 330.0), bp("Rifle 20 in", 20.0, 360.0), bp("Pistol 6 in", 6.0, 305.0)]),
    cp("Rimfire", ".22 WMR", "40 gr JHP", 0.00259, 0.00570, 0.114, "G1", 16.0, 0.0145, [bp("Rifle 18 in", 18.0, 575.0), bp("Rifle 22 in", 22.0, 590.0), bp("Handgun 10 in", 10.0, 480.0)]),
    cp("Pistols", ".25 ACP", "50 gr FMJ", 0.00324, 0.00635, 0.090, "G1", 16.0, 0.0100, [bp("Pocket 2 in", 2.0, 235.0), bp("Pistol 3 in", 3.0, 255.0)]),
    cp("Pistols", ".32 ACP", "71 gr FMJ", 0.00460, 0.00794, 0.110, "G1", 16.0, 0.0115, [bp("Pocket 2.5 in", 2.5, 260.0), bp("Pistol 3.8 in", 3.8, 290.0)]),
    cp("Pistols", ".380 ACP", "95 gr FMJ", 0.00616, 0.00902, 0.110, "G1", 10.0, 0.0135, [bp("Pistol 2.8 in", 2.8, 285.0), bp("Pistol 3.5 in", 3.5, 300.0), bp("Pistol 4 in", 4.0, 315.0)]),
    cp("Pistols", "9mm Luger", "124 gr JHP", 0.00804, 0.00901, 0.150, "G1", 10.0, 0.0150, [bp("Pistol 3.5 in", 3.5, 335.0), bp("Pistol 4 in", 4.0, 350.0), bp("Pistol 5 in", 5.0, 365.0), bp("PCC 16 in", 16.0, 395.0)]),
    cp("Pistols", ".357 SIG", "125 gr JHP", 0.00810, 0.00902, 0.145, "G1", 16.0, 0.0145, [bp("Pistol 4 in", 4.0, 410.0), bp("Pistol 5 in", 5.0, 430.0)]),
    cp("Pistols", ".38 Super", "124 gr FMJ", 0.00804, 0.00904, 0.150, "G1", 16.0, 0.0150, [bp("Pistol 5 in", 5.0, 390.0), bp("Comp 6 in", 6.0, 405.0)]),
    cp("Pistols", ".40 S&W", "180 gr JHP", 0.01166, 0.01017, 0.160, "G1", 16.0, 0.0170, [bp("Pistol 4 in", 4.0, 305.0), bp("Pistol 4.5 in", 4.5, 315.0), bp("PCC 16 in", 16.0, 395.0)]),
    cp("Pistols", ".45 ACP", "230 gr FMJ", 0.01490, 0.01148, 0.195, "G1", 16.0, 0.0190, [bp("Pistol 4.25 in", 4.25, 255.0), bp("Pistol 5 in", 5.0, 265.0), bp("PCC 16 in", 16.0, 335.0)]),
    cp("Pistols", "10mm Auto", "180 gr JHP", 0.01166, 0.01017, 0.170, "G1", 16.0, 0.0170, [bp("Pistol 4.6 in", 4.6, 365.0), bp("Pistol 6 in", 6.0, 395.0), bp("Carbine 16 in", 16.0, 490.0)]),
    cp("Pistols", "5.7x28mm", "40 gr V-MAX", 0.00259, 0.00570, 0.200, "G1", 9.0, 0.0210, [bp("Pistol 4.8 in", 4.8, 520.0), bp("Carbine 16 in", 16.0, 715.0)]),
    cp("Revolvers", ".38 Special", "130 gr FMJ", 0.00842, 0.00907, 0.160, "G1", 18.75, 0.0160, [bp("Revolver 2 in", 2.0, 235.0), bp("Revolver 4 in", 4.0, 265.0), bp("Revolver 6 in", 6.0, 280.0)]),
    cp("Revolvers", ".357 Magnum", "158 gr JHP", 0.01024, 0.00907, 0.200, "G1", 18.75, 0.0180, [bp("Revolver 4 in", 4.0, 380.0), bp("Revolver 6 in", 6.0, 430.0), bp("Carbine 16 in", 16.0, 560.0)]),
    cp("Revolvers", ".41 Magnum", "210 gr JSP", 0.01361, 0.01041, 0.185, "G1", 18.75, 0.0200, [bp("Revolver 4 in", 4.0, 395.0), bp("Revolver 6 in", 6.0, 430.0)]),
    cp("Revolvers", ".44 Magnum", "240 gr JSP", 0.01555, 0.01090, 0.185, "G1", 20.0, 0.0220, [bp("Revolver 4 in", 4.0, 365.0), bp("Revolver 6 in", 6.0, 425.0), bp("Carbine 16 in", 16.0, 535.0)]),
    cp("Revolvers", ".45 Colt", "255 gr lead flat point", 0.01652, 0.01148, 0.180, "G1", 16.0, 0.0200, [bp("Revolver 5.5 in", 5.5, 275.0), bp("Rifle 20 in", 20.0, 410.0)]),
    cp("Revolvers", ".454 Casull", "300 gr XTP", 0.01944, 0.01148, 0.210, "G1", 24.0, 0.0240, [bp("Revolver 7.5 in", 7.5, 500.0), bp("Carbine 18 in", 18.0, 610.0)]),
    cp("Revolvers", ".500 S&W Magnum", "350 gr JHP", 0.02268, 0.01270, 0.210, "G1", 18.0, 0.0260, [bp("Revolver 8.4 in", 8.4, 565.0), bp("Revolver 10.5 in", 10.5, 610.0)]),
    cp("Intermediate Rifle", ".17 Hornet", "20 gr V-MAX", 0.00130, 0.00450, 0.185, "G1", 10.0, 0.0140, [bp("Rifle 22 in", 22.0, 1110.0), bp("Rifle 24 in", 24.0, 1140.0)]),
    cp("Intermediate Rifle", ".204 Ruger", "32 gr V-MAX", 0.00207, 0.00518, 0.192, "G1", 12.0, 0.0170, [bp("Rifle 22 in", 22.0, 1220.0), bp("Rifle 24 in", 24.0, 1280.0)]),
    cp("Intermediate Rifle", ".223 Rem / 5.56 NATO", "55 gr FMJ", 0.00356, 0.00570, 0.255, "G1", 8.0, 0.0195, [bp("AR 10.5 in", 10.5, 805.0), bp("AR 14.5 in", 14.5, 877.0), bp("AR 16 in", 16.0, 898.0), bp("Rifle 20 in", 20.0, 960.0)]),
    cp("Intermediate Rifle", "5.45x39mm", "60 gr FMJ", 0.00389, 0.00562, 0.255, "G1", 8.0, 0.0220, [bp("AK-74 16 in", 16.0, 880.0), bp("Rifle 20 in", 20.0, 910.0)]),
    cp("Intermediate Rifle", ".300 Blackout", "125 gr OTM supersonic", 0.00810, 0.00782, 0.290, "G1", 8.0, 0.0240, [bp("SBR 9 in", 9.0, 640.0), bp("Carbine 16 in", 16.0, 675.0), bp("Subsonic 16 in", 16.0, 305.0)]),
    cp("Intermediate Rifle", "7.62x39mm", "123 gr FMJ", 0.00797, 0.00792, 0.275, "G1", 9.5, 0.0220, [bp("AK 12.5 in", 12.5, 655.0), bp("AK 16 in", 16.0, 715.0), bp("Rifle 20 in", 20.0, 730.0)]),
    cp("Intermediate Rifle", ".30 Carbine", "110 gr FMJ", 0.00713, 0.00782, 0.170, "G1", 20.0, 0.0180, [bp("Carbine 18 in", 18.0, 605.0), bp("Handgun 10 in", 10.0, 470.0)]),
    cp("Intermediate Rifle", ".224 Valkyrie", "75 gr BTHP", 0.00486, 0.00570, 0.215, "G7", 7.0, 0.0260, [bp("AR 18 in", 18.0, 860.0), bp("AR 20 in", 20.0, 885.0), bp("AR 24 in", 24.0, 915.0)]),
    cp("Intermediate Rifle", "6.5 Grendel", "123 gr ELD-M", 0.00797, 0.00671, 0.255, "G7", 8.0, 0.0310, [bp("AR 16 in", 16.0, 760.0), bp("AR 18 in", 18.0, 775.0), bp("AR 20 in", 20.0, 790.0)]),
    cp("Intermediate Rifle", "6mm ARC", "108 gr ELD-M", 0.00700, 0.00617, 0.270, "G7", 7.5, 0.0290, [bp("AR 18 in", 18.0, 820.0), bp("AR 20 in", 20.0, 840.0), bp("Bolt 24 in", 24.0, 885.0)]),
    cp("Intermediate Rifle", ".350 Legend", "150 gr deer load", 0.00972, 0.00900, 0.210, "G1", 16.0, 0.0180, [bp("Rifle 16 in", 16.0, 700.0), bp("Rifle 20 in", 20.0, 716.0)]),
    cp("Full-Power Rifle", ".243 Winchester", "100 gr soft point", 0.00648, 0.00617, 0.405, "G1", 10.0, 0.0270, [bp("Rifle 20 in", 20.0, 900.0), bp("Rifle 22 in", 22.0, 915.0), bp("Rifle 24 in", 24.0, 930.0)]),
    cp("Full-Power Rifle", ".25-06 Remington", "115 gr ballistic tip", 0.00745, 0.00653, 0.230, "G7", 10.0, 0.0300, [bp("Rifle 22 in", 22.0, 935.0), bp("Rifle 24 in", 24.0, 960.0), bp("Rifle 26 in", 26.0, 975.0)]),
    cp("Full-Power Rifle", ".270 Winchester", "130 gr soft point", 0.00842, 0.00704, 0.435, "G1", 10.0, 0.0310, [bp("Rifle 22 in", 22.0, 915.0), bp("Rifle 24 in", 24.0, 945.0), bp("Rifle 26 in", 26.0, 960.0)]),
    cp("Full-Power Rifle", "7mm-08 Remington", "140 gr hunting load", 0.00907, 0.00721, 0.248, "G7", 9.5, 0.0300, [bp("Rifle 18 in", 18.0, 785.0), bp("Rifle 20 in", 20.0, 805.0), bp("Rifle 22 in", 22.0, 820.0)]),
    cp("Full-Power Rifle", ".308 Winchester", "175 gr HPBT", 0.01134, 0.00782, 0.243, "G7", 12.0, 0.0312, [bp("Rifle 16 in", 16.0, 744.0), bp("Rifle 20 in", 20.0, 777.0), bp("Rifle 24 in", 24.0, 802.0)]),
    cp("Full-Power Rifle", ".30-30 Winchester", "150 gr flat nose", 0.00972, 0.00782, 0.190, "G1", 12.0, 0.0250, [bp("Lever 16 in", 16.0, 710.0), bp("Lever 20 in", 20.0, 731.0), bp("Lever 24 in", 24.0, 750.0)]),
    cp("Full-Power Rifle", ".30-06 Springfield", "180 gr soft point", 0.01166, 0.00782, 0.248, "G7", 10.0, 0.0320, [bp("Rifle 22 in", 22.0, 820.0), bp("Rifle 24 in", 24.0, 838.0), bp("Rifle 26 in", 26.0, 853.0)]),
    cp("Full-Power Rifle", ".45-70 Government", "300 gr JHP", 0.01944, 0.01163, 0.214, "G1", 20.0, 0.0260, [bp("Lever 18 in", 18.0, 560.0), bp("Lever 22 in", 22.0, 610.0)]),
    cp("Full-Power Rifle", "8x57 IS", "196 gr soft point", 0.01270, 0.00822, 0.260, "G1", 9.5, 0.0320, [bp("Rifle 23 in", 23.0, 790.0), bp("Rifle 25 in", 25.0, 815.0)]),
    cp("Precision / Long Range", ".260 Remington", "140 gr BTHP", 0.00907, 0.00671, 0.290, "G7", 8.0, 0.0340, [bp("Rifle 22 in", 22.0, 805.0), bp("Rifle 24 in", 24.0, 825.0)]),
    cp("Precision / Long Range", "6mm Creedmoor", "108 gr match", 0.00700, 0.00617, 0.270, "G7", 7.5, 0.0300, [bp("Rifle 24 in", 24.0, 905.0), bp("Rifle 26 in", 26.0, 930.0)]),
    cp("Precision / Long Range", "6.5 Creedmoor", "140 gr BTHP", 0.00907, 0.00671, 0.315, "G7", 8.0, 0.0340, [bp("Rifle 20 in", 20.0, 790.0), bp("Rifle 22 in", 22.0, 820.0), bp("Rifle 24 in", 24.0, 835.0)]),
    cp("Precision / Long Range", "6.5 PRC", "143 gr ELD-X", 0.00927, 0.00671, 0.315, "G7", 8.0, 0.0350, [bp("Rifle 22 in", 22.0, 885.0), bp("Rifle 24 in", 24.0, 905.0), bp("Rifle 26 in", 26.0, 920.0)]),
    cp("Precision / Long Range", ".280 Ackley Improved", "160 gr AccuBond", 0.01037, 0.00721, 0.283, "G7", 9.0, 0.0340, [bp("Rifle 24 in", 24.0, 870.0), bp("Rifle 26 in", 26.0, 890.0)]),
    cp("Precision / Long Range", "7mm Remington Magnum", "162 gr BTHP", 0.01050, 0.00721, 0.315, "G7", 9.25, 0.0360, [bp("Rifle 24 in", 24.0, 900.0), bp("Rifle 26 in", 26.0, 930.0)]),
    cp("Precision / Long Range", ".300 Winchester Magnum", "190 gr HPBT", 0.01231, 0.00782, 0.290, "G7", 10.0, 0.0380, [bp("Rifle 24 in", 24.0, 878.0), bp("Rifle 26 in", 26.0, 899.0)]),
    cp("Precision / Long Range", ".300 PRC", "225 gr ELD-M", 0.01458, 0.00782, 0.391, "G7", 9.0, 0.0440, [bp("Rifle 24 in", 24.0, 860.0), bp("Rifle 26 in", 26.0, 885.0)]),
    cp("Precision / Long Range", ".338 Lapua Magnum", "250 gr Scenar", 0.01620, 0.00861, 0.330, "G7", 10.0, 0.0430, [bp("Rifle 24 in", 24.0, 860.0), bp("Rifle 27 in", 27.0, 900.0), bp("Rifle 30 in", 30.0, 915.0)]),
    cp("Anti-Materiel", ".408 CheyTac", "419 gr monolithic", 0.02715, 0.01036, 0.470, "G7", 13.0, 0.0530, [bp("Rifle 29 in", 29.0, 915.0), bp("Rifle 32 in", 32.0, 930.0)]),
    cp("Anti-Materiel", ".416 Barrett", "395 gr solid", 0.02560, 0.01057, 0.398, "G7", 12.0, 0.0510, [bp("Rifle 29 in", 29.0, 960.0), bp("Rifle 32 in", 32.0, 985.0)]),
    cp("Anti-Materiel", ".50 BMG", "750 gr A-MAX", 0.04860, 0.01295, 0.530, "G7", 15.0, 0.0560, [bp("Rifle 29 in", 29.0, 850.0), bp("Rifle 32 in", 32.0, 880.0)]),
    cp("Anti-Materiel", "14.5x114mm", "994 gr API class", 0.06440, 0.01450, 0.430, "G7", 16.0, 0.0620, [bp("Rifle 39 in", 39.0, 980.0)]),
)


PRESET_BY_NAME = {preset.name: preset for preset in POPULAR_CALIBER_PRESETS}
PRESET_CATEGORY_ORDER = (
    "Rimfire",
    "Pistols",
    "Revolvers",
    "Intermediate Rifle",
    "Full-Power Rifle",
    "Precision / Long Range",
    "Anti-Materiel",
)
PRESET_NAME_ALIASES = {
    "308.": ".308 Winchester",
    "308": ".308 Winchester",
    "223": ".223 Rem / 5.56 NATO",
    "5.56": ".223 Rem / 5.56 NATO",
    "9mm": "9mm Luger",
    "45 acp": ".45 ACP",
    "300 blackout": ".300 Blackout",
    "300 win mag": ".300 Winchester Magnum",
    "30-06": ".30-06 Springfield",
    "50 bmg": ".50 BMG",
}


@dataclass(slots=True)
class Launch:
    muzzle_velocity_mps: float
    elevation_deg: float
    azimuth_deg: float
    muzzle_height_m: float = 1.5
    sample_distance_m: Optional[float] = None

    def validate(self) -> None:
        if self.muzzle_velocity_mps <= 0.0:
            raise ValueError("Muzzle velocity must be positive.")
        if not -10.0 <= self.elevation_deg <= 90.0:
            raise ValueError("Elevation must be in the range [-10, 90] degrees.")
        if self.muzzle_height_m < 0.0:
            raise ValueError("Muzzle height cannot be negative.")
        if self.sample_distance_m is not None and self.sample_distance_m <= 0.0:
            raise ValueError("Sample distance must be positive when provided.")


@dataclass(slots=True)
class RifleProfile:
    name: str = "Default Rifle"
    sight_height_m: float = 0.05
    zero_range_m: Optional[float] = 100.0
    twist_direction: str = "R"
    scope_click_unit: str = "MOA"
    scope_click_value: float = 0.25

    def validate(self) -> None:
        if self.sight_height_m < 0.0:
            raise ValueError("Sight height must be non-negative.")
        if self.zero_range_m is not None and self.zero_range_m <= 0.0:
            raise ValueError("Zero range must be positive when provided.")
        if self.twist_direction.upper() not in {"R", "L"}:
            raise ValueError("Twist direction must be R or L.")
        if self.scope_click_unit.upper() not in {"MOA", "MIL"}:
            raise ValueError("Scope click unit must be MOA or MIL.")
        if self.scope_click_value <= 0.0:
            raise ValueError("Scope click value must be positive.")


@dataclass(slots=True)
class OpticProfile:
    name: str = "Default Optic"
    click_unit: str = "MOA"
    click_value: float = 0.25

    def validate(self) -> None:
        if self.click_unit.upper() not in {"MOA", "MIL"}:
            raise ValueError("Optic click unit must be MOA or MIL.")
        if self.click_value <= 0.0:
            raise ValueError("Optic click value must be positive.")


@dataclass(slots=True)
class SolverConfig:
    dt: float = 0.001
    max_time_s: float = 10.0
    record_trajectory: bool = True
    output_interval_s: float = 0.01
    verbose: bool = True

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("Time step dt must be positive.")
        if self.max_time_s <= 0.0:
            raise ValueError("Maximum simulation time must be positive.")
        if self.output_interval_s <= 0.0:
            raise ValueError("Output interval must be positive.")


@dataclass(slots=True)
class State:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    omega_x: float
    omega_y: float
    omega_z: float


@dataclass(slots=True)
class TrajectorySample:
    time_s: float
    x_m: float
    y_m: float
    z_m: float
    speed_mps: float
    mach: float
    energy_j: float
    drift_m: float
    drop_m: float


@dataclass(slots=True)
class SimulationResult:
    time_of_flight_s: float
    range_m: float
    lateral_m: float
    impact_height_m: float
    drop_from_muzzle_m: float
    speed_mps: float
    impact_mach: float
    energy_j: float
    impact_angle_deg: float
    max_ordinate_m: float
    windage_moa: float
    elevation_moa: float
    windage_mils: float
    elevation_mils: float
    stability_factor_sg: float
    terminated_by_ground_impact: bool
    termination_reason: str
    sample_hit: Optional[TrajectorySample]
    trajectory: list[TrajectorySample] = field(default_factory=list)

    def write_trajectory_csv(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "time_s",
                    "x_m",
                    "y_m",
                    "z_m",
                    "speed_mps",
                    "mach",
                    "energy_j",
                    "drift_m",
                    "drop_m",
                ]
            )
            for sample in self.trajectory:
                writer.writerow(
                    [
                        f"{sample.time_s:.6f}",
                        f"{sample.x_m:.6f}",
                        f"{sample.y_m:.6f}",
                        f"{sample.z_m:.6f}",
                        f"{sample.speed_mps:.6f}",
                        f"{sample.mach:.6f}",
                        f"{sample.energy_j:.6f}",
                        f"{sample.drift_m:.6f}",
                        f"{sample.drop_m:.6f}",
                    ]
                )
        return path


@dataclass(slots=True)
class FullSimulationRow:
    target_range_m: float
    time_of_flight_s: float
    lateral_m: float
    height_m: float
    drop_from_muzzle_m: float
    speed_mps: float
    mach: float
    energy_j: float
    windage_moa: float
    elevation_moa: float
    windage_mils: float
    elevation_mils: float


@dataclass(slots=True)
class FullSimulationResult:
    run_label: str
    increment_m: float
    used_cuda: bool
    rows: list[FullSimulationRow]
    sqlite_path: Path


@dataclass(slots=True)
class ReverseSolution:
    solved: bool
    iterations: int
    target_forward_m: float
    target_height_m: float
    target_lateral_m: float
    solved_elevation_deg: float
    solved_azimuth_deg: float
    elevation_adjustment_moa: float
    elevation_adjustment_mils: float
    windage_adjustment_moa: float
    windage_adjustment_mils: float
    elevation_clicks: float
    windage_clicks: float
    final_vertical_error_m: float
    final_lateral_error_m: float
    target_time_s: float
    target_speed_mps: float
    target_mach: float
    target_energy_j: float
    target_drop_from_muzzle_m: float
    message: str


def get_drag_coefficient(mach: float, bc_model: str) -> float:
    model = bc_model.upper()
    if model == "G7":
        return interpolate_1d(G7_CD_TABLE, mach)
    return interpolate_1d(G1_CD_TABLE, mach)


if cuda is not None:

    @cuda.jit
    def interpolate_rows_cuda(
        targets,
        sample_ranges,
        sample_times,
        sample_drifts,
        sample_heights,
        sample_speeds,
        sample_machs,
        sample_energies,
        sample_drops,
        out_times,
        out_drifts,
        out_heights,
        out_speeds,
        out_machs,
        out_energies,
        out_drops,
    ):
        index = cuda.grid(1)
        if index >= targets.size:
            return

        target = targets[index]
        last = sample_ranges.size - 1

        if target <= sample_ranges[0]:
            left = 0
            right = 0
        elif target >= sample_ranges[last]:
            left = last
            right = last
        else:
            left = 0
            right = last
            for pos in range(1, sample_ranges.size):
                if target <= sample_ranges[pos]:
                    left = pos - 1
                    right = pos
                    break

        x1 = sample_ranges[left]
        x2 = sample_ranges[right]
        ratio = 0.0
        if right != left and x2 != x1:
            ratio = (target - x1) / (x2 - x1)

        out_times[index] = sample_times[left] + (sample_times[right] - sample_times[left]) * ratio
        out_drifts[index] = sample_drifts[left] + (sample_drifts[right] - sample_drifts[left]) * ratio
        out_heights[index] = sample_heights[left] + (sample_heights[right] - sample_heights[left]) * ratio
        out_speeds[index] = sample_speeds[left] + (sample_speeds[right] - sample_speeds[left]) * ratio
        out_machs[index] = sample_machs[left] + (sample_machs[right] - sample_machs[left]) * ratio
        out_energies[index] = sample_energies[left] + (sample_energies[right] - sample_energies[left]) * ratio
        out_drops[index] = sample_drops[left] + (sample_drops[right] - sample_drops[left]) * ratio


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS calibers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                mass_kg REAL NOT NULL,
                diameter_m REAL NOT NULL,
                ballistic_coefficient REAL NOT NULL,
                bc_model TEXT NOT NULL,
                magnus_coefficient REAL NOT NULL,
                spin_decay_rate REAL NOT NULL,
                twist_rate_inches REAL NOT NULL,
                bullet_length_m REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS full_simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_label TEXT NOT NULL,
                caliber_name TEXT,
                bc_model TEXT NOT NULL,
                ballistic_coefficient REAL NOT NULL,
                mass_kg REAL NOT NULL,
                diameter_m REAL NOT NULL,
                muzzle_velocity_mps REAL NOT NULL,
                elevation_deg REAL NOT NULL,
                azimuth_deg REAL NOT NULL,
                muzzle_height_m REAL NOT NULL,
                latitude_deg REAL NOT NULL,
                wind_x REAL NOT NULL,
                wind_y REAL NOT NULL,
                wind_z REAL NOT NULL,
                gravity REAL NOT NULL,
                sample_distance_m REAL NOT NULL,
                increment_m REAL NOT NULL,
                used_cuda INTEGER NOT NULL,
                target_range_m REAL NOT NULL,
                time_of_flight_s REAL NOT NULL,
                lateral_m REAL NOT NULL,
                height_m REAL NOT NULL,
                drop_from_muzzle_m REAL NOT NULL,
                speed_mps REAL NOT NULL,
                mach REAL NOT NULL,
                energy_j REAL NOT NULL,
                windage_moa REAL NOT NULL,
                elevation_moa REAL NOT NULL,
                windage_mils REAL NOT NULL,
                elevation_mils REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rifle_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                sight_height_m REAL NOT NULL,
                zero_range_m REAL,
                twist_direction TEXT NOT NULL,
                scope_click_unit TEXT NOT NULL,
                scope_click_value REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS optic_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                click_unit TEXT NOT NULL,
                click_value REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def fetch_saved_calibers() -> list[sqlite3.Row]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, name, mass_kg, diameter_m, ballistic_coefficient, bc_model,
                   magnus_coefficient, spin_decay_rate, twist_rate_inches, bullet_length_m,
                   created_at
            FROM calibers
            ORDER BY name COLLATE NOCASE
            """
        ).fetchall()
    return rows


def save_caliber(name: str, projectile: Projectile) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO calibers (
                name, mass_kg, diameter_m, ballistic_coefficient, bc_model,
                magnus_coefficient, spin_decay_rate, twist_rate_inches, bullet_length_m
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                mass_kg=excluded.mass_kg,
                diameter_m=excluded.diameter_m,
                ballistic_coefficient=excluded.ballistic_coefficient,
                bc_model=excluded.bc_model,
                magnus_coefficient=excluded.magnus_coefficient,
                spin_decay_rate=excluded.spin_decay_rate,
                twist_rate_inches=excluded.twist_rate_inches,
                bullet_length_m=excluded.bullet_length_m
            """,
            (
                name,
                projectile.mass_kg,
                projectile.diameter_m,
                projectile.ballistic_coefficient,
                projectile.bc_model.upper(),
                projectile.magnus_coefficient,
                projectile.spin_decay_rate,
                projectile.twist_rate_inches,
                projectile.bullet_length_m,
            ),
        )


def fetch_saved_rifle_profiles() -> list[sqlite3.Row]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, name, sight_height_m, zero_range_m, twist_direction,
                   scope_click_unit, scope_click_value, created_at
            FROM rifle_profiles
            ORDER BY name COLLATE NOCASE
            """
        ).fetchall()
    return rows


def save_rifle_profile(profile: RifleProfile) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO rifle_profiles (
                name, sight_height_m, zero_range_m, twist_direction,
                scope_click_unit, scope_click_value
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                sight_height_m=excluded.sight_height_m,
                zero_range_m=excluded.zero_range_m,
                twist_direction=excluded.twist_direction,
                scope_click_unit=excluded.scope_click_unit,
                scope_click_value=excluded.scope_click_value
            """,
            (
                profile.name,
                profile.sight_height_m,
                profile.zero_range_m,
                profile.twist_direction.upper(),
                profile.scope_click_unit.upper(),
                profile.scope_click_value,
            ),
        )


def rifle_profile_from_row(row: sqlite3.Row) -> RifleProfile:
    return RifleProfile(
        name=str(row["name"]),
        sight_height_m=float(row["sight_height_m"]),
        zero_range_m=float(row["zero_range_m"]) if row["zero_range_m"] is not None else None,
        twist_direction=str(row["twist_direction"]),
        scope_click_unit=str(row["scope_click_unit"]),
        scope_click_value=float(row["scope_click_value"]),
    )


def fetch_saved_optic_profiles() -> list[sqlite3.Row]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, name, click_unit, click_value, created_at
            FROM optic_profiles
            ORDER BY name COLLATE NOCASE
            """
        ).fetchall()
    return rows


def save_optic_profile(profile: OpticProfile) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO optic_profiles (name, click_unit, click_value)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                click_unit=excluded.click_unit,
                click_value=excluded.click_value
            """,
            (profile.name, profile.click_unit.upper(), profile.click_value),
        )


def optic_profile_from_row(row: sqlite3.Row) -> OpticProfile:
    return OpticProfile(
        name=str(row["name"]),
        click_unit=str(row["click_unit"]),
        click_value=float(row["click_value"]),
    )


def projectile_from_row(row: sqlite3.Row) -> Projectile:
    return Projectile(
        mass_kg=float(row["mass_kg"]),
        diameter_m=float(row["diameter_m"]),
        ballistic_coefficient=float(row["ballistic_coefficient"]),
        bc_model=str(row["bc_model"]),
        magnus_coefficient=float(row["magnus_coefficient"]),
        spin_decay_rate=float(row["spin_decay_rate"]),
        twist_rate_inches=float(row["twist_rate_inches"]),
        bullet_length_m=float(row["bullet_length_m"]),
    )


def save_full_simulation_rows(
    caliber_name: Optional[str],
    projectile: Projectile,
    env: Environment,
    launch: Launch,
    full_result: FullSimulationResult,
) -> None:
    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO full_simulations (
                run_label, caliber_name, bc_model, ballistic_coefficient, mass_kg, diameter_m,
                muzzle_velocity_mps, elevation_deg, azimuth_deg, muzzle_height_m,
                latitude_deg, wind_x, wind_y, wind_z, gravity,
                sample_distance_m, increment_m, used_cuda,
                target_range_m, time_of_flight_s, lateral_m, height_m,
                drop_from_muzzle_m, speed_mps, mach, energy_j,
                windage_moa, elevation_moa, windage_mils, elevation_mils
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    full_result.run_label,
                    caliber_name,
                    projectile.bc_model.upper(),
                    projectile.ballistic_coefficient,
                    projectile.mass_kg,
                    projectile.diameter_m,
                    launch.muzzle_velocity_mps,
                    launch.elevation_deg,
                    launch.azimuth_deg,
                    launch.muzzle_height_m,
                    env.latitude_deg,
                    env.wind_x,
                    env.wind_y,
                    env.wind_z,
                    env.gravity,
                    launch.sample_distance_m,
                    full_result.increment_m,
                    1 if full_result.used_cuda else 0,
                    row.target_range_m,
                    row.time_of_flight_s,
                    row.lateral_m,
                    row.height_m,
                    row.drop_from_muzzle_m,
                    row.speed_mps,
                    row.mach,
                    row.energy_j,
                    row.windage_moa,
                    row.elevation_moa,
                    row.windage_mils,
                    row.elevation_mils,
                )
                for row in full_result.rows
            ],
        )


def get_air_properties(env: Environment, altitude_m: float) -> tuple[float, float, float]:
    """Return density [kg/m^3], temperature [K], speed of sound [m/s]."""

    altitude = max(0.0, altitude_m)
    lapse = env.temperature_lapse_k_per_m
    gas_constant_dry_air = 287.05
    gamma = 1.4

    if altitude <= 11000.0:
        temperature = env.sea_level_temperature_k - lapse * altitude
        exponent = env.gravity / (gas_constant_dry_air * lapse)
        pressure = env.sea_level_pressure_pa * (temperature / env.sea_level_temperature_k) ** exponent
    else:
        tropopause_temperature = env.sea_level_temperature_k - lapse * 11000.0
        exponent = env.gravity / (gas_constant_dry_air * lapse)
        pressure_11 = env.sea_level_pressure_pa * (tropopause_temperature / env.sea_level_temperature_k) ** exponent
        temperature = tropopause_temperature
        pressure = pressure_11 * math.exp((-env.gravity * (altitude - 11000.0)) / (gas_constant_dry_air * temperature))

    density = pressure / (gas_constant_dry_air * temperature)
    speed_of_sound = math.sqrt(gamma * gas_constant_dry_air * temperature)
    return density, temperature, speed_of_sound


def compute_stability_factor(projectile: Projectile, env: Environment, muzzle_velocity_mps: float) -> float:
    diameter_in = projectile.diameter_m / 0.0254
    length_calibers = projectile.bullet_length_m / projectile.diameter_m
    twist_calibers = projectile.twist_rate_inches / diameter_in
    mass_grains = projectile.mass_kg * 15432.3583529
    temperature_f = (env.sea_level_temperature_k - 273.15) * 9.0 / 5.0 + 32.0
    pressure_inhg = env.sea_level_pressure_pa / 3386.389
    velocity_fps = muzzle_velocity_mps * 3.28084

    density_factor = (temperature_f + 460.0) / (59.0 + 460.0) * (29.92 / pressure_inhg)
    velocity_factor = (velocity_fps / 2800.0) ** (1.0 / 3.0)

    numerator = 30.0 * mass_grains
    denominator = (twist_calibers ** 2) * (diameter_in ** 3) * length_calibers * (1.0 + length_calibers ** 2)
    if denominator <= 0.0:
        return 0.0
    return (numerator / denominator) * density_factor * velocity_factor


def build_input_warnings(projectile: Projectile, launch: Launch) -> list[str]:
    warnings: list[str] = []

    if projectile.diameter_m >= 0.005 and projectile.mass_kg < 0.003:
        warnings.append(
            "Projectile mass is very low for this diameter. Check for a decimal error in kg input."
        )
    if launch.muzzle_velocity_mps > 2000.0:
        warnings.append(
            "Muzzle velocity is extremely high for small-arms ballistics. Check units and decimal placement."
        )
    if launch.muzzle_velocity_mps < 100.0:
        warnings.append(
            "Muzzle velocity is very low. Make sure the input is in meters per second."
        )
    if launch.elevation_deg > 15.0:
        warnings.append(
            "High launch elevation selected. For direct-fire rifle shots, elevations are usually much smaller."
        )
    if projectile.magnus_coefficient > 0.0:
        warnings.append(
            "Magnus coefficient is enabled. Nonzero values are experimental in this point-mass model and can increase drift."
        )

    return warnings


def compute_drag_force(projectile: Projectile, env: Environment, state: State, azimuth_deg: float) -> Vector3:
    wind_global = local_to_global((env.wind_x, env.wind_y, env.wind_z), azimuth_deg)
    velocity = (state.vx, state.vy, state.vz)
    relative_velocity = sub(velocity, wind_global)
    relative_speed = norm(relative_velocity)
    if relative_speed < EPSILON:
        return (0.0, 0.0, 0.0)

    density, _, speed_of_sound = get_air_properties(env, state.y)
    mach = relative_speed / max(speed_of_sound, EPSILON)
    cd_reference = get_drag_coefficient(mach, projectile.bc_model)
    sectional_density = projectile.sectional_density_lb_per_in2
    form_factor = sectional_density / projectile.ballistic_coefficient
    effective_cd = cd_reference * form_factor

    drag_magnitude = 0.5 * density * effective_cd * projectile.area_m2 * relative_speed * relative_speed
    return scale(unit(relative_velocity), -drag_magnitude)


def compute_magnus_force(projectile: Projectile, env: Environment, state: State, azimuth_deg: float) -> Vector3:
    if abs(projectile.magnus_coefficient) < EPSILON:
        return (0.0, 0.0, 0.0)

    wind_global = local_to_global((env.wind_x, env.wind_y, env.wind_z), azimuth_deg)
    velocity = (state.vx, state.vy, state.vz)
    relative_velocity = sub(velocity, wind_global)
    speed = norm(relative_velocity)
    if speed < EPSILON:
        return (0.0, 0.0, 0.0)

    spin_global = local_to_global((state.omega_x, state.omega_y, state.omega_z), azimuth_deg)
    magnus_vector = cross(spin_global, relative_velocity)
    magnus_direction = unit(magnus_vector)
    if norm(magnus_direction) < EPSILON:
        return (0.0, 0.0, 0.0)

    density, _, _ = get_air_properties(env, state.y)
    spin_rate = norm(spin_global)
    v_hat = unit(relative_velocity)
    spin_parallel = scale(v_hat, dot(spin_global, v_hat))
    spin_perpendicular = sub(spin_global, spin_parallel)
    effective_spin_rate = norm(spin_perpendicular)
    if effective_spin_rate < EPSILON:
        return (0.0, 0.0, 0.0)

    spin_parameter = projectile.radius_m * effective_spin_rate / max(speed, EPSILON)
    magnus_gain = projectile.magnus_coefficient * clamp(spin_parameter, 0.0, 0.15)
    magnus_magnitude = 0.5 * density * projectile.area_m2 * speed * speed * magnus_gain
    return scale(magnus_direction, magnus_magnitude)


def compute_coriolis_acceleration(env: Environment, state: State) -> Vector3:
    latitude = math.radians(env.latitude_deg)
    omega = (0.0, EARTH_ROTATION_RATE * math.sin(latitude), EARTH_ROTATION_RATE * math.cos(latitude))
    velocity = (state.vx, state.vy, state.vz)
    return scale(cross(omega, velocity), -2.0)


def derivatives(projectile: Projectile, env: Environment, state: State, azimuth_deg: float) -> tuple[float, ...]:
    drag_force = compute_drag_force(projectile, env, state, azimuth_deg)
    magnus_force = compute_magnus_force(projectile, env, state, azimuth_deg)
    coriolis = compute_coriolis_acceleration(env, state)

    ax = (drag_force[0] + magnus_force[0]) / projectile.mass_kg + coriolis[0]
    ay = (drag_force[1] + magnus_force[1]) / projectile.mass_kg + coriolis[1] - env.gravity
    az = (drag_force[2] + magnus_force[2]) / projectile.mass_kg + coriolis[2]

    return (
        state.vx,
        state.vy,
        state.vz,
        ax,
        ay,
        az,
        -projectile.spin_decay_rate * state.omega_x,
        -projectile.spin_decay_rate * state.omega_y,
        -projectile.spin_decay_rate * state.omega_z,
    )


def rk4_step(projectile: Projectile, env: Environment, state: State, azimuth_deg: float, dt: float) -> State:
    k1 = derivatives(projectile, env, state, azimuth_deg)
    s2 = State(
        state.x + 0.5 * dt * k1[0],
        state.y + 0.5 * dt * k1[1],
        state.z + 0.5 * dt * k1[2],
        state.vx + 0.5 * dt * k1[3],
        state.vy + 0.5 * dt * k1[4],
        state.vz + 0.5 * dt * k1[5],
        state.omega_x + 0.5 * dt * k1[6],
        state.omega_y + 0.5 * dt * k1[7],
        state.omega_z + 0.5 * dt * k1[8],
    )
    k2 = derivatives(projectile, env, s2, azimuth_deg)

    s3 = State(
        state.x + 0.5 * dt * k2[0],
        state.y + 0.5 * dt * k2[1],
        state.z + 0.5 * dt * k2[2],
        state.vx + 0.5 * dt * k2[3],
        state.vy + 0.5 * dt * k2[4],
        state.vz + 0.5 * dt * k2[5],
        state.omega_x + 0.5 * dt * k2[6],
        state.omega_y + 0.5 * dt * k2[7],
        state.omega_z + 0.5 * dt * k2[8],
    )
    k3 = derivatives(projectile, env, s3, azimuth_deg)

    s4 = State(
        state.x + dt * k3[0],
        state.y + dt * k3[1],
        state.z + dt * k3[2],
        state.vx + dt * k3[3],
        state.vy + dt * k3[4],
        state.vz + dt * k3[5],
        state.omega_x + dt * k3[6],
        state.omega_y + dt * k3[7],
        state.omega_z + dt * k3[8],
    )
    k4 = derivatives(projectile, env, s4, azimuth_deg)

    return State(
        state.x + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        state.y + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        state.z + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
        state.vx + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]),
        state.vy + (dt / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]),
        state.vz + (dt / 6.0) * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]),
        state.omega_x + (dt / 6.0) * (k1[6] + 2.0 * k2[6] + 2.0 * k3[6] + k4[6]),
        state.omega_y + (dt / 6.0) * (k1[7] + 2.0 * k2[7] + 2.0 * k3[7] + k4[7]),
        state.omega_z + (dt / 6.0) * (k1[8] + 2.0 * k2[8] + 2.0 * k3[8] + k4[8]),
    )


def interpolate_state(a: State, b: State, ratio: float) -> State:
    return State(
        lerp(a.x, b.x, ratio),
        lerp(a.y, b.y, ratio),
        lerp(a.z, b.z, ratio),
        lerp(a.vx, b.vx, ratio),
        lerp(a.vy, b.vy, ratio),
        lerp(a.vz, b.vz, ratio),
        lerp(a.omega_x, b.omega_x, ratio),
        lerp(a.omega_y, b.omega_y, ratio),
        lerp(a.omega_z, b.omega_z, ratio),
    )


def make_trajectory_sample(
    state: State,
    time_s: float,
    env: Environment,
    projectile: Projectile,
    launch: Launch,
) -> TrajectorySample:
    velocity = (state.vx, state.vy, state.vz)
    speed = norm(velocity)
    _, _, speed_of_sound = get_air_properties(env, state.y)
    mach = speed / max(speed_of_sound, EPSILON)
    energy = 0.5 * projectile.mass_kg * speed * speed
    x_local, y_local, z_local = global_to_local((state.x, state.y, state.z), launch.azimuth_deg)

    return TrajectorySample(
        time_s=time_s,
        x_m=x_local,
        y_m=y_local,
        z_m=z_local,
        speed_mps=speed,
        mach=mach,
        energy_j=energy,
        drift_m=x_local,
        drop_m=launch.muzzle_height_m - y_local,
    )


def simulate(projectile: Projectile, env: Environment, launch: Launch, config: SolverConfig) -> SimulationResult:
    projectile.validate()
    env.validate()
    launch.validate()
    config.validate()

    elevation_rad = math.radians(launch.elevation_deg)
    azimuth_rad = math.radians(launch.azimuth_deg)

    horizontal_speed = launch.muzzle_velocity_mps * math.cos(elevation_rad)
    initial_vy = launch.muzzle_velocity_mps * math.sin(elevation_rad)
    initial_vx = horizontal_speed * math.sin(azimuth_rad)
    initial_vz = horizontal_speed * math.cos(azimuth_rad)

    axial_spin = (2.0 * math.pi / projectile.twist_m_per_turn) * launch.muzzle_velocity_mps
    state = State(
        x=0.0,
        y=launch.muzzle_height_m,
        z=0.0,
        vx=initial_vx,
        vy=initial_vy,
        vz=initial_vz,
        omega_x=0.0,
        omega_y=0.0,
        omega_z=axial_spin,
    )

    time_s = 0.0
    max_ordinate_m = launch.muzzle_height_m
    trajectory: list[TrajectorySample] = []
    sample_hit: Optional[TrajectorySample] = None
    next_output_time = 0.0
    terminated_by_ground_impact = False
    termination_reason = "max_time_reached"
    _, _, forward_hat = basis_from_azimuth_deg(launch.azimuth_deg)

    if config.record_trajectory:
        trajectory.append(make_trajectory_sample(state, time_s, env, projectile, launch))
        next_output_time += config.output_interval_s

    while time_s < config.max_time_s:
        previous_state = state
        previous_time = time_s
        state = rk4_step(projectile, env, state, launch.azimuth_deg, config.dt)
        time_s += config.dt
        max_ordinate_m = max(max_ordinate_m, state.y)

        previous_forward = dot((previous_state.x, previous_state.y, previous_state.z), forward_hat)
        current_forward = dot((state.x, state.y, state.z), forward_hat)

        if launch.sample_distance_m is not None and sample_hit is None:
            if previous_forward <= launch.sample_distance_m <= current_forward:
                span = current_forward - previous_forward
                ratio = 0.0 if abs(span) < EPSILON else (launch.sample_distance_m - previous_forward) / span
                hit_state = interpolate_state(previous_state, state, ratio)
                hit_time = lerp(previous_time, time_s, ratio)
                sample_hit = make_trajectory_sample(hit_state, hit_time, env, projectile, launch)

        while config.record_trajectory and next_output_time <= time_s:
            span = time_s - previous_time
            ratio = 0.0 if span < EPSILON else (next_output_time - previous_time) / span
            snapshot_state = interpolate_state(previous_state, state, ratio)
            trajectory.append(make_trajectory_sample(snapshot_state, next_output_time, env, projectile, launch))
            next_output_time += config.output_interval_s

        if state.y <= 0.0:
            y_span = state.y - previous_state.y
            ratio = 0.0 if abs(y_span) < EPSILON else (0.0 - previous_state.y) / y_span
            ratio = clamp(ratio, 0.0, 1.0)
            state = interpolate_state(previous_state, state, ratio)
            time_s = lerp(previous_time, time_s, ratio)
            terminated_by_ground_impact = True
            termination_reason = "ground_impact"

            if config.record_trajectory:
                trajectory.append(make_trajectory_sample(state, time_s, env, projectile, launch))
            break
    else:
        if config.record_trajectory:
            trajectory.append(make_trajectory_sample(state, time_s, env, projectile, launch))

    final_velocity = (state.vx, state.vy, state.vz)
    final_speed = norm(final_velocity)
    _, _, speed_of_sound = get_air_properties(env, state.y)
    final_mach = final_speed / max(speed_of_sound, EPSILON)
    final_energy = 0.5 * projectile.mass_kg * final_speed * final_speed

    x_local, y_local, z_local = global_to_local((state.x, state.y, state.z), launch.azimuth_deg)
    _, vy_local, vz_local = global_to_local(final_velocity, launch.azimuth_deg)
    impact_angle_deg = math.degrees(math.atan2(vy_local, vz_local))

    range_m = z_local
    lateral_m = x_local
    drop_from_muzzle_m = launch.muzzle_height_m - y_local
    windage_moa = (lateral_m / range_m) * MOA_PER_RAD if range_m > EPSILON else 0.0
    elevation_moa = (drop_from_muzzle_m / range_m) * MOA_PER_RAD if range_m > EPSILON else 0.0
    windage_mils = (lateral_m / range_m) * MILS_PER_RAD if range_m > EPSILON else 0.0
    elevation_mils = (drop_from_muzzle_m / range_m) * MILS_PER_RAD if range_m > EPSILON else 0.0

    result = SimulationResult(
        time_of_flight_s=time_s,
        range_m=range_m,
        lateral_m=lateral_m,
        impact_height_m=y_local,
        drop_from_muzzle_m=drop_from_muzzle_m,
        speed_mps=final_speed,
        impact_mach=final_mach,
        energy_j=final_energy,
        impact_angle_deg=impact_angle_deg,
        max_ordinate_m=max_ordinate_m,
        windage_moa=windage_moa,
        elevation_moa=elevation_moa,
        windage_mils=windage_mils,
        elevation_mils=elevation_mils,
        stability_factor_sg=compute_stability_factor(projectile, env, launch.muzzle_velocity_mps),
        terminated_by_ground_impact=terminated_by_ground_impact,
        termination_reason=termination_reason,
        sample_hit=sample_hit,
        trajectory=trajectory,
    )

    if config.verbose:
        print_report(result, projectile, env, launch, config)

    return result


def print_report(
    result: SimulationResult,
    projectile: Projectile,
    env: Environment,
    launch: Launch,
    config: SolverConfig,
) -> None:
    print(f"\n{PROGRAM_NAME}")
    print("=" * len(PROGRAM_NAME))
    print(f"Projectile: {projectile.mass_kg:.5f} kg | {projectile.diameter_m * 1000.0:.2f} mm | BC {projectile.bc_model.upper()}={projectile.ballistic_coefficient:.4f}")
    print(f"Launch: {launch.muzzle_velocity_mps:.2f} m/s | Elev {launch.elevation_deg:.3f} deg | Azimuth {launch.azimuth_deg:.3f} deg | Height {launch.muzzle_height_m:.2f} m")
    print(f"Environment: Lat {env.latitude_deg:.2f} deg | Wind (x,y,z)=({env.wind_x:.2f}, {env.wind_y:.2f}, {env.wind_z:.2f}) m/s | dt={config.dt:.6f} s")
    print("\nSummary")
    print(f"Time of flight: {result.time_of_flight_s:.4f} s")
    if result.terminated_by_ground_impact:
        print(f"Range: {result.range_m:.2f} m | Lateral drift: {result.lateral_m:.3f} m | Impact height: {result.impact_height_m:.3f} m")
        print(f"Drop from muzzle: {result.drop_from_muzzle_m:.3f} m | Max ordinate: {result.max_ordinate_m:.3f} m")
        print(f"Impact speed: {result.speed_mps:.2f} m/s | Mach: {result.impact_mach:.3f} | Energy: {result.energy_j:.2f} J")
        print(f"Impact angle: {result.impact_angle_deg:.3f} deg | Stability factor Sg: {result.stability_factor_sg:.3f}")
    else:
        print(f"Range at simulation end: {result.range_m:.2f} m | Lateral drift: {result.lateral_m:.3f} m | Height at simulation end: {result.impact_height_m:.3f} m")
        print(f"Vertical offset from muzzle: {-result.drop_from_muzzle_m:.3f} m | Max ordinate: {result.max_ordinate_m:.3f} m")
        print(f"Speed at simulation end: {result.speed_mps:.2f} m/s | Mach: {result.impact_mach:.3f} | Energy: {result.energy_j:.2f} J")
        print(f"Flight angle at simulation end: {result.impact_angle_deg:.3f} deg | Stability factor Sg: {result.stability_factor_sg:.3f}")
        print("Simulation ended before ground impact. Increase maximum time to compute true impact.")
    print(f"Scope adjustment: {result.windage_moa:.3f} MOA windage | {result.elevation_moa:.3f} MOA elevation")
    print(f"Scope adjustment: {result.windage_mils:.3f} mil windage | {result.elevation_mils:.3f} mil elevation")

    if result.sample_hit is not None:
        sample = result.sample_hit
        print("\nSample point")
        print(f"Distance: {sample.z_m:.2f} m | Drift: {sample.x_m:.3f} m | Height: {sample.y_m:.3f} m")
        print(f"Speed: {sample.speed_mps:.2f} m/s | Mach: {sample.mach:.3f} | Energy: {sample.energy_j:.2f} J")


def build_range_targets(max_distance_m: float, increment_m: float) -> list[float]:
    targets: list[float] = []
    current = increment_m
    while current < max_distance_m - EPSILON:
        targets.append(current)
        current += increment_m
    targets.append(max_distance_m)
    return targets


def interpolate_full_sim_rows_cpu(
    trajectory: list[TrajectorySample],
    targets: list[float],
) -> list[FullSimulationRow]:
    rows: list[FullSimulationRow] = []
    sample_index = 1

    for target in targets:
        while sample_index < len(trajectory) and trajectory[sample_index].z_m < target:
            sample_index += 1

        if sample_index >= len(trajectory):
            left = trajectory[-1]
            right = trajectory[-1]
        else:
            left = trajectory[sample_index - 1]
            right = trajectory[sample_index]

        span = right.z_m - left.z_m
        ratio = 0.0 if abs(span) < EPSILON else (target - left.z_m) / span
        time_of_flight_s = lerp(left.time_s, right.time_s, ratio)
        lateral_m = lerp(left.x_m, right.x_m, ratio)
        height_m = lerp(left.y_m, right.y_m, ratio)
        drop_m = lerp(left.drop_m, right.drop_m, ratio)
        speed_mps = lerp(left.speed_mps, right.speed_mps, ratio)
        mach = lerp(left.mach, right.mach, ratio)
        energy_j = lerp(left.energy_j, right.energy_j, ratio)

        rows.append(
            FullSimulationRow(
                target_range_m=target,
                time_of_flight_s=time_of_flight_s,
                lateral_m=lateral_m,
                height_m=height_m,
                drop_from_muzzle_m=drop_m,
                speed_mps=speed_mps,
                mach=mach,
                energy_j=energy_j,
                windage_moa=(lateral_m / target) * MOA_PER_RAD if target > EPSILON else 0.0,
                elevation_moa=(drop_m / target) * MOA_PER_RAD if target > EPSILON else 0.0,
                windage_mils=(lateral_m / target) * MILS_PER_RAD if target > EPSILON else 0.0,
                elevation_mils=(drop_m / target) * MILS_PER_RAD if target > EPSILON else 0.0,
            )
        )

    return rows


def interpolate_full_sim_rows_cuda(
    trajectory: list[TrajectorySample],
    targets: list[float],
) -> tuple[list[FullSimulationRow], bool]:
    if cuda is None or np is None:
        return interpolate_full_sim_rows_cpu(trajectory, targets), False

    try:
        if not cuda.is_available():
            return interpolate_full_sim_rows_cpu(trajectory, targets), False
    except Exception:
        return interpolate_full_sim_rows_cpu(trajectory, targets), False

    sample_ranges = np.array([sample.z_m for sample in trajectory], dtype=np.float64)
    sample_times = np.array([sample.time_s for sample in trajectory], dtype=np.float64)
    sample_drifts = np.array([sample.x_m for sample in trajectory], dtype=np.float64)
    sample_heights = np.array([sample.y_m for sample in trajectory], dtype=np.float64)
    sample_speeds = np.array([sample.speed_mps for sample in trajectory], dtype=np.float64)
    sample_machs = np.array([sample.mach for sample in trajectory], dtype=np.float64)
    sample_energies = np.array([sample.energy_j for sample in trajectory], dtype=np.float64)
    sample_drops = np.array([sample.drop_m for sample in trajectory], dtype=np.float64)
    target_array = np.array(targets, dtype=np.float64)

    out_times = np.zeros_like(target_array)
    out_drifts = np.zeros_like(target_array)
    out_heights = np.zeros_like(target_array)
    out_speeds = np.zeros_like(target_array)
    out_machs = np.zeros_like(target_array)
    out_energies = np.zeros_like(target_array)
    out_drops = np.zeros_like(target_array)

    try:
        threads_per_block = 128
        blocks_per_grid = (target_array.size + threads_per_block - 1) // threads_per_block
        interpolate_rows_cuda[blocks_per_grid, threads_per_block](
            target_array,
            sample_ranges,
            sample_times,
            sample_drifts,
            sample_heights,
            sample_speeds,
            sample_machs,
            sample_energies,
            sample_drops,
            out_times,
            out_drifts,
            out_heights,
            out_speeds,
            out_machs,
            out_energies,
            out_drops,
        )
        cuda.synchronize()
    except Exception:
        return interpolate_full_sim_rows_cpu(trajectory, targets), False

    rows = [
        FullSimulationRow(
            target_range_m=float(target_array[index]),
            time_of_flight_s=float(out_times[index]),
            lateral_m=float(out_drifts[index]),
            height_m=float(out_heights[index]),
            drop_from_muzzle_m=float(out_drops[index]),
            speed_mps=float(out_speeds[index]),
            mach=float(out_machs[index]),
            energy_j=float(out_energies[index]),
            windage_moa=(float(out_drifts[index]) / float(target_array[index])) * MOA_PER_RAD if target_array[index] > EPSILON else 0.0,
            elevation_moa=(float(out_drops[index]) / float(target_array[index])) * MOA_PER_RAD if target_array[index] > EPSILON else 0.0,
            windage_mils=(float(out_drifts[index]) / float(target_array[index])) * MILS_PER_RAD if target_array[index] > EPSILON else 0.0,
            elevation_mils=(float(out_drops[index]) / float(target_array[index])) * MILS_PER_RAD if target_array[index] > EPSILON else 0.0,
        )
        for index in range(target_array.size)
    ]
    return rows, True


def interpolate_trajectory_sample_by_range(
    trajectory: list[TrajectorySample],
    target_range_m: float,
) -> Optional[TrajectorySample]:
    if len(trajectory) < 2 or target_range_m < 0.0:
        return None
    if target_range_m > trajectory[-1].z_m + EPSILON:
        return None

    for index in range(1, len(trajectory)):
        left = trajectory[index - 1]
        right = trajectory[index]
        if target_range_m <= right.z_m:
            span = right.z_m - left.z_m
            ratio = 0.0 if abs(span) < EPSILON else (target_range_m - left.z_m) / span
            return TrajectorySample(
                time_s=lerp(left.time_s, right.time_s, ratio),
                x_m=lerp(left.x_m, right.x_m, ratio),
                y_m=lerp(left.y_m, right.y_m, ratio),
                z_m=lerp(left.z_m, right.z_m, ratio),
                speed_mps=lerp(left.speed_mps, right.speed_mps, ratio),
                mach=lerp(left.mach, right.mach, ratio),
                energy_j=lerp(left.energy_j, right.energy_j, ratio),
                drift_m=lerp(left.drift_m, right.drift_m, ratio),
                drop_m=lerp(left.drop_m, right.drop_m, ratio),
            )
    return trajectory[-1]


def run_full_simulation(
    projectile: Projectile,
    env: Environment,
    launch: Launch,
    single_result: SimulationResult,
    increment_m: float,
    run_label: str,
) -> FullSimulationResult:
    if launch.sample_distance_m is None:
        raise ValueError("Full simulation requires a sample distance.")
    if increment_m <= 0.0:
        raise ValueError("Full simulation increment must be positive.")
    if single_result.sample_hit is None:
        raise ValueError("Single-run sample point was not reached. Increase max time or reduce sample distance.")

    max_distance_m = launch.sample_distance_m
    targets = build_range_targets(max_distance_m, increment_m)
    rows, used_cuda = interpolate_full_sim_rows_cuda(single_result.trajectory, targets)

    return FullSimulationResult(
        run_label=run_label,
        increment_m=increment_m,
        used_cuda=used_cuda,
        rows=rows,
        sqlite_path=DATABASE_PATH,
    )


def print_full_simulation_report(full_result: FullSimulationResult) -> None:
    print("\nFull simulation results")
    print("-----------------------")
    print(f"Rows: {len(full_result.rows)} | Increment: {full_result.increment_m:.2f} m | CUDA: {'yes' if full_result.used_cuda else 'no (CPU fallback)'}")
    print("Range(m) | TOF(s) | Drift(m) | Height(m) | Speed(m/s) | Energy(J) | Elev(MOA) | Wind(MOA)")
    for row in full_result.rows:
        print(
            f"{row.target_range_m:8.1f} | "
            f"{row.time_of_flight_s:6.3f} | "
            f"{row.lateral_m:8.3f} | "
            f"{row.height_m:9.3f} | "
            f"{row.speed_mps:10.2f} | "
            f"{row.energy_j:9.2f} | "
            f"{row.elevation_moa:9.3f} | "
            f"{row.windage_moa:9.3f}"
        )


def format_adjustment_direction(value: float, positive_label: str, negative_label: str, zero_label: str = "HOLD") -> str:
    if abs(value) < 1e-9:
        return zero_label
    return positive_label if value > 0.0 else negative_label


def simulate_sample_point(
    projectile: Projectile,
    env: Environment,
    launch: Launch,
    config: SolverConfig,
    target_forward_m: float,
    reference_azimuth_deg: Optional[float] = None,
) -> Optional[TrajectorySample]:
    reference_azimuth = launch.azimuth_deg if reference_azimuth_deg is None else reference_azimuth_deg
    config.validate()

    elevation_rad = math.radians(launch.elevation_deg)
    azimuth_rad = math.radians(launch.azimuth_deg)
    horizontal_speed = launch.muzzle_velocity_mps * math.cos(elevation_rad)
    initial_vy = launch.muzzle_velocity_mps * math.sin(elevation_rad)
    initial_vx = horizontal_speed * math.sin(azimuth_rad)
    initial_vz = horizontal_speed * math.cos(azimuth_rad)
    axial_spin = (2.0 * math.pi / projectile.twist_m_per_turn) * launch.muzzle_velocity_mps

    state = State(
        x=0.0,
        y=launch.muzzle_height_m,
        z=0.0,
        vx=initial_vx,
        vy=initial_vy,
        vz=initial_vz,
        omega_x=0.0,
        omega_y=0.0,
        omega_z=axial_spin,
    )

    time_s = 0.0
    previous_local = global_to_local((state.x, state.y, state.z), reference_azimuth)

    while time_s < config.max_time_s and state.y > 0.0:
        previous_state = state
        previous_time = time_s
        state = rk4_step(projectile, env, state, launch.azimuth_deg, config.dt)
        time_s += config.dt

        current_local = global_to_local((state.x, state.y, state.z), reference_azimuth)
        previous_forward = previous_local[2]
        current_forward = current_local[2]

        if previous_forward <= target_forward_m <= current_forward:
            span = current_forward - previous_forward
            ratio = 0.0 if abs(span) < EPSILON else (target_forward_m - previous_forward) / span
            hit_state = interpolate_state(previous_state, state, ratio)
            hit_time = lerp(previous_time, time_s, ratio)
            return make_trajectory_sample(hit_state, hit_time, env, projectile, Launch(
                muzzle_velocity_mps=launch.muzzle_velocity_mps,
                elevation_deg=launch.elevation_deg,
                azimuth_deg=reference_azimuth,
                muzzle_height_m=launch.muzzle_height_m,
                sample_distance_m=target_forward_m,
            ))

        previous_local = current_local

    return None


def solve_linear_2x2(a11: float, a12: float, a21: float, a22: float, b1: float, b2: float) -> Optional[tuple[float, float]]:
    determinant = a11 * a22 - a12 * a21
    if abs(determinant) < 1e-12:
        return None
    x = (b1 * a22 - b2 * a12) / determinant
    y = (a11 * b2 - a21 * b1) / determinant
    return x, y


def solve_zero_elevation(
    projectile: Projectile,
    env: Environment,
    launch: Launch,
    rifle: RifleProfile,
    config: SolverConfig,
) -> Optional[float]:
    if rifle.zero_range_m is None:
        return launch.elevation_deg

    target_forward_m = rifle.zero_range_m
    target_height_m = launch.muzzle_height_m + rifle.sight_height_m
    reference_azimuth = launch.azimuth_deg

    def height_error(elevation_deg: float) -> Optional[float]:
        sample = simulate_sample_point(
            projectile,
            env,
            Launch(
                muzzle_velocity_mps=launch.muzzle_velocity_mps,
                elevation_deg=elevation_deg,
                azimuth_deg=reference_azimuth,
                muzzle_height_m=launch.muzzle_height_m,
                sample_distance_m=target_forward_m,
            ),
            config,
            target_forward_m,
            reference_azimuth,
        )
        if sample is None:
            return None
        return sample.y_m - target_height_m

    scan_points = [(-2.0 + 0.25 * index) for index in range(int((8.0 - (-2.0)) / 0.25) + 1)]
    lower = None
    upper = None
    err_low = None
    err_high = None
    previous_angle = None
    previous_error = None

    for angle in scan_points:
        error = height_error(angle)
        if error is None:
            continue
        if previous_angle is not None and previous_error is not None and previous_error * error <= 0.0:
            lower = previous_angle
            upper = angle
            err_low = previous_error
            err_high = error
            break
        previous_angle = angle
        previous_error = error

    if lower is None or upper is None or err_low is None or err_high is None:
        return None

    for _ in range(28):
        mid = 0.5 * (lower + upper)
        err_mid = height_error(mid)
        if err_mid is None:
            return None
        if abs(err_mid) < 1e-4:
            return mid
        if err_low * err_mid <= 0.0:
            upper = mid
            err_high = err_mid
        else:
            lower = mid
            err_low = err_mid

    return 0.5 * (lower + upper)


def reverse_solve_target(
    projectile: Projectile,
    env: Environment,
    launch: Launch,
    rifle: RifleProfile,
    optic: OpticProfile,
    config: SolverConfig,
    target_forward_m: float,
    target_height_m: float,
    target_lateral_m: float,
) -> ReverseSolution:
    if target_forward_m <= 0.0:
        return ReverseSolution(False, 0, target_forward_m, target_height_m, target_lateral_m, launch.elevation_deg, launch.azimuth_deg, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Target distance must be positive.")

    reference_azimuth = launch.azimuth_deg
    initial_elevation_guess = math.degrees(math.atan2(target_height_m - launch.muzzle_height_m, target_forward_m))
    initial_azimuth_guess = reference_azimuth + math.degrees(math.atan2(target_lateral_m, target_forward_m))
    current_elevation = initial_elevation_guess
    current_azimuth = initial_azimuth_guess
    last_lateral_error = 0.0
    last_vertical_error = 0.0
    max_iterations = 16
    perturb_deg = 0.05

    for iteration in range(1, max_iterations + 1):
        base_launch = Launch(
            muzzle_velocity_mps=launch.muzzle_velocity_mps,
            elevation_deg=current_elevation,
            azimuth_deg=current_azimuth,
            muzzle_height_m=launch.muzzle_height_m,
            sample_distance_m=target_forward_m,
        )
        base_sample = simulate_sample_point(projectile, env, base_launch, config, target_forward_m, reference_azimuth)
        if base_sample is None:
            return ReverseSolution(
                False, iteration, target_forward_m, target_height_m, target_lateral_m,
                current_elevation, current_azimuth, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                "Solver could not reach the requested target distance. Increase max time or reduce target distance."
            )

        lateral_error = base_sample.x_m - target_lateral_m
        vertical_error = base_sample.y_m - target_height_m
        last_lateral_error = lateral_error
        last_vertical_error = vertical_error

        if abs(lateral_error) < 0.01 and abs(vertical_error) < 0.01:
            break

        elev_launch = Launch(
            muzzle_velocity_mps=launch.muzzle_velocity_mps,
            elevation_deg=current_elevation + perturb_deg,
            azimuth_deg=current_azimuth,
            muzzle_height_m=launch.muzzle_height_m,
            sample_distance_m=target_forward_m,
        )
        az_launch = Launch(
            muzzle_velocity_mps=launch.muzzle_velocity_mps,
            elevation_deg=current_elevation,
            azimuth_deg=current_azimuth + perturb_deg,
            muzzle_height_m=launch.muzzle_height_m,
            sample_distance_m=target_forward_m,
        )
        elev_sample = simulate_sample_point(projectile, env, elev_launch, config, target_forward_m, reference_azimuth)
        az_sample = simulate_sample_point(projectile, env, az_launch, config, target_forward_m, reference_azimuth)

        if elev_sample is None or az_sample is None:
            return ReverseSolution(
                False, iteration, target_forward_m, target_height_m, target_lateral_m,
                current_elevation, current_azimuth, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                lateral_error, vertical_error,
                "Solver could not compute the Jacobian near the target solution."
            )

        d_lat_delev = (elev_sample.x_m - base_sample.x_m) / perturb_deg
        d_lat_daz = (az_sample.x_m - base_sample.x_m) / perturb_deg
        d_vert_delev = (elev_sample.y_m - base_sample.y_m) / perturb_deg
        d_vert_daz = (az_sample.y_m - base_sample.y_m) / perturb_deg

        delta = solve_linear_2x2(
            d_lat_delev,
            d_lat_daz,
            d_vert_delev,
            d_vert_daz,
            -lateral_error,
            -vertical_error,
        )
        if delta is None:
            return ReverseSolution(
                False, iteration, target_forward_m, target_height_m, target_lateral_m,
                current_elevation, current_azimuth, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                lateral_error, vertical_error,
                "Solver Jacobian became singular. Try a different initial guess."
            )

        delta_elevation_deg, delta_azimuth_deg = delta
        current_elevation += clamp(delta_elevation_deg, -5.0, 5.0)
        current_azimuth += clamp(delta_azimuth_deg, -5.0, 5.0)
    if abs(last_lateral_error) >= 0.01 or abs(last_vertical_error) >= 0.01:
        return ReverseSolution(
            False, max_iterations, target_forward_m, target_height_m, target_lateral_m,
            current_elevation, current_azimuth, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            last_lateral_error, last_vertical_error,
            "Solver did not converge within the iteration limit."
        )

    zero_elevation_deg = solve_zero_elevation(projectile, env, launch, rifle, config)
    if zero_elevation_deg is None:
        zero_elevation_deg = launch.elevation_deg
    elevation_delta_rad = math.radians(current_elevation - zero_elevation_deg)
    windage_delta_rad = math.radians(current_azimuth - reference_azimuth)
    elevation_moa = elevation_delta_rad * MOA_PER_RAD
    elevation_mils = elevation_delta_rad * MILS_PER_RAD
    windage_moa = windage_delta_rad * MOA_PER_RAD
    windage_mils = windage_delta_rad * MILS_PER_RAD
    click_basis_elevation = elevation_moa if optic.click_unit.upper() == "MOA" else elevation_mils
    click_basis_windage = windage_moa if optic.click_unit.upper() == "MOA" else windage_mils
    final_sample = simulate_sample_point(
        projectile,
        env,
        Launch(
            muzzle_velocity_mps=launch.muzzle_velocity_mps,
            elevation_deg=current_elevation,
            azimuth_deg=current_azimuth,
            muzzle_height_m=launch.muzzle_height_m,
            sample_distance_m=target_forward_m,
        ),
        config,
        target_forward_m,
        reference_azimuth,
    )
    if final_sample is None:
        return ReverseSolution(
            False, iteration, target_forward_m, target_height_m, target_lateral_m,
            current_elevation, current_azimuth, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            last_vertical_error, last_lateral_error, 0.0, 0.0, 0.0, 0.0,
            "Solver converged mathematically but failed to recover the final target sample."
        )

    return ReverseSolution(
        solved=True,
        iterations=iteration,
        target_forward_m=target_forward_m,
        target_height_m=target_height_m,
        target_lateral_m=target_lateral_m,
        solved_elevation_deg=current_elevation,
        solved_azimuth_deg=current_azimuth,
        elevation_adjustment_moa=elevation_moa,
        elevation_adjustment_mils=elevation_mils,
        windage_adjustment_moa=windage_moa,
        windage_adjustment_mils=windage_mils,
        elevation_clicks=compute_scope_clicks(click_basis_elevation, optic),
        windage_clicks=compute_scope_clicks(click_basis_windage, optic),
        final_vertical_error_m=last_vertical_error,
        final_lateral_error_m=last_lateral_error,
        target_time_s=final_sample.time_s,
        target_speed_mps=final_sample.speed_mps,
        target_mach=final_sample.mach,
        target_energy_j=final_sample.energy_j,
        target_drop_from_muzzle_m=final_sample.drop_m,
        message="Reverse solution converged.",
    )


def print_reverse_solution(solution: ReverseSolution, optic: OpticProfile) -> None:
    print("\nReverse Solution")
    print("----------------")
    print(
        f"Target: forward {solution.target_forward_m:.2f} m | "
        f"height {solution.target_height_m:.2f} m | "
        f"lateral {solution.target_lateral_m:.2f} m"
    )
    print(solution.message)
    if not solution.solved:
        print(
            f"Last errors -> lateral: {solution.final_lateral_error_m:.3f} m | "
            f"vertical: {solution.final_vertical_error_m:.3f} m"
        )
        return

    print(f"Solved launch elevation: {solution.solved_elevation_deg:.4f} deg")
    print(f"Solved launch azimuth:   {solution.solved_azimuth_deg:.4f} deg")
    print(f"Elevation adjustment: {solution.elevation_adjustment_moa:.3f} MOA | {solution.elevation_adjustment_mils:.3f} mil")
    print(f"Windage adjustment:  {solution.windage_adjustment_moa:.3f} MOA | {solution.windage_adjustment_mils:.3f} mil")
    elevation_direction = format_adjustment_direction(solution.elevation_clicks, "UP", "DOWN")
    windage_direction = format_adjustment_direction(solution.windage_clicks, "RIGHT", "LEFT")
    print(
        f"{optic.click_unit.upper()} clicks -> elevation {elevation_direction} {abs(solution.elevation_clicks):.1f} | "
        f"windage {windage_direction} {abs(solution.windage_clicks):.1f}"
    )
    print(
        f"At target -> TOF {solution.target_time_s:.3f} s | "
        f"speed {solution.target_speed_mps:.2f} m/s | "
        f"Mach {solution.target_mach:.3f} | "
        f"energy {solution.target_energy_j:.2f} J | "
        f"drop from muzzle {solution.target_drop_from_muzzle_m:.3f} m"
    )
    print(
        f"Residual error -> lateral {solution.final_lateral_error_m:.3f} m | "
        f"vertical {solution.final_vertical_error_m:.3f} m | iterations {solution.iterations}"
    )


def render_terminal_trajectory_animation(
    result: SimulationResult,
    width: int = 72,
    height: int = 20,
    frame_count: int = 80,
    frame_delay_s: float = 0.035,
    target_forward_m: Optional[float] = None,
    target_height_m: Optional[float] = None,
    vertical_compression: float = 0.45,
) -> None:
    if len(result.trajectory) < 2:
        print("\nTrajectory animation unavailable: not enough recorded samples.")
        return

    width = max(30, width)
    height = max(10, height)
    frame_count = max(2, frame_count)
    frame_delay_s = max(0.0, frame_delay_s)

    samples = result.trajectory
    distances = [sample.z_m for sample in samples]
    heights = [sample.y_m for sample in samples]
    drifts = [sample.x_m for sample in samples]

    min_distance = min(distances)
    max_distance = max(distances)
    min_height = min(0.0, min(heights))
    max_height = max(heights)
    if target_forward_m is not None:
        max_distance = max(max_distance, target_forward_m)
    if target_height_m is not None:
        max_height = max(max_height, target_height_m)
        min_height = min(min_height, target_height_m)
    x_span = max(max_distance - min_distance, 1.0)
    y_span = max(max_height - min_height, 1.0)
    effective_y_span = max(y_span / max(vertical_compression, 0.05), y_span)
    y_mid = 0.5 * (max_height + min_height)
    min_height = y_mid - 0.5 * effective_y_span
    max_height = y_mid + 0.5 * effective_y_span
    y_span = max(max_height - min_height, 1.0)

    def x_to_col(distance: float) -> int:
        ratio = (distance - min_distance) / x_span
        return int(round(clamp(ratio, 0.0, 1.0) * (width - 1)))

    def y_to_row(height_m: float) -> int:
        ratio = (height_m - min_height) / y_span
        return int(round((1.0 - clamp(ratio, 0.0, 1.0)) * (height - 1)))

    def sample_at_time(target_time: float) -> TrajectorySample:
        if target_time <= samples[0].time_s:
            return samples[0]
        if target_time >= samples[-1].time_s:
            return samples[-1]

        for index in range(1, len(samples)):
            a = samples[index - 1]
            b = samples[index]
            if target_time <= b.time_s:
                span = max(b.time_s - a.time_s, EPSILON)
                ratio = (target_time - a.time_s) / span
                return TrajectorySample(
                    time_s=lerp(a.time_s, b.time_s, ratio),
                    x_m=lerp(a.x_m, b.x_m, ratio),
                    y_m=lerp(a.y_m, b.y_m, ratio),
                    z_m=lerp(a.z_m, b.z_m, ratio),
                    speed_mps=lerp(a.speed_mps, b.speed_mps, ratio),
                    mach=lerp(a.mach, b.mach, ratio),
                    energy_j=lerp(a.energy_j, b.energy_j, ratio),
                    drift_m=lerp(a.drift_m, b.drift_m, ratio),
                    drop_m=lerp(a.drop_m, b.drop_m, ratio),
                )
        return samples[-1]

    ground_row = y_to_row(0.0)
    final_time = samples[-1].time_s
    target_col = x_to_col(target_forward_m) if target_forward_m is not None else None
    target_row = y_to_row(target_height_m) if target_height_m is not None else None

    print("\nTrajectory animation")
    print("Press Ctrl+C to stop playback.\n")

    try:
        for frame_index in range(frame_count + 1):
            target_time = final_time * frame_index / frame_count
            current = sample_at_time(target_time)
            current_col = x_to_col(current.z_m)
            current_row = y_to_row(current.y_m)

            grid = [[" " for _ in range(width)] for _ in range(height)]

            for col in range(width):
                grid[ground_row][col] = "_"

            for sample in samples:
                if sample.time_s > current.time_s:
                    break
                col = x_to_col(sample.z_m)
                row = y_to_row(sample.y_m)
                if 0 <= row < height and 0 <= col < width:
                    grid[row][col] = "."

            if target_row is not None and target_col is not None:
                if 0 <= target_row < height and 0 <= target_col < width:
                    grid[target_row][target_col] = "X"

            if 0 <= current_row < height and 0 <= current_col < width:
                grid[current_row][current_col] = "O"

            lines = ["".join(row) for row in grid]
            status = (
                f"t={current.time_s:5.3f}s  "
                f"range={current.z_m:7.1f}m  "
                f"height={current.y_m:6.2f}m  "
                f"drift={current.x_m:6.3f}m  "
                f"speed={current.speed_mps:7.1f}m/s"
            )
            if target_forward_m is not None and target_height_m is not None:
                status += f"  target=({target_forward_m:.1f}m,{target_height_m:.2f}m)"
            scale = f"0m{' ' * max(1, width - 10)}{max_distance:.0f}m"

            print("\x1b[2J\x1b[H", end="")
            print(status)
            print(scale[:width])
            for line in lines:
                print(line)
            time.sleep(frame_delay_s)
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")


def build_3d_animation_html(
    result: SimulationResult,
    target_forward_m: Optional[float] = None,
    target_height_m: Optional[float] = None,
    target_lateral_m: Optional[float] = None,
    title: str = "3D Trajectory Animation",
) -> str:
    if len(result.trajectory) < 2:
        raise ValueError("3D animation requires recorded trajectory samples.")

    samples_payload = [
        {
            "t": round(sample.time_s, 6),
            "x": round(sample.x_m, 6),
            "y": round(sample.y_m, 6),
            "z": round(sample.z_m, 6),
            "s": round(sample.speed_mps, 6),
            "m": round(sample.mach, 6),
            "e": round(sample.energy_j, 6),
        }
        for sample in result.trajectory
    ]
    target_payload = {
        "x": 0.0 if target_lateral_m is None else round(target_lateral_m, 6),
        "y": 0.0 if target_height_m is None else round(target_height_m, 6),
        "z": 0.0 if target_forward_m is None else round(target_forward_m, 6),
        "enabled": target_forward_m is not None and target_height_m is not None,
    }

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: Consolas, "SF Mono", monospace;
      background: #0f1720;
      color: #e8f0f7;
      overflow: hidden;
    }}
    .wrap {{
      position: relative;
      min-height: 100vh;
    }}
    canvas {{
      width: 100vw;
      height: 100vh;
      display: block;
      background:
        radial-gradient(circle at top, rgba(44, 98, 140, 0.25), transparent 35%),
        linear-gradient(180deg, #111a22 0%, #0b1117 100%);
    }}
    .panel {{
      position: absolute;
      top: 18px;
      right: 18px;
      width: min(360px, calc(100vw - 36px));
      padding: 20px;
      background: rgba(8, 12, 17, 0.88);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 18px;
      backdrop-filter: blur(10px);
      box-shadow: 0 18px 40px rgba(0,0,0,0.35);
    }}
    .panel h1 {{
      margin: 0 0 10px;
      font-size: 1.25rem;
    }}
    .panel p {{
      margin: 0 0 16px;
      color: #9eb0bf;
      line-height: 1.5;
    }}
    .metric {{
      margin: 0 0 14px;
      padding: 12px 14px;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px;
      background: rgba(255,255,255,0.03);
    }}
    .metric .label {{
      display: block;
      font-size: 0.75rem;
      color: #8aa1b3;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric .value {{
      font-size: 1.15rem;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 20px;
    }}
    .sliders {{
      display: grid;
      gap: 10px;
      margin-top: 18px;
    }}
    .sliders label {{
      display: grid;
      gap: 6px;
      font-size: 0.9rem;
      color: #c4d3de;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      background: #2a88c9;
      color: white;
      font: inherit;
      cursor: pointer;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    .angle-row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 16px;
    }}
    .angle-row button {{
      background: #203244;
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .footer-note {{
      margin-top: 12px;
      font-size: 0.82rem;
      color: #8aa1b3;
    }}
    .camera-pad {{
      display: grid;
      grid-template-columns: repeat(3, 44px);
      gap: 8px;
      margin-top: 16px;
      align-items: center;
      justify-content: start;
    }}
    .camera-pad button {{
      width: 44px;
      height: 44px;
      padding: 0;
      background: #1d2a36;
      border: 1px solid rgba(255,255,255,0.08);
      font-size: 1rem;
    }}
    .zoom-row {{
      display: flex;
      gap: 10px;
      margin-top: 12px;
    }}
    .zoom-row button {{
      background: #1d2a36;
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .bullet-tag {{
      position: absolute;
      min-width: 180px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(8, 12, 17, 0.90);
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: 0 14px 30px rgba(0,0,0,0.28);
      pointer-events: none;
      transform: translate(-50%, -120%);
      backdrop-filter: blur(8px);
    }}
    .bullet-tag .tag-title {{
      display: block;
      font-size: 0.72rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #8aa1b3;
      margin-bottom: 6px;
    }}
    .bullet-tag .tag-line {{
      font-size: 0.88rem;
      color: #edf4fb;
      line-height: 1.4;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <canvas id="scene"></canvas>
    <div class="bullet-tag" id="bulletTag">
      <span class="tag-title">Projectile</span>
      <div class="tag-line" id="bulletTagLine">t 0.000 s | 0.0 m</div>
    </div>
    <div class="panel">
      <h1>{title}</h1>
      <p>Self-contained 3D playback of the bullet flight path. Blue line is the trail, orange dot is the projectile, red marker is the target.</p>
      <div class="metric"><span class="label">Time</span><span class="value" id="timeValue">0.000 s</span></div>
      <div class="metric"><span class="label">Forward</span><span class="value" id="rangeValue">0.0 m</span></div>
      <div class="metric"><span class="label">Height</span><span class="value" id="heightValue">0.0 m</span></div>
      <div class="metric"><span class="label">Lateral</span><span class="value" id="driftValue">0.0 m</span></div>
      <div class="metric"><span class="label">Speed</span><span class="value" id="speedValue">0.0 m/s</span></div>
      <div class="metric"><span class="label">Energy</span><span class="value" id="energyValue">0 J</span></div>
      <div class="controls">
        <button id="toggle">Pause</button>
        <label>Speed <input id="speed" type="range" min="0.25" max="4" step="0.25" value="1"></label>
      </div>
      <div class="sliders">
        <label>Yaw
          <input id="yaw" type="range" min="-180" max="180" step="1" value="49">
        </label>
        <label>Pitch
          <input id="pitch" type="range" min="-80" max="80" step="1" value="32">
        </label>
        <label>Zoom
          <input id="zoom" type="range" min="0.6" max="2.2" step="0.05" value="1.2">
        </label>
      </div>
      <div class="angle-row">
        <button id="viewIso">Isometric</button>
        <button id="viewSide">Side</button>
        <button id="viewTop">Top</button>
        <button id="viewFront">Front</button>
      </div>
      <div class="camera-pad">
        <span></span>
        <button id="camUp">W</button>
        <span></span>
        <button id="camLeft">A</button>
        <button id="camDown">S</button>
        <button id="camRight">D</button>
      </div>
      <div class="zoom-row">
        <button id="zoomIn">Zoom +</button>
        <button id="zoomOut">Zoom -</button>
      </div>
      <div class="footer-note">Tip: use the sliders, WASD keys, arrow keys, or the camera pad to inspect drift, arc, and target alignment.</div>
    </div>
  </div>
  <script>
    const samples = {json.dumps(samples_payload)};
    const target = {json.dumps(target_payload)};
    const canvas = document.getElementById('scene');
    const ctx = canvas.getContext('2d');
    const timeValue = document.getElementById('timeValue');
    const rangeValue = document.getElementById('rangeValue');
    const heightValue = document.getElementById('heightValue');
    const driftValue = document.getElementById('driftValue');
    const speedValue = document.getElementById('speedValue');
    const energyValue = document.getElementById('energyValue');
    const toggle = document.getElementById('toggle');
    const speedSlider = document.getElementById('speed');
    const yawSlider = document.getElementById('yaw');
    const pitchSlider = document.getElementById('pitch');
    const zoomSlider = document.getElementById('zoom');
    const viewIso = document.getElementById('viewIso');
    const viewSide = document.getElementById('viewSide');
    const viewTop = document.getElementById('viewTop');
    const viewFront = document.getElementById('viewFront');
    const camUp = document.getElementById('camUp');
    const camLeft = document.getElementById('camLeft');
    const camDown = document.getElementById('camDown');
    const camRight = document.getElementById('camRight');
    const zoomIn = document.getElementById('zoomIn');
    const zoomOut = document.getElementById('zoomOut');
    const bulletTag = document.getElementById('bulletTag');
    const bulletTagLine = document.getElementById('bulletTagLine');

    let playing = true;
    let playback = 0;
    let lastFrame = null;
    const totalDuration = samples[samples.length - 1].t || 1;

    function resize() {{
      canvas.width = canvas.clientWidth * devicePixelRatio;
      canvas.height = canvas.clientHeight * devicePixelRatio;
      ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    }}
    window.addEventListener('resize', resize);
    resize();

    const maxForward = Math.max(...samples.map(s => s.z), target.enabled ? target.z : 0);
    const maxHeight = Math.max(...samples.map(s => s.y), target.enabled ? target.y : 0, 1);
    const minHeight = Math.min(0, ...samples.map(s => s.y), target.enabled ? target.y : 0);
    const maxLateral = Math.max(...samples.map(s => Math.abs(s.x)), target.enabled ? Math.abs(target.x) : 0, 1);

    function interpolateSample(time) {{
      if (time <= samples[0].t) return samples[0];
      if (time >= samples[samples.length - 1].t) return samples[samples.length - 1];
      for (let i = 1; i < samples.length; i += 1) {{
        const a = samples[i - 1];
        const b = samples[i];
        if (time <= b.t) {{
          const span = Math.max(b.t - a.t, 1e-9);
          const r = (time - a.t) / span;
          return {{
            t: a.t + (b.t - a.t) * r,
            x: a.x + (b.x - a.x) * r,
            y: a.y + (b.y - a.y) * r,
            z: a.z + (b.z - a.z) * r,
            s: a.s + (b.s - a.s) * r,
            m: a.m + (b.m - a.m) * r,
            e: a.e + (b.e - a.e) * r
          }};
        }}
      }}
      return samples[samples.length - 1];
    }}

    function rawProject(point, yaw, pitch) {{
      const x = (point.x / maxLateral) * 110;
      const y = ((point.y - minHeight) / Math.max(maxHeight - minHeight, 1e-6)) * 180;
      const z = (point.z / Math.max(maxForward, 1e-6)) * 520;

      const x1 = x * Math.cos(yaw) - z * Math.sin(yaw);
      const z1 = x * Math.sin(yaw) + z * Math.cos(yaw);
      const y1 = y * Math.cos(pitch) - z1 * Math.sin(pitch);
      return {{
        x: x1,
        y: -y1,
      }};
    }}

    function getView() {{
      return {{
        yaw: Number(yawSlider.value) * Math.PI / 180,
        pitch: Number(pitchSlider.value) * Math.PI / 180,
        zoom: Number(zoomSlider.value),
      }};
    }}

    function computeTransform(view) {{
      const points = samples.map(s => rawProject(s, view.yaw, view.pitch));
      if (target.enabled) points.push(rawProject(target, view.yaw, view.pitch));
      const minX = Math.min(...points.map(p => p.x));
      const maxX = Math.max(...points.map(p => p.x));
      const minY = Math.min(...points.map(p => p.y));
      const maxY = Math.max(...points.map(p => p.y));
      const spanX = Math.max(maxX - minX, 1);
      const spanY = Math.max(maxY - minY, 1);
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const fitScale = Math.min((width * 0.78) / spanX, (height * 0.78) / spanY) * view.zoom;
      return {{
        scale: fitScale,
        offsetX: width / 2 - ((minX + maxX) / 2) * fitScale,
        offsetY: height / 2 - ((minY + maxY) / 2) * fitScale,
      }};
    }}

    function project(point, view, transform) {{
      const raw = rawProject(point, view.yaw, view.pitch);
      return {{
        x: raw.x * transform.scale + transform.offsetX,
        y: raw.y * transform.scale + transform.offsetY,
      }};
    }}

    function drawGrid(view, transform) {{
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.lineWidth = 1;
      ctx.fillStyle = 'rgba(210,225,238,0.72)';
      ctx.font = '12px Consolas, monospace';
      for (let i = 0; i <= 6; i += 1) {{
        const z = maxForward * (i / 6);
        const a = project({{x: -maxLateral, y: 0, z}}, view, transform);
        const b = project({{x: maxLateral, y: 0, z}}, view, transform);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        if (i > 0) {{
          const labelPoint = project({{x: maxLateral * 0.92, y: 0, z}}, view, transform);
          ctx.fillText(`${{Math.round(z)}}`, labelPoint.x + 6, labelPoint.y - 4);
        }}
      }}
    }}

    function render(sample) {{
      const view = getView();
      const transform = computeTransform(view);
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      drawGrid(view, transform);

      ctx.strokeStyle = '#58a7e4';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      let started = false;
      for (const s of samples) {{
        if (s.t > sample.t) break;
        const p = project(s, view, transform);
        if (!started) {{
          ctx.moveTo(p.x, p.y);
          started = true;
        }} else {{
          ctx.lineTo(p.x, p.y);
        }}
      }}
      ctx.stroke();

      const shooter = project({{x: 0, y: 0, z: 0}}, view, transform);
      ctx.fillStyle = '#4ade80';
      ctx.beginPath();
      ctx.arc(shooter.x, shooter.y, 8, 0, Math.PI * 2);
      ctx.fill();

      if (target.enabled) {{
        const t = project(target, view, transform);
        ctx.fillStyle = '#ff5d4d';
        ctx.beginPath();
        ctx.arc(t.x, t.y, 9, 0, Math.PI * 2);
        ctx.fill();
      }}

      const p = project(sample, view, transform);
      ctx.fillStyle = '#ffb454';
      ctx.beginPath();
      ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
      ctx.fill();

      bulletTag.style.left = `${{p.x}}px`;
      bulletTag.style.top = `${{p.y}}px`;
      bulletTagLine.textContent = `t ${{sample.t.toFixed(3)}} s | z ${{sample.z.toFixed(1)}} m | y ${{sample.y.toFixed(2)}} m | v ${{sample.s.toFixed(1)}} m/s`;

      timeValue.textContent = `${{sample.t.toFixed(3)}} s`;
      rangeValue.textContent = `${{sample.z.toFixed(1)}} m`;
      heightValue.textContent = `${{sample.y.toFixed(2)}} m`;
      driftValue.textContent = `${{sample.x.toFixed(3)}} m`;
      speedValue.textContent = `${{sample.s.toFixed(1)}} m/s`;
      energyValue.textContent = `${{sample.e.toFixed(0)}} J`;
    }}

    function tick(timestamp) {{
      if (lastFrame === null) lastFrame = timestamp;
      const delta = (timestamp - lastFrame) / 1000;
      lastFrame = timestamp;
      if (playing) {{
        playback += delta * Number(speedSlider.value);
        if (playback > totalDuration) {{
          playback = totalDuration;
          playing = false;
          toggle.textContent = 'Replay';
        }}
      }}
      render(interpolateSample(playback));
      requestAnimationFrame(tick);
    }}

    toggle.addEventListener('click', () => {{
      if (playback >= totalDuration) {{
        playback = 0;
        playing = true;
        toggle.textContent = 'Pause';
        return;
      }}
      playing = !playing;
      toggle.textContent = playing ? 'Pause' : 'Resume';
    }});

    function setView(yawDeg, pitchDeg, zoom) {{
      yawSlider.value = String(yawDeg);
      pitchSlider.value = String(pitchDeg);
      zoomSlider.value = String(zoom);
    }}

    function adjustCamera(deltaYaw, deltaPitch, deltaZoom = 0) {{
      yawSlider.value = String(Math.max(-180, Math.min(180, Number(yawSlider.value) + deltaYaw)));
      pitchSlider.value = String(Math.max(-80, Math.min(80, Number(pitchSlider.value) + deltaPitch)));
      zoomSlider.value = String(Math.max(0.6, Math.min(2.2, Number(zoomSlider.value) + deltaZoom)));
    }}

    viewIso.addEventListener('click', () => setView(49, 32, 1.2));
    viewSide.addEventListener('click', () => setView(90, 8, 1.35));
    viewTop.addEventListener('click', () => setView(90, 75, 1.05));
    viewFront.addEventListener('click', () => setView(0, 10, 1.15));
    camUp.addEventListener('click', () => adjustCamera(0, 5));
    camDown.addEventListener('click', () => adjustCamera(0, -5));
    camLeft.addEventListener('click', () => adjustCamera(-6, 0));
    camRight.addEventListener('click', () => adjustCamera(6, 0));
    zoomIn.addEventListener('click', () => adjustCamera(0, 0, 0.1));
    zoomOut.addEventListener('click', () => adjustCamera(0, 0, -0.1));
    window.addEventListener('keydown', (event) => {{
      const key = event.key.toLowerCase();
      if (['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright', '+', '=', '-', '_'].includes(key)) {{
        event.preventDefault();
      }}
      if (key === 'w' || key === 'arrowup') adjustCamera(0, 5);
      if (key === 's' || key === 'arrowdown') adjustCamera(0, -5);
      if (key === 'a' || key === 'arrowleft') adjustCamera(-6, 0);
      if (key === 'd' || key === 'arrowright') adjustCamera(6, 0);
      if (key === '+' || key === '=') adjustCamera(0, 0, 0.1);
      if (key === '-' || key === '_') adjustCamera(0, 0, -0.1);
    }});

    render(samples[0]);
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""


def export_3d_animation_html(
    result: SimulationResult,
    output_path: str | Path,
    target_forward_m: Optional[float] = None,
    target_height_m: Optional[float] = None,
    target_lateral_m: Optional[float] = None,
    title: str = "3D Trajectory Animation",
) -> Path:
    path = Path(output_path)
    html = build_3d_animation_html(
        result,
        target_forward_m=target_forward_m,
        target_height_m=target_height_m,
        target_lateral_m=target_lateral_m,
        title=title,
    )
    path.write_text(html, encoding="utf-8")
    return path


def open_local_file(path: str | Path) -> None:
    resolved = Path(path).resolve()
    try:
        if hasattr(os, "startfile"):
            os.startfile(str(resolved))
        else:
            webbrowser.open(resolved.as_uri())
    except Exception:
        pass


def get_float(prompt: str, default: Optional[float] = None, min_value: Optional[float] = None) -> float:
    while True:
        text = input(prompt).strip()
        if text == "" and default is not None:
            value = default
        else:
            try:
                value = float(text)
            except ValueError:
                print("Invalid number. Try again.")
                continue

        if min_value is not None and value < min_value:
            print(f"Value must be at least {min_value}.")
            continue
        return value


def get_optional_float(prompt: str, min_value: Optional[float] = None) -> Optional[float]:
    while True:
        text = input(prompt).strip()
        if text == "":
            return None
        try:
            value = float(text)
        except ValueError:
            print("Invalid number. Leave blank to skip.")
            continue

        if min_value is not None and value < min_value:
            print(f"Value must be at least {min_value}.")
            continue
        return value


def get_choice(prompt: str, choices: tuple[str, ...], default: str) -> str:
    allowed = {choice.upper(): choice.upper() for choice in choices}
    while True:
        text = input(prompt).strip().upper()
        if not text:
            return default.upper()
        if text in allowed:
            return allowed[text]
        print(f"Choose one of: {', '.join(choices)}")


def get_text(prompt: str, default: Optional[str] = None) -> str:
    text = input(prompt).strip()
    if text == "" and default is not None:
        return default
    return text


def print_saved_calibers(calibers: list[sqlite3.Row]) -> None:
    print("\nSaved calibers")
    print("--------------")
    for row in calibers:
        print(
            f"[{row['id']}] {row['name']} | "
            f"{float(row['diameter_m']) * 1000.0:.2f} mm | "
            f"{float(row['mass_kg']) * 1000.0:.2f} g | "
            f"{row['bc_model']} BC={float(row['ballistic_coefficient']):.4f}"
        )


def print_saved_rifle_profiles(profiles: list[sqlite3.Row]) -> None:
    print("\nSaved rifle profiles")
    print("--------------------")
    for row in profiles:
        zero_display = f"{float(row['zero_range_m']):.1f} m" if row["zero_range_m"] is not None else "None"
        print(
            f"[{row['id']}] {row['name']} | "
            f"sight {float(row['sight_height_m']):.3f} m | "
            f"zero {zero_display} | "
            f"twist {row['twist_direction']}"
        )


def print_saved_optic_profiles(profiles: list[sqlite3.Row]) -> None:
    print("\nSaved optic profiles")
    print("--------------------")
    for row in profiles:
        print(f"[{row['id']}] {row['name']} | {row['click_value']} {row['click_unit']} per click")


def prompt_projectile() -> tuple[Projectile, Optional[str]]:
    print("\n--- Projectile Data ---")
    projectile = Projectile(
        mass_kg=get_float("Mass (kg): ", min_value=1e-9),
        diameter_m=get_float("Diameter (m): ", min_value=1e-9),
        ballistic_coefficient=get_float("BC value: ", min_value=1e-9),
        bc_model=get_choice("BC model (G1/G7) [G7]: ", ("G1", "G7"), "G7"),
        magnus_coefficient=get_float("Magnus coefficient [0.0]: ", default=0.0, min_value=0.0),
        spin_decay_rate=get_float("Spin decay rate (1/s) [0.015]: ", default=0.015, min_value=0.0),
        twist_rate_inches=get_float("Twist rate (in/turn) [8.0]: ", default=8.0, min_value=1e-9),
        bullet_length_m=get_float("Bullet length (m): ", min_value=1e-9),
    )
    caliber_name: Optional[str] = None
    save_choice = input("Save this caliber for reuse? (yes/no) [yes]: ").strip().lower()
    if save_choice in {"", "yes"}:
        caliber_name = get_text("Caliber name: ")
        if caliber_name:
            save_caliber(caliber_name, projectile)
            print(f"Saved caliber '{caliber_name}' to {DATABASE_PATH.resolve()}")
    return projectile, caliber_name


def print_popular_calibers() -> None:
    print("\nPopular caliber presets")
    print("-----------------------")
    running_index = 1
    for category in PRESET_CATEGORY_ORDER:
        category_presets = [preset for preset in POPULAR_CALIBER_PRESETS if preset.category == category]
        if not category_presets:
            continue
        print(f"\n{category}")
        print("~" * len(category))
        for preset in category_presets:
            print(f"[{running_index}] {preset.name} | {preset.description}")
            running_index += 1


def choose_barrel_preset(caliber_name: Optional[str]) -> Optional[float]:
    if caliber_name is None:
        return None

    normalized_name = PRESET_NAME_ALIASES.get(caliber_name, caliber_name)
    preset = PRESET_BY_NAME.get(normalized_name)
    if preset is None or not preset.barrel_presets:
        return None

    print(f"\nSuggested barrel lengths for {preset.name}")
    print("--------------------------------")
    for index, barrel in enumerate(preset.barrel_presets, start=1):
        print(f"[{index}] {barrel.label} -> {barrel.muzzle_velocity_mps:.0f} m/s")

    choice = input("Choose barrel preset or press Enter for custom velocity: ").strip()
    if not choice:
        return None
    try:
        barrel_index = int(choice)
    except ValueError:
        print("Invalid barrel preset. Falling back to manual velocity entry.")
        return None
    if 1 <= barrel_index <= len(preset.barrel_presets):
        selected = preset.barrel_presets[barrel_index - 1]
        print(f"Using suggested muzzle velocity from {selected.label}: {selected.muzzle_velocity_mps:.0f} m/s")
        return selected.muzzle_velocity_mps

    print("Barrel preset not found. Falling back to manual velocity entry.")
    return None


def choose_projectile() -> tuple[Projectile, Optional[str]]:
    print_popular_calibers()
    calibers = fetch_saved_calibers()
    preset_choice = input("\nLoad popular preset? Enter number or press Enter to skip: ").strip()
    if preset_choice:
        try:
            preset_index = int(preset_choice)
        except ValueError:
            print("Invalid preset number. Continuing to saved or manual selection.")
        else:
            if 1 <= preset_index <= len(POPULAR_CALIBER_PRESETS):
                preset = POPULAR_CALIBER_PRESETS[preset_index - 1]
                print(f"Loaded preset '{preset.name}' ({preset.description}).")
                return preset.projectile, preset.name
            print("Preset number not found. Continuing to saved or manual selection.")

    if calibers:
        print_saved_calibers(calibers)
        choice = input("\nLoad saved caliber? Enter id or press Enter for new: ").strip()
        if choice:
            try:
                caliber_id = int(choice)
            except ValueError:
                print("Invalid id. Switching to new caliber entry.")
            else:
                for row in calibers:
                    if int(row["id"]) == caliber_id:
                        print(f"Loaded caliber '{row['name']}'.")
                        return projectile_from_row(row), str(row["name"])
                print("Caliber id not found. Switching to new caliber entry.")
    return prompt_projectile()


def prompt_environment() -> Environment:
    print("\n--- Environment ---")
    temperature_c = get_float("Sea-level temperature (C) [15.0]: ", default=15.0)
    pressure_hpa = get_float("Sea-level pressure (hPa) [1013.25]: ", default=1013.25, min_value=1.0)
    return Environment(
        latitude_deg=get_float("Latitude (deg) [32.0]: ", default=32.0),
        wind_x=get_float("Wind X/right (+) m/s [0.0]: ", default=0.0),
        wind_y=get_float("Wind Y/up (+) m/s [0.0]: ", default=0.0),
        wind_z=get_float("Wind Z/forward (+) m/s [0.0]: ", default=0.0),
        gravity=get_float("Gravity (m/s^2) [9.80665]: ", default=9.80665, min_value=1e-9),
        sea_level_pressure_pa=pressure_hpa * 100.0,
        sea_level_temperature_k=temperature_c + 273.15,
        relative_humidity=get_float("Relative humidity 0..1 [0.0]: ", default=0.0, min_value=0.0),
    )


def prompt_rifle_profile() -> RifleProfile:
    print("\n--- Rifle & Optic Profile ---")
    profile = RifleProfile(
        name=get_text("Profile name [Default Rifle]: ", default="Default Rifle"),
        sight_height_m=get_float("Sight height over bore (m) [0.05]: ", default=0.05, min_value=0.0),
        zero_range_m=get_optional_float("Zero range (m) [100]: ", min_value=1e-9),
        twist_direction=get_choice("Twist direction (R/L) [R]: ", ("R", "L"), "R"),
        scope_click_unit=get_choice("Scope click unit (MOA/MIL) [MOA]: ", ("MOA", "MIL"), "MOA"),
        scope_click_value=get_float("Scope click value [0.25]: ", default=0.25, min_value=1e-9),
    )
    if profile.zero_range_m is None:
        profile.zero_range_m = 100.0
    return profile


def choose_optic_profile() -> OpticProfile:
    profiles = fetch_saved_optic_profiles()
    if profiles:
        print_saved_optic_profiles(profiles)
        choice = input("\nLoad saved optic profile? Enter id or press Enter for new: ").strip()
        if choice:
            try:
                profile_id = int(choice)
            except ValueError:
                print("Invalid optic id. Switching to new optic entry.")
            else:
                for row in profiles:
                    if int(row["id"]) == profile_id:
                        print(f"Loaded optic profile '{row['name']}'.")
                        return optic_profile_from_row(row)
                print("Optic id not found. Switching to new optic entry.")

    print("\n--- Optic Profile ---")
    profile = OpticProfile(
        name=get_text("Optic profile name [Default Optic]: ", default="Default Optic"),
        click_unit=get_choice("Scope click unit (MOA/MIL) [MOA]: ", ("MOA", "MIL"), "MOA"),
        click_value=get_float("Scope click value [0.25]: ", default=0.25, min_value=1e-9),
    )
    save_choice = input("Save this optic profile? (yes/no) [yes]: ").strip().lower()
    if save_choice in {"", "yes"}:
        save_optic_profile(profile)
        print(f"Saved optic profile '{profile.name}' to {DATABASE_PATH.resolve()}")
    return profile


def choose_rifle_profile() -> RifleProfile:
    profiles = fetch_saved_rifle_profiles()
    if profiles:
        print_saved_rifle_profiles(profiles)
        choice = input("\nLoad saved rifle profile? Enter id or press Enter for new: ").strip()
        if choice:
            try:
                profile_id = int(choice)
            except ValueError:
                print("Invalid rifle id. Switching to new rifle entry.")
            else:
                for row in profiles:
                    if int(row["id"]) == profile_id:
                        print(f"Loaded rifle profile '{row['name']}'.")
                        return rifle_profile_from_row(row)
                print("Rifle id not found. Switching to new rifle entry.")

    profile = prompt_rifle_profile()
    save_choice = input("Save this rifle profile? (yes/no) [yes]: ").strip().lower()
    if save_choice in {"", "yes"}:
        save_rifle_profile(profile)
        print(f"Saved rifle profile '{profile.name}' to {DATABASE_PATH.resolve()}")
    return profile


def prompt_launch(default_muzzle_velocity_mps: Optional[float] = None) -> Launch:
    print("\n--- Launch Conditions ---")
    velocity_prompt = "Muzzle velocity (m/s): "
    if default_muzzle_velocity_mps is not None:
        velocity_prompt = f"Muzzle velocity (m/s) [{default_muzzle_velocity_mps:.0f}]: "
    return Launch(
        muzzle_velocity_mps=get_float(velocity_prompt, default=default_muzzle_velocity_mps, min_value=1e-9),
        elevation_deg=get_float("Elevation (deg) [0.0]: ", default=0.0),
        azimuth_deg=get_float("Azimuth (deg) [0.0]: ", default=0.0),
        muzzle_height_m=get_float("Muzzle height (m) [1.5]: ", default=1.5, min_value=0.0),
        sample_distance_m=get_optional_float("Sample distance forward (m) [skip]: ", min_value=1e-9),
    )


def prompt_solver() -> SolverConfig:
    print("\n--- Solver Settings ---")
    return SolverConfig(
        dt=get_float("Time step dt (s) [0.001]: ", default=0.001, min_value=1e-9),
        max_time_s=get_float("Maximum time (s) [10.0]: ", default=10.0, min_value=1e-9),
        record_trajectory=get_choice("Record trajectory (yes/no) [yes]: ", ("YES", "NO"), "YES") == "YES",
        output_interval_s=get_float("Trajectory output interval (s) [0.01]: ", default=0.01, min_value=1e-9),
        verbose=True,
    )


def compute_line_of_sight_adjustment(
    sample: TrajectorySample,
    zero_sample: Optional[TrajectorySample],
    rifle: RifleProfile,
) -> tuple[float, float, float, float]:
    zero_height = zero_sample.y_m if zero_sample is not None else 0.0
    zero_range = zero_sample.z_m if zero_sample is not None else 0.0

    sight_line_height = rifle.sight_height_m + zero_height
    if zero_range > EPSILON:
        slope = (zero_height + rifle.sight_height_m) / zero_range
        sight_line_height = rifle.sight_height_m + slope * sample.z_m

    offset_m = sight_line_height - sample.y_m
    elevation_moa = (offset_m / sample.z_m) * MOA_PER_RAD if sample.z_m > EPSILON else 0.0
    elevation_mils = (offset_m / sample.z_m) * MILS_PER_RAD if sample.z_m > EPSILON else 0.0
    return offset_m, elevation_moa, elevation_mils, sight_line_height


def compute_scope_clicks(adjustment_value: float, optic: OpticProfile) -> float:
    if optic.click_value <= EPSILON:
        return 0.0
    return adjustment_value / optic.click_value


def print_range_card(result: SimulationResult, rifle: RifleProfile, optic: OpticProfile) -> None:
    if len(result.trajectory) < 2:
        print("\nRange card unavailable: trajectory recording is required.")
        return

    step_m = get_float("\nRange card increment (m) [50]: ", default=50.0, min_value=1e-9)
    max_range = result.range_m
    if max_range <= step_m:
        print("Range card unavailable: simulated distance is shorter than the requested increment.")
        return

    zero_sample = None
    if rifle.zero_range_m is not None:
        zero_sample = interpolate_trajectory_sample_by_range(result.trajectory, rifle.zero_range_m)

    print("\nDOPE / Range Card")
    print("-----------------")
    print(
        f"Rifle: {rifle.name} | Sight height: {rifle.sight_height_m:.3f} m | "
        f"Zero: {rifle.zero_range_m:.1f} m | Optic: {optic.name} | Clicks: {optic.click_value:.3f} {optic.click_unit.upper()}"
    )
    print(f"Range(m) | Drop LOS(m) | Elev(MOA) | Elev(mil) | Wind(MOA) | Elev Clicks | Wind Clicks | Speed(m/s) | TOF(s) | Energy(J)")

    targets = build_range_targets(max_range, step_m)
    for target in targets:
        sample = interpolate_trajectory_sample_by_range(result.trajectory, target)
        if sample is None:
            continue
        offset_m, elevation_moa, elevation_mils, _ = compute_line_of_sight_adjustment(sample, zero_sample, rifle)
        click_basis = elevation_moa if optic.click_unit.upper() == "MOA" else elevation_mils
        clicks = compute_scope_clicks(click_basis, optic)
        wind_click_basis = ((sample.x_m / sample.z_m) * MOA_PER_RAD) if optic.click_unit.upper() == "MOA" and sample.z_m > EPSILON else (((sample.x_m / sample.z_m) * MILS_PER_RAD) if sample.z_m > EPSILON else 0.0)
        wind_clicks = compute_scope_clicks(wind_click_basis, optic)
        elevation_direction = format_adjustment_direction(clicks, "UP", "DOWN")
        wind_direction = format_adjustment_direction(wind_clicks, "RIGHT", "LEFT")
        print(
            f"{sample.z_m:8.1f} | "
            f"{offset_m:10.3f} | "
            f"{elevation_moa:9.3f} | "
            f"{elevation_mils:9.3f} | "
            f"{((sample.x_m / sample.z_m) * MOA_PER_RAD) if sample.z_m > EPSILON else 0.0:9.3f} | "
            f"{elevation_direction} {abs(clicks):5.1f} | "
            f"{wind_direction} {abs(wind_clicks):5.1f} | "
            f"{sample.speed_mps:10.1f} | "
            f"{sample.time_s:6.3f} | "
            f"{sample.energy_j:9.1f}"
        )


def maybe_export_trajectory(result: SimulationResult) -> None:
    if not result.trajectory:
        return
    animate = input("\nPlay trajectory animation in terminal? (yes/no) [yes]: ").strip().lower()
    if animate in {"", "yes"}:
        render_terminal_trajectory_animation(result)

    answer = input("\nExport trajectory to CSV? (yes/no) [no]: ").strip().lower()
    if answer == "yes":
        raw_path = input("CSV path [trajectory.csv]: ").strip() or "trajectory.csv"
        csv_path = result.write_trajectory_csv(raw_path)
        print(f"Trajectory written to {csv_path.resolve()}")


def maybe_export_3d_animation(
    result: SimulationResult,
    target_forward_m: Optional[float] = None,
    target_height_m: Optional[float] = None,
    target_lateral_m: Optional[float] = None,
    title: str = "3D Trajectory Animation",
) -> None:
    choice = input("Export full-time 3D trajectory animation HTML? (yes/no) [no]: ").strip().lower()
    if choice != "yes":
        return
    raw_path = input("3D animation path [trajectory_3d.html]: ").strip() or "trajectory_3d.html"
    html_path = export_3d_animation_html(
        result,
        raw_path,
        target_forward_m=target_forward_m,
        target_height_m=target_height_m,
        target_lateral_m=target_lateral_m,
        title=title,
    )
    open_local_file(html_path)
    print(f"3D animation written to {html_path.resolve()}")


def run_compare_mode(env: Environment, rifle: RifleProfile, optic: OpticProfile) -> None:
    print("\n=== Compare Mode ===")
    projectile_a, caliber_a = choose_projectile()
    velocity_a = choose_barrel_preset(caliber_a)
    launch_a = prompt_launch(velocity_a)
    config_a = prompt_solver()
    result_a = simulate(projectile_a, env, launch_a, config_a)

    print("\n--- Comparison Setup B ---")
    projectile_b, caliber_b = choose_projectile()
    velocity_b = choose_barrel_preset(caliber_b)
    launch_b = prompt_launch(velocity_b)
    config_b = prompt_solver()
    result_b = simulate(projectile_b, env, launch_b, config_b)

    print("\nCompare Summary")
    print("---------------")
    print(f"A: {caliber_a or 'Custom A'} | Range {result_a.range_m:.1f} m | Speed {result_a.speed_mps:.1f} m/s | Energy {result_a.energy_j:.1f} J")
    print(f"B: {caliber_b or 'Custom B'} | Range {result_b.range_m:.1f} m | Speed {result_b.speed_mps:.1f} m/s | Energy {result_b.energy_j:.1f} J")
    print(f"Range delta (A-B): {result_a.range_m - result_b.range_m:.1f} m")
    print(f"Energy delta (A-B): {result_a.energy_j - result_b.energy_j:.1f} J")

    choice = input("\nGenerate DOPE card for which setup? (A/B/none) [none]: ").strip().upper()
    if choice == "A":
        print_range_card(result_a, rifle, optic)
    elif choice == "B":
        print_range_card(result_b, rifle, optic)


def run_monte_carlo_mode(
    projectile: Projectile,
    env: Environment,
    launch: Launch,
    config: SolverConfig,
) -> None:
    if np is None:
        print("\nMonte Carlo mode requires numpy and is unavailable in this environment.")
        return

    print("\n=== Monte Carlo Mode ===")
    sample_count = int(get_float("Number of trials [100]: ", default=100.0, min_value=1.0))
    velocity_sd = get_float("Muzzle velocity SD (m/s) [5.0]: ", default=5.0, min_value=0.0)
    elevation_sd = get_float("Elevation SD (deg) [0.1]: ", default=0.1, min_value=0.0)
    wind_sd = get_float("Crosswind X SD (m/s) [0.5]: ", default=0.5, min_value=0.0)

    rng = np.random.default_rng()
    ranges = []
    drops = []
    drifts = []
    speeds = []
    impacts = 0

    silent_config = SolverConfig(
        dt=config.dt,
        max_time_s=config.max_time_s,
        record_trajectory=False,
        output_interval_s=config.output_interval_s,
        verbose=False,
    )

    for _ in range(sample_count):
        varied_launch = Launch(
            muzzle_velocity_mps=max(1e-9, launch.muzzle_velocity_mps + float(rng.normal(0.0, velocity_sd))),
            elevation_deg=launch.elevation_deg + float(rng.normal(0.0, elevation_sd)),
            azimuth_deg=launch.azimuth_deg,
            muzzle_height_m=launch.muzzle_height_m,
            sample_distance_m=launch.sample_distance_m,
        )
        varied_env = Environment(
            latitude_deg=env.latitude_deg,
            wind_x=env.wind_x + float(rng.normal(0.0, wind_sd)),
            wind_y=env.wind_y,
            wind_z=env.wind_z,
            gravity=env.gravity,
            sea_level_pressure_pa=env.sea_level_pressure_pa,
            sea_level_temperature_k=env.sea_level_temperature_k,
            relative_humidity=env.relative_humidity,
            temperature_lapse_k_per_m=env.temperature_lapse_k_per_m,
        )
        trial = simulate(projectile, varied_env, varied_launch, silent_config)
        ranges.append(trial.range_m)
        drops.append(trial.drop_from_muzzle_m)
        drifts.append(trial.lateral_m)
        speeds.append(trial.speed_mps)
        if trial.terminated_by_ground_impact:
            impacts += 1

    ranges_arr = np.array(ranges)
    drops_arr = np.array(drops)
    drifts_arr = np.array(drifts)
    speeds_arr = np.array(speeds)

    print("\nMonte Carlo Summary")
    print("-------------------")
    print(f"Trials: {sample_count} | Ground impacts: {impacts}/{sample_count}")
    print(f"Range mean/std: {ranges_arr.mean():.2f} / {ranges_arr.std():.2f} m")
    print(f"Drop mean/std: {drops_arr.mean():.2f} / {drops_arr.std():.2f} m")
    print(f"Drift mean/std: {drifts_arr.mean():.2f} / {drifts_arr.std():.2f} m")
    print(f"Impact speed mean/std: {speeds_arr.mean():.2f} / {speeds_arr.std():.2f} m/s")


def run_reverse_calculation_mode(
    projectile: Projectile,
    env: Environment,
    rifle: RifleProfile,
    optic: OpticProfile,
    launch: Launch,
    config: SolverConfig,
) -> None:
    print("\n=== Reverse Calculation ===")
    target_forward_m = get_float("Target forward distance (m): ", min_value=1e-9)
    target_height_m = get_float("Target height relative to shooter ground plane (m) [0.0]: ", default=0.0)
    target_lateral_m = get_float("Target lateral offset right(+) / left(-) (m) [0.0]: ", default=0.0)
    solution = reverse_solve_target(
        projectile,
        env,
        launch,
        rifle,
        optic,
        config,
        target_forward_m,
        target_height_m,
        target_lateral_m,
    )
    print_reverse_solution(solution, optic)
    if not solution.solved:
        return

    animate_choice = input("Show solved target in trajectory animation? (yes/no) [yes]: ").strip().lower()
    if animate_choice not in {"", "yes"}:
        return

    solved_launch = Launch(
        muzzle_velocity_mps=launch.muzzle_velocity_mps,
        elevation_deg=solution.solved_elevation_deg,
        azimuth_deg=solution.solved_azimuth_deg,
        muzzle_height_m=launch.muzzle_height_m,
        sample_distance_m=target_forward_m,
    )
    solved_config = SolverConfig(
        dt=config.dt,
        max_time_s=config.max_time_s,
        record_trajectory=True,
        output_interval_s=config.output_interval_s,
        verbose=False,
    )
    solved_result = simulate(projectile, env, solved_launch, solved_config)
    render_terminal_trajectory_animation(
        solved_result,
        target_forward_m=target_forward_m,
        target_height_m=target_height_m,
    )
    maybe_export_3d_animation(
        solved_result,
        target_forward_m=target_forward_m,
        target_height_m=target_height_m,
        target_lateral_m=target_lateral_m,
        title="Reverse Solution 3D Trajectory",
    )


def feature_menu(
    projectile: Projectile,
    caliber_name: Optional[str],
    env: Environment,
    rifle: RifleProfile,
    optic: OpticProfile,
    launch: Launch,
    config: SolverConfig,
    result: SimulationResult,
) -> None:
    while True:
        print("\nFeature Categories")
        print("------------------")
        print("[1] Shooter Tools: DOPE / Range Card")
        print("[2] Shooter Tools: Reverse Calculation")
        print("[3] Batch Analysis: Full Simulation")
        print("[4] Visuals & Export")
        print("[5] Comparison")
        print("[6] Uncertainty / Monte Carlo")
        print("[7] Back to main run loop")
        choice = input("Choose feature category: ").strip()

        if choice == "1":
            print_range_card(result, rifle, optic)
        elif choice == "2":
            run_reverse_calculation_mode(projectile, env, rifle, optic, launch, config)
        elif choice == "3":
            maybe_run_full_simulation(projectile, caliber_name, env, launch, result)
        elif choice == "4":
            animate = input("Play trajectory animation? (yes/no) [yes]: ").strip().lower()
            if animate in {"", "yes"}:
                render_terminal_trajectory_animation(result)
            export = input("Export trajectory CSV? (yes/no) [no]: ").strip().lower()
            if export == "yes":
                raw_path = input("CSV path [trajectory.csv]: ").strip() or "trajectory.csv"
                csv_path = result.write_trajectory_csv(raw_path)
                print(f"Trajectory written to {csv_path.resolve()}")
            maybe_export_3d_animation(result, title="3D Trajectory Animation")
        elif choice == "5":
            run_compare_mode(env, rifle, optic)
        elif choice == "6":
            run_monte_carlo_mode(projectile, env, launch, config)
        elif choice == "7":
            break
        else:
            print("Unknown option. Choose 1-7.")


def maybe_run_full_simulation(
    projectile: Projectile,
    caliber_name: Optional[str],
    env: Environment,
    launch: Launch,
    single_result: SimulationResult,
) -> None:
    choice = input("\nRun full simulation? (yes/no) [no]: ").strip().lower()
    if choice != "yes":
        return

    if not single_result.trajectory:
        print("Full simulation requires trajectory recording. Re-run with trajectory recording enabled.")
        return
    if launch.sample_distance_m is None:
        print("Full simulation requires a sample distance. This run did not provide one.")
        return
    if single_result.sample_hit is None:
        print("Single run did not reach the sample distance. Increase max time or reduce the sample distance.")
        return

    increment_m = get_float("Full simulation range increment (m): ", min_value=1e-9)
    default_label = time.strftime("run_%Y%m%d_%H%M%S")
    run_label = get_text(f"Run label [{default_label}]: ", default=default_label)
    full_result = run_full_simulation(projectile, env, launch, single_result, increment_m, run_label)
    save_full_simulation_rows(caliber_name, projectile, env, launch, full_result)
    print_full_simulation_report(full_result)
    print(f"Stored full simulation in SQLite: {full_result.sqlite_path.resolve()}")


def main() -> None:
    initialize_database()
    print(PROGRAM_NAME)
    print("High-fidelity point-mass solver with spin effects, atmosphere, and event interpolation.")

    while True:
        print("\nMain Menu")
        print("---------")
        print("[1] Single Run / Fun Shot")
        print("[2] Compare Two Setups")
        print("[3] Exit")
        main_choice = input("Choose session type: ").strip()
        if main_choice == "3":
            break
        if main_choice not in {"1", "2"}:
            print("Unknown session type. Choose 1-3.")
            continue

        env = prompt_environment()
        optic = choose_optic_profile()
        rifle = choose_rifle_profile()
        rifle.scope_click_unit = optic.click_unit.upper()
        rifle.scope_click_value = optic.click_value
        rifle.validate()
        if main_choice == "2":
            run_compare_mode(env, rifle, optic)
        else:
            projectile, caliber_name = choose_projectile()
            suggested_velocity = choose_barrel_preset(caliber_name)
            launch = prompt_launch(suggested_velocity)
            warnings = build_input_warnings(projectile, launch)
            if warnings:
                print("\nInput warnings")
                print("--------------")
                for warning in warnings:
                    print(f"- {warning}")
            config = prompt_solver()
            result = simulate(projectile, env, launch, config)
            feature_menu(projectile, caliber_name, env, rifle, optic, launch, config, result)

        again = input("\nRun another shot? (yes/no) [no]: ").strip().lower()
        if again != "yes":
            break

    print("Simulation session complete.")


if __name__ == "__main__":
    main()
