"""The fixed words this project uses for traffic levels and vehicle kinds.

Every module imports the names from here instead of typing the strings again.
A typo then fails at import time rather than silently creating a new category
halfway through a run.
"""

from __future__ import annotations

from enum import StrEnum


class CongestionLevel(StrEnum):
    """How busy the road is in one clip.

    These are the three words the dataset itself uses in ``ImageMaster``:

    - ``LIGHT``  - traffic flows freely
    - ``MEDIUM`` - traffic moves, but slower than free flow
    - ``HEAVY``  - traffic is stopped or crawling
    """

    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


#: Order used for report tables, confusion matrices and colour scales.
#: Least busy first, so a chart reads left to right as "getting worse".
CONGESTION_ORDER: tuple[CongestionLevel, ...] = (
    CongestionLevel.LIGHT,
    CongestionLevel.MEDIUM,
    CongestionLevel.HEAVY,
)


class VehicleClass(StrEnum):
    """The kinds of vehicle we report separately.

    These match the COCO names the detector was trained on, so the mapping from
    a detector class id to one of these lives in the config file and not here.
    """

    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"


#: Order used whenever vehicle classes are listed, most common first.
VEHICLE_ORDER: tuple[VehicleClass, ...] = (
    VehicleClass.CAR,
    VehicleClass.TRUCK,
    VehicleClass.BUS,
    VehicleClass.MOTORCYCLE,
)


class Weather(StrEnum):
    """Weather word recorded for each clip in ``info.txt``.

    Kept because accuracy is reported per weather condition: the rain clips are
    the hard ones and they deserve their own number.
    """

    CLEAR = "clear"
    OVERCAST = "overcast"
    RAIN = "rain"
