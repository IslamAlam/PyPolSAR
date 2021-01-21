# type: ignore

"""
Utilities for dealing with RAT files from DLR-HR

[1](https://www.dlr.de/hr/Portaldata/32/Resources/images/institut/sar-technologie/f-sar/F-SAR_DIMS-products.pdf)
Notes
[2](https://www.dlr.de/hr/en/desktopdefault.aspx/tabid-2326/3776_read-48006)
-----
Appendix 2: Rat Format

The following table indicates the binary structure of RAT (version 2) files on disk. All floating point and
complex data follow IEEE standards and are stored with little endian byte ordering.

| Group            | Tagname      | Length [byte] | Type                    | Example                   | Description                                                                                                     |
|------------------|--------------|---------------|-------------------------|---------------------------|-----------------------------------------------------------------------------------------------------------------|
| RAT (100 bytes)  |   MagicLong  |       4       | 1 x long                | 844382546                 | Magic number for recognizing RAT                                                                                |
|                  |    Version   |       4       | 1 x float               | 2                         | RAT Version number                                                                                              |
|                  |     NDIM     |       4       | 1 x long                | 2                         | Number of  dimensions of data matrix                                                                            |
|                  |   NCHANNEL   |       4       | 1 x long                | 1                         | Number of channels                                                                                              |
|                  |      DIM     |       32      | 8 x long                | 1000,2000,0,0,0,0,0,0     | Number of samples per dimension                                                                                 |
|                  |      VAR     |       4       | 1 x long                | 4                         | IDL variable type (1- byte,2-integer, 4   –float, 6 –complex)                                                   |
|                  |      SUB     |       8       | 2 x long                | 5, 8                      | Subsampling factors                                                                                             |
|                  |    RATTYPE   |       4       | 1x long                 | 100                       | RAT type                                                                                                        |
|                  |   RESERVED   |       36      | 9 x long                | 9x0                       | <empty>                                                                                                         |
| INFO (100 bytes) |     INFO     |      100      | string                  |                           | Description of file content                                                                                     |
| GEO (100 bytes)  |  PROJECTION  |       2       | 1 x int                 | 1                         | Projection Type (0=Lat/Long, 1 = UTM,   2= Gauss-Krüger)                                                        |
|                  |    PS_EAST   |       8       | 1 x double              | 1                         | Sampling in Easting ([deg] or [m])                                                                              |
|                  |   PS_NORTH   |       8       | 1 x double              | 1                         | Sampling in Northing ([deg] or [m])                                                                             |
|                  |   MIN_EAST   |       8       | 1 x double              | 436041                    | Minimum easting (lower left corner)                                                                             |
|                  |   MIN_NORTH  |       8       | 1 x double              | 5921365                   | Minimum northing (lower left corner)                                                                            |
|                  |              |               |                         |                           |                                                                                                                 |
|                  |     ZONE     |       2       |                         | 32                        | Projection zone                                                                                                 |
|                  |  HEMISPHERE  |       2       |                         | 1                         | Hemisphere (1 – north, 2-  south)                                                                               |
|                  |   LONG0SCL   |       8       | 1 x double              | 0.99996                   | Scaling factor at central meridian                                                                              |
|                  | MAX_AXIS_ELL |       8       | 1 x double              | 6378137                   | Ellipsoid major axis                                                                                            |
|                  | MIN_AXIS_ELL |       8       | 1 x double              | 6356752.3                 | Ellipsoid minor axis                                                                                            |
|                  |  DATUM_SHIFT |      100      | 7 x7 x double + 64 byte |                           | Datum Shift Parameters (3x translation,   3x Rotation, 1x Scaling) in case other than WGS-84 ellipsoid is used. |
|                  |   RESERVED   |       18      | 18 x byte               |                           | <empty>                                                                                                         |
| STAT (100bytes)  |     STAT     |      100      | 25 x long               | 0                         | Reserved for statistical values of data   matrix.                                                               |
| DATE (100bytes)  |  START_TIME  |       19      | string                  | 2012-11-14      T18:20:06 | Start time of data acquisiton                                                                                   |
|                  |   STOP_TIME  |       19      | string                  | 2012-11-14      T18:21:29 | Stop time of data acquisiton                                                                                    |
|                  |   RESERVED   |       62      | 62 x byte               |                           | <empty>                                                                                                         |
| RESERVED1        |   RESERVED   |      100      | 25 x long               |                           | <empty>                                                                                                         |
| RESERVED2        |   RESERVED   |      100      | 25 x long               |                           | <empty>                                                                                                         |
| RESERVED3        |   RESERVED   |      100      | 25 x long               |                           | <empty>                                                                                                         |
| RESERVED4        |   RESERVED   |      100      | 25 x long               |                           | <empty>                                                                                                         |

[3. MD Tabel](https://www.tablesgenerator.com/markdown_tables)
"""

from .geo_rat import *
# RAT file read and write utilities
from .ste_io import loadrat, readrat, saverat

__all__ = ["loadrat", "saverat", "readrat"]
