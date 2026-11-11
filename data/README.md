# Weather data

`weather/Wag/2020.csv` is outdoor weather for Wageningen (the Netherlands), 150 days (day numbers 0-150)
from the weather log of the 2nd Autonomous Greenhouse Challenge, at 5-minute steps, in gl_gym's weather file
format (columns: time [s], global radiation, wind speed, air temperature, sky temperature, a column
gl_gym leaves unnamed, CO2 concentration, day number, RH). The scripts call this location "Wag" and
year 2020.

- Source: Hemming, S., de Zwart, F., Elings, A., Petropoulou, A., Righini, I. (2020). Autonomous
  Greenhouse Challenge, Second Edition (2019), version 2, file `Weather/Weather.csv`.
  4TU.ResearchData, doi:10.4121/uuid:88d22c60-21b3-4ea8-90db-20249a5be2a7.
  Licence: CC0.
- Converted to gl_gym's format by the author; the conversion script is not part of this repository.
- sha256: `048024cd1ca82a3d71f2de57931d4fd5a12ccbbf4a4290db0c43151554863524`

Amsterdam 2008-2011 weather is not copied here: the code reads the files bundled with gl_gym 0.3.2
(GreenLight-Gym2, commit 2d3febb, AGPL-3.0-or-later), which are byte-identical to the ones used for
the paper. Amsterdam 2010 is the evaluation weather; 2008, 2009 and 2011 are the RL training
weather. No other years are used.
