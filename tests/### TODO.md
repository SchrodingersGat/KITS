### TODO


- [ ] Thermal transfer axially for resistive elements
    * Does not seem to transfer heat from one side of a resistive element to the other effectively.
    * Thermal losses I2R through these elements don't make it into the board Cu?

- [ ] Board meshing options:
    * ignore traces
    * include only critical traces (opt-in)
    * adaptive meshing based on thermal gradients
    * define "areas of interest" for finer meshing
    * define "areas of disinterest" for coarser meshing

- [ ] Resistance values
    * determine resistance values from schematic when creating resistive elements