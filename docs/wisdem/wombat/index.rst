WOMBAT
=====

Overview
--------

The land-based and offshore Windfarm Operations and Maintenance cost-Beneft Anaylysis Tool (WOMBAT) is a low fidelity,
process-based, discrete event simulation model to understand the cost, energy production, and downtime implications
of technological and maintenance-based changes to the operations and maintenance (O&M) phase of the wind life cycle.

WOMBAT allows for the modeling of arbitrarily simple or complex wind turbines, substations, cables, and hydrogen
electrolyzers through fixed-interval maintenance and Weibull-distributed failure events. Paired with the ability
to generically model many servicing equipment, WOMBAT enables users to model a plethora of wind O&M scenarios.

Documentation
-------------

WOMBAT maintains its own Github `repository <https://github.com/WISDEM/WOMBAT>`_ and
`documentation <https://wisdem.github.io/WOMBAT/>`_. WISDEM uses the default scenarios included in
the WOMBAT package, so annual updates for inflation and improved assumptions are automatically
applied to the the model.


Usage
_____

WOMBAT can be easily used as a standalone module through WISDEM or by installing from its own
`repository <https://github.com/WISDEM/WOMBAT>`_ as a separate project. For examples and
documentation on WOMBAT usage, please read the
`How To Use WOMBAT guide <https://wisdem.github.io/WOMBAT/examples/how_to.html>`_ and the linked
`API reference <https://wisdem.github.io/WOMBAT/API/index.html>`_ guides. When using WOMBAT through
WISDEM, use the following import:

>>> import wisdem.wombat

instead of 

>>> import wombat
