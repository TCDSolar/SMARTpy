"""
====
Demo
====

Stuff in here is treated as rst so this works

* a
* b

or

#. x
#. y
#. z

Sections
============

Subsections
-----------------

Subsubsections
^^^^^^^^^^^^^^^^^^^^

Paragraphs
''''''''''

code that is to be run goes outside these blocks
"""

import numpy as np
from matplotlib import pyplot as plt

#####################################################
#
# More text
# ---------
#
# .. note::
#     More text something about imports
#

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
