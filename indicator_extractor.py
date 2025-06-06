import sys
import os

# 1) Compute the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

# 2) Add it to sys.path (if it isn’t already there)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

class ExtractIndicators:
    def ___init___(self):
        """Initialise the data extraction."""
        
    def cleanData(self):
        """Cleans the data so that only useful data is kept."""
        
    def getSubs(self):
        """Explanation."""
        return 
        
    def earlyWin(self):
        """Explanation."""
        