from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QSlider
import math

class LogarithmicSlider(QSlider):
    """
    A QSlider subclass that maps its linear integer range to a logarithmic scale.
    """
    # Define a new signal that emits the natural (logarithmic) float value
    naturalValueChanged = Signal(float)

    def __init__(self, min_val, max_val, parent=None):
        super().__init__(parent)
        self.setOrientation(Qt.Horizontal)
        self._scale = 1000 
        self.setMinimum(0)
        self.setMaximum(self._scale)
        
        self._min_val = min_val
        self._max_val = max_val
        self.valueChanged.connect(self._on_internal_value_changed)
        
        self.setNaturalValue(min_val)
    
    def get_natural_value(self):
        log_min = math.log10(self._min_val)
        log_max = math.log10(self._max_val)

        log_value = log_min + (self.value() / self._scale) * (log_max - log_min)
        natural_value = math.pow(10, log_value)
        
        return float(natural_value)
    
    def _on_internal_value_changed(self, value):
        log_min = math.log10(self._min_val)
        log_max = math.log10(self._max_val)
        
        log_value = log_min + (value / self._scale) * (log_max - log_min)
        natural_value = math.pow(10, log_value)
        
        self.naturalValueChanged.emit(natural_value)

    def setNaturalValue(self, value):
        # Set the internal slider position based on a natural value
        log_min = math.log10(self._min_val)
        log_max = math.log10(self._max_val)
        log_value = math.log10(value)
        
        # Calculate the corresponding integer position on the linear scale
        internal_value = int(self._scale * (log_value - log_min) / (log_max - log_min))
        self.setValue(internal_value)
