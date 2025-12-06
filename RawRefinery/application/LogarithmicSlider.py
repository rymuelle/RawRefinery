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
        # Use a large integer scale for better precision in the linear range
        self._scale = 1000 
        self.setMinimum(0)
        self.setMaximum(self._scale)
        
        self._min_val = min_val
        self._max_val = max_val

        # Connect the internal integer value change to a custom handler
        self.valueChanged.connect(self._on_internal_value_changed)
        
        self.setNaturalValue(min_val) # Set initial value
    
    def get_natural_value(self):
        # Convert the internal linear integer value to the logarithmic float value
        log_min = math.log10(self._min_val)
        log_max = math.log10(self._max_val)
        
        # Calculate the log value corresponding to the slider's integer position
        log_value = log_min + (self.value() / self._scale) * (log_max - log_min)
        
        # Convert back from log to the natural value
        natural_value = math.pow(10, log_value)
        return float(natural_value)
    
    def _on_internal_value_changed(self, value):
        # Convert the internal linear integer value to the logarithmic float value
        log_min = math.log10(self._min_val)
        log_max = math.log10(self._max_val)
        
        # Calculate the log value corresponding to the slider's integer position
        log_value = log_min + (value / self._scale) * (log_max - log_min)
        
        # Convert back from log to the natural value
        natural_value = math.pow(10, log_value)
        
        # Emit the custom signal with the natural (float) value
        self.naturalValueChanged.emit(natural_value)

    def setNaturalValue(self, value):
        # Set the internal slider position based on a natural value
        log_min = math.log10(self._min_val)
        log_max = math.log10(self._max_val)
        log_value = math.log10(value)
        
        # Calculate the corresponding integer position on the linear scale
        internal_value = int(self._scale * (log_value - log_min) / (log_max - log_min))
        self.setValue(internal_value)
