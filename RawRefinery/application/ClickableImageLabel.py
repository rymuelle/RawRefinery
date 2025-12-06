from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap, QImage, QMouseEvent
from PySide6.QtCore import Qt, QSize, Signal

class ClickableImageLabel(QLabel):
    # Signal to emit when the image is clicked, passing mapped coordinates
    imageClicked = Signal(float, float)

    def __init__(self, parent=None, proportional=False):
        super().__init__(parent)
        self.original_image_size = QSize() # To store the original QImage size
        self.proportional = proportional

    def set_image(self, qimage: QImage):
        self.original_image_size = qimage.size()
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.size(), 
                                      Qt.KeepAspectRatio, 
                                      Qt.SmoothTransformation)
        self.setPixmap(pixmap)
        self.setScaledContents(True)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            # Get click position relative to the QLabel
            label_x = event.pos().x()
            label_y = event.pos().y()

            # Calculate scaling factors
            label_width = self.width()
            label_height = self.height()
            image_width = self.original_image_size.width()
            image_height = self.original_image_size.height()

            if self.proportional:
                if label_width > 0 and label_height > 0 and image_width > 0 and image_height > 0:
                    # Map click coordinates back to original QImage dimensions
                    mapped_x = label_x / label_width
                    mapped_y = label_y / label_height

                    self.imageClicked.emit(mapped_x, mapped_y)
            else:
                if label_width > 0 and label_height > 0 and image_width > 0 and image_height > 0:
                    # Map click coordinates back to original QImage dimensions
                    mapped_x = int(label_x * (image_width / label_width))
                    mapped_y = int(label_y * (image_height / label_height))

                    self.imageClicked.emit(mapped_x, mapped_y)

        super().mousePressEvent(event)
