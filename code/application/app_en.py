import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, 
    QHBoxLayout, QLineEdit, QDialog, QFormLayout, QMessageBox, QSpacerItem, 
    QSizePolicy, QScrollArea, QTextEdit, QFileDialog, QStackedWidget, QGridLayout
)
from PyQt5.QtGui import QPixmap, QFont, QIcon, QGuiApplication, QFontDatabase
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtMultimedia import QCamera, QCameraImageCapture, QCameraInfo
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
from PIL import Image
import torch
import torch.nn as nn
import numpy as np

# Import the custom ConcatModel and pre-trained model paths from concatmodel.py
from oiodnet import OIODNet

# -------------------
# Registration dialog module
# -------------------
class RegisterDialog(QDialog):
    def __init__(self, font_family):
        super().__init__()
        self.setWindowTitle('Register')
        self.setFixedSize(400, 200)
        self.font_family = font_family
        self.center()

        # Create layout
        self.layout = QVBoxLayout()
        self.form_layout = QFormLayout()

        # Create username and password input boxes
        self.username_input = QLineEdit(self)
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)

        # Set input box styles
        self.username_input.setStyleSheet(f"""
            QLineEdit {{
                border: 2px solid #dcdcdc;
                border-radius: 5px;
                padding: 5px;
                font-size: 16px;
                background-color: #f9f9f9;
                font-family: 'Times New Roman';
                font-weight: bold;
            }}
            QLineEdit:focus {{
                border-color: #a0a0a0;
            }}
        """)
        self.password_input.setStyleSheet(f"""
            QLineEdit {{
                border: 2px solid #dcdcdc;
                border-radius: 5px;
                padding: 5px;
                font-size: 16px;
                background-color: #f9f9f9;
                font-family: 'Times New Roman';
                font-weight: bold;
            }}
            QLineEdit:focus {{
                border-color: #a0a0a0;
            }}
        """)

        # Create labels and add to form layout
        username_label = QLabel("Username:", self)
        username_label.setFont(QFont('Times New Roman', 16, QFont.Bold))
        username_label.setAlignment(Qt.AlignCenter)

        password_label = QLabel("Password:", self)
        password_label.setFont(QFont('Times New Roman', 16, QFont.Bold))
        password_label.setAlignment(Qt.AlignCenter)

        self.form_layout.addRow(username_label, self.username_input)
        self.form_layout.addRow(password_label, self.password_input)

        # Create register button and connect to the function for registering users
        self.register_button = QPushButton('Register', self)
        self.register_button.clicked.connect(self.register_user)

        # Set button style
        self.register_button.setFont(QFont('Times New Roman', 16, QFont.Bold))
        self.register_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 8px 16px;
                min-width: 130px;
                font-family: 'Times New Roman';
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #45a049;
            }}
        """)

        # Add form layout and button layout to the main layout
        self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.layout.addLayout(self.form_layout)
        self.layout.addWidget(self.register_button, alignment=Qt.AlignCenter)
        self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Set main layout
        self.setLayout(self.layout)

    # Center the dialog
    def center(self):
        qr = self.frameGeometry()
        cp = QGuiApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # Function to register users
    def register_user(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        # Check if username and password are empty
        if not username or not password:
            QMessageBox.warning(self, "Error", "Username and password cannot be empty")
            return

        # User data structure
        user_data = {
            'username': username,
            'password': password
        }

        user_file = '/Users/code/Desktop_Application/users.json'
        if os.path.exists(user_file):
            with open(user_file, 'r') as file:
                users = json.load(file)
        else:
            users = {}

        # Check if username already exists
        if username in users:
            QMessageBox.warning(self, "Error", "Username already exists")
            return

        # Save user information to a JSON file
        users[username] = user_data
        with open(user_file, 'w') as file:
            json.dump(users, file, indent=4)

        # Notify the user of successful registration
        QMessageBox.information(self, "Success", "User registration successful")
        self.accept()

# -------------------
# Login dialog module
# -------------------
class LoginDialog(QDialog):
    def __init__(self, font_family):
        super().__init__()
        self.setWindowTitle('Login')
        self.setFixedSize(400, 200)
        self.font_family = font_family
        self.center()

        # Create layout
        self.layout = QVBoxLayout()
        self.form_layout = QFormLayout()

        # Create username and password input boxes
        self.username_input = QLineEdit(self)
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)

        # Set input box styles
        self.username_input.setStyleSheet(f"""
            QLineEdit {{
                border: 2px solid #dcdcdc;
                border-radius: 5px;
                padding: 5px;
                font-size: 16px;
                background-color: #f9f9f9;
                font-family: 'Times New Roman';
                font-weight: bold;
            }}
            QLineEdit:focus {{
                border-color: #a0a0a0;
            }}
        """)
        self.password_input.setStyleSheet(f"""
            QLineEdit {{
                border: 2px solid #dcdcdc;
                border-radius: 5px;
                padding: 5px;
                font-size: 16px;
                background-color: #f9f9f9;
                font-family: 'Times New Roman';
                font-weight: bold;
            }}
            QLineEdit:focus {{
                border-color: #a0a0a0;
            }}
        """)

        # Create labels and add to form layout
        username_label = QLabel("Username:", self)
        username_label.setFont(QFont('Times New Roman', 16, QFont.Bold))
        username_label.setAlignment(Qt.AlignCenter)

        password_label = QLabel("Password:", self)
        password_label.setFont(QFont('Times New Roman', 16, QFont.Bold))
        password_label.setAlignment(Qt.AlignCenter)

        self.form_layout.addRow(username_label, self.username_input)
        self.form_layout.addRow(password_label, self.password_input)

        # Create login button and connect to the function for logging in users
        self.login_button = QPushButton('Login', self)
        self.login_button.clicked.connect(self.login_user)

        # Set button style
        self.login_button.setFont(QFont('Times New Roman', 16, QFont.Bold))
        self.login_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #2196F3;
                color: white;
                border-radius: 10px;
                padding: 8px 16px;
                min-width: 130px;
                font-family: 'Times New Roman';
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #0b7dda;
            }}
        """)

        # Add form layout and button layout to the main layout
        self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.layout.addLayout(self.form_layout)
        self.layout.addWidget(self.login_button, alignment=Qt.AlignCenter)
        self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Set main layout
        self.setLayout(self.layout)

    # Center the dialog
    def center(self):
        qr = self.frameGeometry()
        cp = QGuiApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # Function to log in users
    def login_user(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        # Check if username and password are empty
        if not username or not password:
            QMessageBox.warning(self, "Error", "Username and password cannot be empty")
            return

        # Read the user file and check login information
        user_file = '/Users/code/Desktop_Application/users.json'
        if os.path.exists(user_file):
            with open(user_file, 'r') as file:
                users = json.load(file)
        else:
            users = {}

        # Validate username and password
        if username not in users or users[username]['password'] != password:
            QMessageBox.warning(self, "Error", "Invalid username or password")
            return

        # Notify the user of successful login
        success_msg = QMessageBox(self)
        success_msg.setWindowTitle("Success")
        success_msg.setFont(QFont('Times New Roman', 16, QFont.Bold))
        success_msg.setStyleSheet("QLabel{min-width: 100px;}")

        success_msg.setIconPixmap(QPixmap("/Users/code/Desktop_Application/icons/smile.png").scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        success_msg.setText("<p style='text-align: center;'>Login successful</p>")

        success_msg.exec_()

        self.accept()

# -------------------
# Button window module
# -------------------
class ButtonWindow(QMainWindow):
    def __init__(self, font_family):
        super().__init__()
        self.setWindowTitle("OIODNet")
        self.setGeometry(100, 100, 1300, 800)
        self.font_family = font_family
        self.center()

        self.setStyleSheet("background-color: white;")

        # Create the main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Top rectangle and logo
        top_rect = QWidget(self)
        top_rect.setFixedHeight(50)
        top_rect.setStyleSheet("background-color: #2133ED;")

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(10, 0, 10, 0)

        # Add application logo
        self.svg_widget = QSvgWidget("/Users/code/Desktop_Application/icons/logo.svg")
        self.svg_widget.setFixedSize(40, 40)
        top_layout.addWidget(self.svg_widget, alignment=Qt.AlignLeft | Qt.AlignVCenter)

        # Add title
        title_label = QLabel("OIODNet", self)
        title_label.setFont(QFont(self.font_family, 18, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        top_layout.addWidget(title_label, alignment=Qt.AlignLeft | Qt.AlignVCenter)

        top_layout.addStretch()
        top_rect.setLayout(top_layout)

        # Add the top rectangle to the main layout
        main_layout.addWidget(top_rect, alignment=Qt.AlignTop)

        # Image display area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(20, 20, 20, 10)

        self.page_image_label = QLabel()
        pixmap = QPixmap("/Users/code/Desktop_Application/background_page/page_0.png")
        if pixmap.isNull():
            self.page_image_label.setText("Unable to load image")
        else:
            self.page_image_label.setPixmap(pixmap)
            self.page_image_label.setAlignment(Qt.AlignCenter)
            content_layout.addWidget(self.page_image_label)
            self.adjust_page_image_size()

        content_widget.setLayout(content_layout)
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

        spacer = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Fixed)
        main_layout.addSpacerItem(spacer)

        # Create function buttons
        button_names = [
            "Background", "Data Collection", "Data Processing", "Model Selection", 
            "Model Evaluation", "Coating Preparation", "Experimental Validation", "Future Prospects"
        ]
        button_icons = [
            "/Users/code/Desktop_Application/icons/old-man.png",
            "/Users/code/Desktop_Application/icons/data-collection.png",
            "/Users/code/Desktop_Application/icons/image-processing.png",
            "/Users/code/Desktop_Application/icons/deep-learning.png",
            "/Users/code/Desktop_Application/icons/analytics.png",
            "/Users/code/Desktop_Application/icons/laboratory.png",
            "/Users/code/Desktop_Application/icons/management.png",
            "/Users/code/Desktop_Application/icons/potential.png"
        ]

        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)

        row, col = 0, 0
        for name, icon_path in zip(button_names, button_icons):
            button = QPushButton(name, self)
            button.setFont(QFont(self.font_family, 20, QFont.Bold))
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: #007bff;
                    color: white;
                    border-radius: 10px;
                    padding: 8px 8px 8px 8px;
                    margin: 10px 10;
                    font-family: 'Times New Roman';
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #0056b3;
                }}
            """)
            if icon_path:
                icon = QIcon(icon_path)
                button.setIcon(icon)
                button.setIconSize(QSize(24, 24))
            button.clicked.connect(lambda checked, name=name: self.open_main_window(name))
            grid_layout.addWidget(button, row, col)
            col += 1
            if col > 3:
                col = 0
                row += 1

        main_layout.addLayout(grid_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # Center the window
    def center(self):
        qr = self.frameGeometry()
        cp = QGuiApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # Adjust the image size
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_page_image_size()

    def adjust_page_image_size(self):
        if self.page_image_label.pixmap():
            container_width = self.width() - 40
            container_height = self.height() - 250
            pixmap = self.page_image_label.pixmap()
            if not pixmap.isNull():
                pixmap = pixmap.scaled(container_width, container_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.page_image_label.setPixmap(pixmap)

    # Open the main window and switch to the corresponding function page
    def open_main_window(self, name):
        self.main_window = MainWindow(self.font_family, name, self)
        QTimer.singleShot(0, self.main_window.show)
        self.close()

# -------------------
# Welcome window module
# -------------------
class WelcomeWindow(QMainWindow):
    def __init__(self, font_family):
        super().__init__()
        self.setWindowTitle("Welcome")
        self.setFixedSize(400, 200)
        self.font_family = font_family
        self.center()

        self.layout = QVBoxLayout()

        # Welcome label and logo
        self.label = QLabel("Welcome to OIODNet", self)
        self.label.setFont(QFont(self.font_family, 18, QFont.Bold))
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.svg_widget = QSvgWidget("/Users/code/Desktop_Application/icons/logo.svg")
        self.svg_widget.setFixedSize(70, 70)
        self.layout.addWidget(self.svg_widget, alignment=Qt.AlignCenter)

        # Login and Register buttons
        button_layout = QHBoxLayout()
        self.login_button = QPushButton("Login", self)
        self.login_button.setFont(QFont(self.font_family, 16, QFont.Bold))
        self.login_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #007bff;
                color: white;
                border-radius: 10px;
                padding: 8px 16px;
                min-width: 130px;
                margin: 5px;
                font-family: 'Times New Roman';
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #0056b3;
            }}
        """)
        self.login_button.clicked.connect(self.open_login_dialog)
        button_layout.addWidget(self.login_button)

        self.register_button = QPushButton("Register", self)
        self.register_button.setFont(QFont(self.font_family, 16, QFont.Bold))
        self.register_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #28a745;
                color: white;
                border-radius: 10px;
                padding: 8px 16px;
                min-width: 130px;
                margin: 5px;
                font-family: 'Times New Roman';
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #218838;
            }}
        """)
        self.register_button.clicked.connect(self.open_register_dialog)
        button_layout.addWidget(self.register_button)

        self.layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    # Center the window
    def center(self):
        qr = self.frameGeometry()
        cp = QGuiApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # Open login dialog
    def open_login_dialog(self):
        login_dialog = LoginDialog(self.font_family)
        if login_dialog.exec_() == QDialog.Accepted:
            self.open_button_window()

    # Open registration dialog
    def open_register_dialog(self):
        register_dialog = RegisterDialog(self.font_family)
        register_dialog.exec_()

    # Open button window
    def open_button_window(self):
        self.button_window = ButtonWindow(self.font_family)
        self.button_window.show()
        self.close()

# -------------------
# Main window module
# -------------------
class MainWindow(QMainWindow):
    def __init__(self, font_family, selected_tab, parent=None):
        super().__init__(parent)
        self.font_family = font_family
        self.parent_window = parent

        self.setWindowTitle("Osteogenic capability predictor")
        self.setGeometry(100, 100, 1300, 800)
        self.setMinimumSize(800, 600)
        self.setMaximumSize(1920, 1080)
        self.center()  # Call the center method

        self.setStyleSheet("background-color: white;")

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Top rectangle and logo
        top_rect = QWidget(self)
        top_rect.setFixedHeight(50)
        top_rect.setStyleSheet("background-color: #2133ED;")

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(10, 0, 10, 0)

        self.svg_widget = QSvgWidget("/Users/code/Desktop_Application/icons/logo.svg")
        self.svg_widget.setFixedSize(40, 40)
        top_layout.addWidget(self.svg_widget, alignment=Qt.AlignLeft | Qt.AlignVCenter)

        title_label = QLabel("OIODNet", self)
        title_label.setFont(QFont(self.font_family, 18, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        top_layout.addWidget(title_label, alignment=Qt.AlignLeft | Qt.AlignVCenter)

        top_layout.addStretch()
        self.back_button = QPushButton("Return", self)
        self.back_button.setFont(QFont(self.font_family, 18, QFont.Bold))
        self.back_button.setFixedWidth(100)
        self.back_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #f44336;
                color: white;
                border-radius: 10px;
                padding: 8px;
                font-family: 'Times New Roman';
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #d32f2f;
            }}
        """)
        self.back_button.clicked.connect(self.go_back)
        
        top_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum))
        top_layout.addWidget(self.back_button, alignment=Qt.AlignRight | Qt.AlignVCenter)

        top_rect.setLayout(top_layout)

        main_layout.addWidget(top_rect)

        # QStackedWidget for switching between multiple pages
        self.stacked_widget = QStackedWidget()

        # Define page contents
        pages = [
            ("Background", self.create_background_section()),
            ("Data Collection", self.create_data_collection_section()),
            ("Data Processing", self.create_section_with_image("Data Processing", "/Users/code/Desktop_Application/background_page/page_5.png")),
            ("Model Selection", self.create_section_with_image("Model Selection", "/Users/code/Desktop_Application/background_page/page_6.png")),
            ("Model Evaluation", self.create_evaluation_section()),
            ("Coating Preparation", self.create_section_with_image("Coating Preparation", "/Users/code/Desktop_Application/background_page/page_4.png")),
            ("Experimental Validation", self.create_experiment_tab()),
            ("Future Prospects", self.create_section_with_image("Future Prospects", "/Users/code/Desktop_Application/background_page/page_7.png"))
        ]

        # Map page names to their corresponding content
        self.page_map = {}
        for title, content in pages:
            self.page_map[title] = content
            self.stacked_widget.addWidget(content)

        main_layout.addWidget(self.stacked_widget)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Load OIODNet and load the trained weights
        try:
            self.model = OIODNet().to('cpu')  # Initialize model
            checkpoint_path = '/Users/code/OIODNet/oiodnet_average_weights.pth'
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))  # Load weights
            self.model.eval()  # Set the model to evaluation mode
        except Exception as e:
            print(f"Error loading model: {e}")

        # Switch to the user-selected page
        self.select_tab(selected_tab)
        self.update()
        self.repaint()

    # Add center method
    def center(self):
        qr = self.frameGeometry()
        cp = QGuiApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # Create a page with an image
    def create_section_with_image(self, title, image_path):
        section_widget = QWidget()
        section_layout = QVBoxLayout()

        title_label = QLabel(title, self)
        title_label.setFont(QFont(self.font_family, 40, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"color: #2980b9; margin: 20px 0; font-family: '{self.font_family}'; font-weight: bold;")
        section_layout.addWidget(title_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        content = QWidget()
        content_layout = QVBoxLayout()

        self.page_image_label = QLabel()
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.page_image_label.setText("Unable to load image")
        else:
            self.page_image_label.setPixmap(pixmap)
            self.page_image_label.setAlignment(Qt.AlignCenter)
            content_layout.addWidget(self.page_image_label)
            self.adjust_page_image_size()

        content.setLayout(content_layout)
        scroll_area.setWidget(content)
        section_layout.addWidget(scroll_area)

        section_widget.setLayout(section_layout)

        return section_widget

    # Adjust the page image size
    def adjust_page_image_size(self):
        if hasattr(self, 'page_image_label') and self.page_image_label.pixmap():
            container_width = self.width() - 40  # Adjust width, leaving some margin
            container_height = self.height() - 280  # Adjust height, leaving some margin
            pixmap = self.page_image_label.pixmap()
            if not pixmap.isNull():
                pixmap = pixmap.scaled(container_width, container_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.page_image_label.setPixmap(pixmap)

    # Create background section
    def create_background_section(self):
        stacked_widget = QStackedWidget()

        pages = [
            ("Significance of fundamental research on orthopedic implant surface modification and clinical application prospects", "/Users/code/Desktop_Application/background_page/page_1.png"),
            ("Traditional experimental evaluation of osteogenic properties of orthopedic implants", "/Users/code/Desktop_Application/background_page/page_2.png"),
            ("AI-enabled high-throughput screening of orthopedic implant surfaces' osteogenic properties based on cell morphology", "/Users/code/Desktop_Application/background_page/page_3.png"),
            ("AI prediction of osteogenic properties of orthopedic implant surfaces and surface coating preparation", "/Users/code/Desktop_Application/background_page/page_4.png")
        ]

        for title, image_path in pages:
            page = QWidget()
            layout = QVBoxLayout()

            title_label = QLabel(title, self)
            title_label.setFont(QFont(self.font_family, 40, QFont.Bold))
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet(f"color: #2C68AE; font-family: '{self.font_family}'; font-weight: bold; margin-top: 0px;")
            layout.addWidget(title_label)

            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setStyleSheet("QScrollArea { border: none; }")

            content_widget = QWidget()
            content_layout = QVBoxLayout()

            self.page_image_label = QLabel()
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                self.page_image_label.setText("Unable to load image")
            else:
                self.page_image_label.setPixmap(pixmap)
                self.page_image_label.setAlignment(Qt.AlignCenter)
                content_layout.addWidget(self.page_image_label)
                self.adjust_page_image_size()

            content_widget.setLayout(content_layout)
            scroll_area.setWidget(content_widget)
            layout.addWidget(scroll_area)

            page.setLayout(layout)
            stacked_widget.addWidget(page)

        button_layout = QHBoxLayout()
        prev_button = QPushButton(self)
        prev_button.setIcon(QIcon("/Users/code/Desktop_Application/icons/left_arrow.png"))
        prev_button.setIconSize(QSize(48, 48))
        prev_button.setStyleSheet("border: none;")
        prev_button.setFixedSize(60, 60)
        prev_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(stacked_widget.currentIndex() - 1 if stacked_widget.currentIndex() > 0 else 0))

        next_button = QPushButton(self)
        next_button.setIcon(QIcon("/Users/code/Desktop_Application/icons/right_arrow.png"))
        next_button.setIconSize(QSize(48, 48))
        next_button.setStyleSheet("border: none;")
        next_button.setFixedSize(60, 60)
        next_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(stacked_widget.currentIndex() + 1 if stacked_widget.currentIndex() < stacked_widget.count() - 1 else stacked_widget.count() - 1))

        button_layout.addWidget(prev_button, alignment=Qt.AlignLeft | Qt.AlignBottom)
        button_layout.addWidget(next_button, alignment=Qt.AlignRight | Qt.AlignBottom)

        main_layout = QVBoxLayout()
        main_layout.addWidget(stacked_widget)
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        return container

    # Create data collection page
    def create_data_collection_section(self):
        stacked_widget = QStackedWidget()

        image_dir = "/Users/code/Desktop_Application/ALP/"
        pages = [f"{image_dir}{i}.png" for i in range(1, 13)]

        for image_path in pages:
            page = QWidget()
            layout = QVBoxLayout()

            title_label = QLabel("Early time-point cell morphology image database", self)
            title_label.setFont(QFont(self.font_family, 40, QFont.Bold))
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet(f"color: #2980b9; margin: 20px 0; font-family: '{self.font_family}'; font-weight: bold;")
            layout.addWidget(title_label)

            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setStyleSheet("QScrollArea { border: none; }")
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            content_widget = QWidget()
            content_layout = QVBoxLayout()

            self.page_image_label = QLabel()
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                self.page_image_label.setText("Unable to load image")
            else:
                self.page_image_label.setPixmap(pixmap)
                self.page_image_label.setAlignment(Qt.AlignCenter)
                content_layout.addWidget(self.page_image_label)
                self.adjust_page_image_size()

            content_widget.setLayout(content_layout)
            scroll_area.setWidget(content_widget)
            layout.addWidget(scroll_area)

            page.setLayout(layout)
            stacked_widget.addWidget(page)

        button_layout = QHBoxLayout()
        prev_button = QPushButton(self)
        prev_button.setIcon(QIcon("/Users/code/Desktop_Application/icons/left_arrow.png"))
        prev_button.setIconSize(QSize(48, 48))
        prev_button.setStyleSheet("border: none;")
        prev_button.setFixedSize(60, 60)
        prev_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(stacked_widget.currentIndex() - 1 if stacked_widget.currentIndex() > 0 else 0))

        next_button = QPushButton(self)
        next_button.setIcon(QIcon("/Users/code/Desktop_Application/icons/right_arrow.png"))
        next_button.setIconSize(QSize(48, 48))
        next_button.setStyleSheet("border: none;")
        next_button.setFixedSize(60, 60)
        next_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(stacked_widget.currentIndex() + 1 if stacked_widget.currentIndex() < stacked_widget.count() - 1 else stacked_widget.count() - 1))

        button_layout.addWidget(prev_button, alignment=Qt.AlignLeft | Qt.AlignBottom)
        button_layout.addWidget(next_button, alignment=Qt.AlignRight | Qt.AlignBottom)

        main_layout = QVBoxLayout()
        main_layout.addWidget(stacked_widget)
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        return container

    # Create model evaluation page
    def create_evaluation_section(self):
        stacked_widget = QStackedWidget()

        image_dir = "/Users/code/Desktop_Application/background_page/"
        pages = [f"{image_dir}page_{i}.png" for i in range(8, 13)]

        for image_path in pages:
            page = QWidget()
            layout = QVBoxLayout()

            title_label = QLabel("Model Evaluation", self)
            title_label.setFont(QFont(self.font_family, 40, QFont.Bold))
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet(f"color: #2980b9; margin: 20px 0; font-family: '{self.font_family}'; font-weight: bold;")
            layout.addWidget(title_label)

            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setStyleSheet("QScrollArea { border: none; }")
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            content_widget = QWidget()
            content_layout = QVBoxLayout()

            self.page_image_label = QLabel()
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                self.page_image_label.setText("Unable to load image")
            else:
                self.page_image_label.setPixmap(pixmap)
                self.page_image_label.setAlignment(Qt.AlignCenter)
                content_layout.addWidget(self.page_image_label)
                self.adjust_page_image_size()

            content_widget.setLayout(content_layout)
            scroll_area.setWidget(content_widget)
            layout.addWidget(scroll_area)

            page.setLayout(layout)
            stacked_widget.addWidget(page)

        button_layout = QHBoxLayout()
        prev_button = QPushButton(self)
        prev_button.setIcon(QIcon("/Users/code/Desktop_Application/icons/left_arrow.png"))
        prev_button.setIconSize(QSize(48, 48))
        prev_button.setStyleSheet("border: none;")
        prev_button.setFixedSize(60, 60)
        prev_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(stacked_widget.currentIndex() - 1 if stacked_widget.currentIndex() > 0 else 0))

        next_button = QPushButton(self)
        next_button.setIcon(QIcon("/Users/code/Desktop_Application/icons/right_arrow.png"))
        next_button.setIconSize(QSize(48, 48))
        next_button.setStyleSheet("border: none;")
        next_button.setFixedSize(60, 60)
        next_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(stacked_widget.currentIndex() + 1 if stacked_widget.currentIndex() < stacked_widget.count() - 1 else stacked_widget.count() - 1))

        button_layout.addWidget(prev_button, alignment=Qt.AlignLeft | Qt.AlignBottom)
        button_layout.addWidget(next_button, alignment=Qt.AlignRight | Qt.AlignBottom)

        main_layout = QVBoxLayout()
        main_layout.addWidget(stacked_widget)
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        return container

    # Create experimental validation page
    def create_experiment_tab(self):
        tab_widget = QWidget()
        layout = QVBoxLayout()

        title_label = QLabel("Orthopedic Implants-Osteogenic Differentiation (OIODNet)", self)
        title_label.setFont(QFont(self.font_family, 30, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"color: #2980b9; margin: 20px 0; font-family: '{self.font_family}'; font-weight: bold;")
        layout.addWidget(title_label)

        instruction_label = QLabel("Please take or upload an early-time point cell image", self)
        instruction_label.setFont(QFont(self.font_family, 20, QFont.Bold))
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setStyleSheet(f"color: #666; margin-bottom: 20px; font-family: '{self.font_family}'; font-weight: bold;")
        layout.addWidget(instruction_label)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(300, 300)
        self.image_label.setStyleSheet("border: none; margin-bottom: 20px;")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.result_label = QLabel("", self)
        self.result_label.setFont(QFont(self.font_family, 24, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(f"color: #007bff; font-family: '{self.font_family}'; font-weight: bold;")
        layout.addWidget(self.result_label)

        button_layout = QHBoxLayout()
        
        self.capture_button = QPushButton("Please click to photograph", self)
        self.capture_button.setFont(QFont(self.font_family, 14, QFont.Bold))
        self.set_icon(self.capture_button, '/Users/code/Desktop_Application/icons/upload_icon.png')
        self.capture_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #007bff;
                color: white;
                border-radius: 10px;
                padding: 4px 8px;
                min-width: 100px;
                font-family: '{self.font_family}';
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #0056b3;
            }}
        """)
        self.capture_button.setIconSize(QSize(24, 24))
        self.capture_button.clicked.connect(self.setup_camera)
        button_layout.addWidget(self.capture_button)

        self.upload_button = QPushButton("Please upload images", self)
        self.upload_button.setFont(QFont(self.font_family, 14, QFont.Bold))
        self.set_icon(self.upload_button, '/Users/code/Desktop_Application/icons/upload_icon.png')
        self.upload_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #007bff;
                color: white;
                border-radius: 10px;
                padding: 4px 8px;
                min-width: 100px;
                font-family: '{self.font_family}';
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #0056b3;
            }}
        """)
        self.upload_button.setIconSize(QSize(24, 24))
        self.upload_button.clicked.connect(self.upload_image)
        button_layout.addWidget(self.upload_button)

        layout.addStretch()
        layout.addLayout(button_layout)

        tab_widget.setLayout(layout)
        return tab_widget

    # Setup the camera
    def setup_camera(self):
        available_cameras = QCameraInfo.availableCameras()
        if available_cameras:
            self.camera = QCamera(available_cameras[0])
            self.camera.setViewfinder(QCameraViewfinder())
            self.camera.start()

            self.image_capture = QCameraImageCapture(self.camera)
            self.image_capture.setCaptureDestination(QCameraImageCapture.CaptureToFile)
            self.image_capture.imageCaptured.connect(self.process_captured_image)
            self.capture_image()

    # Capture image
    def capture_image(self):
        if self.camera and self.camera.state() == QCamera.ActiveState:
            self.image_capture.capture("/tmp/captured_image.jpg")

    # Process captured image
    def process_captured_image(self, request_id, image):
        self.image_label.setPixmap(QPixmap.fromImage(image))
        self.image_label.setScaledContents(True)
        self.predict_performance("/tmp/captured_image.jpg")

    # Set icon
    def set_icon(self, button, icon_path_base):
        icon_path = self.get_image_path(icon_path_base)
        if icon_path:
            button.setIcon(QIcon(icon_path))

    def get_image_path(self, base_path):
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']:
            full_path = base_path + ext
            if os.path.exists(full_path):
                return full_path
        return None

    # Upload image
    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
            self.predict_performance(file_name)

    # Predict image performance
    def predict_performance(self, file_path):
        image = Image.open(file_path).resize((224, 224))
        image = np.array(image) / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to('cpu')
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            class_label = 'CLASS1' if predicted.item() == 0 else 'CLASS2'
        self.result_label.setText(f"Predict: {class_label}")

    # Select tab
    def select_tab(self, name):
        if name in self.page_map:
            self.stacked_widget.setCurrentWidget(self.page_map[name])

    # Return to the parent window
    def go_back(self):
        self.parent_window.show()
        self.close()

# -------------------
# Main function
# -------------------
def main():
    app = QApplication(sys.argv)

    # Load font
    font_family = "Times New Roman"

    welcome_window = WelcomeWindow(font_family)
    welcome_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()