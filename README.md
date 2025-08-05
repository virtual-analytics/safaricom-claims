# Minet Insurance Analytics Platform

## 1. Project Overview

The Minet Insurance Analytics Platform is a comprehensive, data-driven web application built with Django. It is designed to provide advanced analytics, predictive modeling, and operational tools for managing insurance claims. The platform offers modules for claim volume prediction, fraud detection, client management, and in-depth reporting, with a specialized dashboard for key clients like Safaricom.

## 2. Features

The platform is divided into several key functional areas:

*   **Core Features:**
    *   **User Authentication:** Secure login, logout, and session management.
    *   **Central Dashboard (`/home/`):** A landing page for authenticated users, providing an overview of key metrics.
    *   **Client Management (`/client_management/`):** Tools for managing client information.

*   **Data Processing & Preparation:**
    *   **Data Cleaning (`/data_cleaning/`):** An interactive interface to clean and preprocess raw claims data.
    *   **AJAX-powered Cleaning (`/clean_data_ajax/`):** Asynchronous data cleaning operations for a smoother user experience.

*   **Claims Prediction (`/claims-prediction/`):**
    *   Forecast future claim volumes.
    *   Visualize confidence intervals for predictions.
    *   Simulate the impact of various factors on claim numbers.
    *   Provide model explainability to understand prediction drivers.
    *   Dynamically update charts and filters via AJAX.

*   **Fraud Detection (`/fraud-detection/`):**
    *   Assign risk scores to individual claims.
    *   Identify suspicious providers and diagnosis patterns.
    *   Analyze monthly trends in fraudulent activity.
    *   Generate lists of potential fraud cases for investigation.
    *   Visualize fraud hotspots with a geospatial heatmap.

*   **Advanced Analytics & Reporting:**
    *   **Safaricom Client Dashboard (`/safaricom/`):** A dedicated analytics suite for Safaricom, showing claim distributions, temporal analysis, provider efficiency, and diagnosis patterns.
    *   **Exploratory Analysis (`/exploratory_analysis/`):** Tools for ad-hoc data exploration and visualization.
    *   **Reporting (`/reporting/`):** Generate and download custom reports with drill-down capabilities and advanced filters.
    *   **Impact Analysis (`/impact_analysis/`):** Assess the impact of specific events or changes on claim metrics.

*   **Machine Learning Operations:**
    *   **Model Training (`/model_training/`):** Interface for training and retraining predictive models.
    *   **Batch Predictions (`/make_predictions/`):** Run the trained models on new data to generate predictions.

*   **Experimental Features:**
    *   **Agentic AI (`/agentic_ai/`):** An R&D module for exploring agent-based AI solutions.

## 3. Tech Stack

*   **Backend:** Python, Django
*   **Data Science:** Pandas, Scikit-learn, (Potentially: TensorFlow/PyTorch, Statsmodels)
*   **Database:** (Presumed: PostgreSQL / MySQL / SQLite for development)
*   **Frontend:** HTML, CSS, JavaScript (with extensive use of AJAX/Fetch API)
*   **Deployment:** (Presumed: Gunicorn, Nginx, Docker)

## 4. Prerequisites

Before you begin, ensure you have the following installed on your system:
*   Python (3.8+ recommended)
*   `pip` (Python package installer)
*   `virtualenv` (for creating isolated Python environments)
*   A database system (e.g., PostgreSQL)

## 5. Installation and Setup

Follow these steps to get the project running on your local machine.

**1. Clone the Repository**
```bash
git clone <your-repository-url>
cd Minet
```

**2. Create and Activate a Virtual Environment**
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
*(Note: A `requirements.txt` file is assumed. If it doesn't exist, you'll need to create one.)*
```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables**
Create a `.env` file in the root directory (`d:\Minet\`) by copying an example file (e.g., `.env.example`).

Now, open the `.env` file and fill in the required values for your local setup:
```ini
# .env file
# Django Settings
SECRET_KEY='your-secret-key-here'
DEBUG=True

# Database Settings (Example for PostgreSQL)
DB_NAME='minet_db'
DB_USER='minet_user'
DB_PASSWORD='your_db_password'
DB_HOST='localhost'
DB_PORT='5432'
```

**5. Apply Database Migrations**
```bash
python manage.py migrate
```

**6. Create a Superuser**
This will allow you to access the Django admin panel.
```bash
python manage.py createsuperuser
```
Follow the prompts to create your admin account.

**7. Run the Development Server**
```bash
python manage.py runserver
```
The application will be available at `http://127.0.0.1:8000`.

## 6. URL Structure

The application's endpoints are organized logically by feature.

*   `/`: Landing page
*   `/login/`, `/logout/`: User authentication
*   `/home/`: Main user dashboard

*   **Safaricom Module:** `/safaricom/*`
*   **Claims Prediction Module:** `/claims-prediction/*`
*   **Fraud Detection Module:** `/fraud-detection/*`
*   **Reporting Module:** `/reporting/*`

*   **AJAX Endpoints:**
    *   `/clean_data_ajax/`
    *   `/claims_overview_ajax/`
    *   `/fraud_detection_ajax/`
    *   `/exploratory_analysis_ajax/`
    *   `/advanced_analysis_ajax/`

## 7. Running Tests

To run the automated test suite for the application, use the following command:
```bash
python manage.py test myapp
```

## 8. Contributing

Contributions are welcome! Please follow these steps:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

## 9. License

This project is licensed under the MIT License - see the LICENSE.md file for details. (Please update with your chosen license).