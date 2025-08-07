from django.urls import path
from . import views

urlpatterns = [
    # Existing URLs
    path('', views.landing, name='landing'),
    path('login/', views.login_view, name='login'),
    path('home/', views.home_view, name='home'),
    path('logout/', views.logout_view, name='logout'),

    # Claims Prediction Dataset View
    path('claim_prediction/', views.claims_prediction_dataset_view, name='claim_prediction'),

    path('fraud_detection/', views.fraud_detection, name='fraud_detection'),
    path('client_management/', views.client_management, name='client_management'),
    path('reports/', views.reports, name='reports'),
    path('exploratory_analysis/', views.exploratory_analysis, name='exploratory_analysis'),
    path('model_training/', views.model_training, name='model_training'),
    path('make_predictions/', views.make_predictions, name='make_predictions'),
    path('impact_analysis/', views.impact_analysis, name='impact_analysis'),
    path('agentic_ai/', views.agentic_ai, name='agentic_ai'),
    path('data_cleaning/', views.data_cleaning, name='data_cleaning'),
    path('clean_data_ajax/', views.clean_data_ajax, name='clean_data_ajax'),
    path('claims_overview_ajax/', views.claims_overview_ajax, name='claims_overview_ajax'),
    path('fraud_detection_ajax/', views.fraud_detection_ajax, name='fraud_detection_ajax'),
    path('exploratory_analysis_ajax/', views.exploratory_analysis_ajax, name='exploratory_analysis_ajax'),
    path('advanced_analysis_ajax/', views.advanced_analysis_ajax, name='advanced_analysis_ajax'),
    
    # Safaricom URLs
    path('safaricom/', views.safaricom_home, name='safaricom_home'),
    path('safaricom/reports/', views.advanced_analysis, name='safaricom_reports'),
    path('safaricom/data/', views.get_claim_data, name='safaricom_data'),
    path('safaricom/claim-distribution/', views.claim_distribution, name='safaricom_claim_distribution'),
    path('safaricom/temporal-analysis/', views.temporal_analysis, name='safaricom_temporal_analysis'),
    path('safaricom/provider-efficiency/', views.provider_efficiency, name='safaricom_provider_efficiency'),
    path('safaricom/diagnosis-patterns/', views.diagnosis_patterns, name='safaricom_diagnosis_patterns'),
    
    # Claims Prediction URLs
    path('claims-prediction/', views.claims_prediction_home, name='claims_prediction'),
    path('claims-prediction/forecasted-volume/', views.claims_prediction_home, name='forecasted_volume'),
    path('claims-prediction/confidence-intervals/', views.confidence_intervals, name='confidence_intervals'),
    path('claims-prediction/impact-simulation/', views.impact_simulation, name='impact_simulation'),
    path('claims-prediction/explainability/', views.explainability, name='explainability'),
    path('claims-prediction/update-charts/', views.update_charts_ajax, name='update_charts_ajax'),
    
    # Fraud Detection URLs
    path('fraud-detection/', views.fraud_detection_home, name='fraud_detection'),
    path('fraud-detection/risk-scores/', views.fraud_detection_home, name='risk_scores'),
    path('fraud-detection/suspicious-providers/', views.suspicious_providers, name='suspicious_providers'),
    path('fraud-detection/diagnosis-patterns/', views.diagnosis_patterns, name='diagnosis_patterns'),
    path('fraud-detection/monthly-trends/', views.monthly_trends, name='monthly_trends'),
    path('fraud-detection/potential-cases/', views.potential_cases, name='potential_cases'),
    path('fraud-detection/geospatial-heatmap/', views.geospatial_heatmap, name='geospatial_heatmap'),
    
    # Reporting URLs
    path('reporting/', views.reporting_home, name='reporting'),
    path('reporting/download-reports/', views.reporting_home, name='download_reports'),
    path('reporting/claim-drilldown/', views.claim_drilldown, name='claim_drilldown'),
    path('reporting/custom-filters/', views.custom_filters, name='custom_filters'),

    # -------------------------------
    # NEW: Provider Efficiency View
    # -------------------------------
    path('provider-efficiency/', views.provider_efficiency_view, name='provider_efficiency'),
    # urls.py
    path('diagnostic-patterns1/', views.diagnostic_patterns_view1, name='diagnostic-patterns1'),

]
