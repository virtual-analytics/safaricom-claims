from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.db.models import Sum, Count
from .models import ClaimRecord
import calendar
from django.db.models.functions import Cast
from django.db.models import DateField, DateTimeField, CharField, Case, When, Value
from django.db import connection
from django.utils import timezone
from datetime import datetime, timedelta
import pandas as pd
from plotly.offline import plot as plotly_plot
import plotly.express as px
from plotly.offline import plot
from datetime import datetime
from django.http import JsonResponse
import os
from django.conf import settings
from myapp.models import ClaimRecord, UserProfile
import pandas as pd
import numpy as np
from django.db.models.functions import ExtractHour, ExtractMonth, ExtractYear, ExtractWeekDay
import plotly.io as pio
from datetime import datetime
import json
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils
from plotly.offline import plot
from django.template.loader import render_to_string
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly
from django.db.models.functions import ExtractHour, ExtractYear
from django.db.models import Case, When, Value, CharField
from django.db.models import Count, Sum, Avg, F, Q

from datetime import datetime, timedelta

import plotly.graph_objects as go

from collections import defaultdict



templates_dir = os.path.join(settings.BASE_DIR, 'myapp', 'templates', 'myapp')
os.makedirs(templates_dir, exist_ok=True)

# Create your views here.

def landing(request):
    if request.user.is_authenticated:
        return redirect('home')  # Optional: skip landing for logged-in users
    return render(request, 'myapp/landing.html')



def login_view(request):
    if request.user.is_authenticated:
        # Users who are already logged in are redirected.
        return redirect('claim_prediction') 

    if request.method == 'POST':
        # Use Django's AuthenticationForm to validate the login credentials.
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            
            # The existing redirection logic based on user role is preserved.
            try:
                profile = user.userprofile
                if profile.role == 'admin':
                    return redirect('/admin/')
                elif profile.role == 'analyst':
                    return redirect('exploratory_analysis')
                elif profile.role == 'manager':
                    return redirect('reports')
                elif profile.role == 'safaricom':
                    return redirect('safaricom_home')
                else:
                    return redirect('home')
            except UserProfile.DoesNotExist:
                # A sensible default for users without a profile.
                return redirect('home')
    else:
        # For a GET request, create a new, blank form.
        form = AuthenticationForm()

    # Render the login page with the form.
    return render(request, 'myapp/login.html', {'form': form})


def get_database_tables():
    """Return all non-SQLite internal tables from the database."""
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            AND name NOT LIKE 'sqlite_%'
            AND name NOT LIKE 'django_%'
            AND name NOT LIKE 'auth_%'
            AND name NOT LIKE 'sessions%'
        """)
        return [row[0] for row in cursor.fetchall()]
    



# =========================================
# HOME VIEW - Claims Data Upload & Summary
# =========================================
@login_required(login_url='login')
def home_view(request):
    dataset_ids = get_database_tables()
    selected_id = request.GET.get('dataset_id')
    show_stats = 'desc_btn' in request.GET
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    desc_stats = None
    visualizations = None
    username = request.user.username

    # Save dataset selection to session for later use
    if selected_id:
        request.session['selected_dataset'] = selected_id
    else:
        # If no GET param but session exists, use session dataset
        selected_id = request.session.get('selected_dataset')

    if selected_id and selected_id in dataset_ids:
        df = pd.read_sql(f'SELECT * FROM "{selected_id}"', connection)

        if not df.empty:
            # Ensure amount is numeric
            if 'amount' in df.columns:
                df['amount'] = pd.to_numeric(
                    df['amount'].astype(str).str.replace(r'[^\d.]', '', regex=True).replace('', '0'),
                    errors='coerce'
                )

            # Parse and filter by date
            if 'claim_prov_date' in df.columns:
                df['datetime'] = pd.to_datetime(df['claim_prov_date'], errors='coerce', dayfirst=True)

                # Fill missing dates randomly
                if df['datetime'].isna().any():
                    start_dt = pd.to_datetime('2023-01-01')
                    end_dt = pd.to_datetime(timezone.now().date())
                    random_dates = pd.to_datetime(
                        np.random.randint(start_dt.value // 10**9, end_dt.value // 10**9, size=df['datetime'].isna().sum()),
                        unit='s'
                    )
                    df.loc[df['datetime'].isna(), 'datetime'] = random_dates

                # Apply date filters
                if start_date:
                    df = df[df['datetime'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['datetime'] <= pd.to_datetime(end_date)]

            # Descriptive statistics
            if show_stats:
                desc_stats = df.describe(include='all').transpose().reset_index().to_dict(orient='records')

            # Summary statistics
            if 'amount' in df.columns and 'claim_me' in df.columns:
                unique_members_count = df['claim_me'].nunique()
                summary_stats = {
                    'total_claims': len(df),
                    'total_amount': df['amount'].sum(),
                    'unique_members': unique_members_count,
                    'avg_claim': (df['amount'].sum() / unique_members_count) if unique_members_count else 0
                }

                # Claims Over Time chart
                claims_time_chart = None
                if 'datetime' in df.columns:
                    df_time = df.set_index('datetime').sort_index()
                    daily_df = df_time.resample('D').size().reset_index(name='count')
                    weekly_df = df_time.resample('W-MON').size().reset_index(name='count')
                    monthly_df = df_time.resample('M').size().reset_index(name='count')

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=daily_df['datetime'], y=daily_df['count'],
                        mode='lines+markers', name='Daily Claims',
                        line=dict(color='#e30613'), visible=True
                    ))
                    fig.add_trace(go.Scatter(
                        x=weekly_df['datetime'], y=weekly_df['count'],
                        mode='lines+markers', name='Weekly Claims',
                        line=dict(color='#e30613'), visible=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=monthly_df['datetime'], y=monthly_df['count'],
                        mode='lines+markers', name='Monthly Claims',
                        line=dict(color='#e30613'), visible=False
                    ))

                    fig.update_layout(
                        title="Claims Submitted Over Time",
                        xaxis_title="Date",
                        yaxis_title="Number of Claims",
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        updatemenus=[dict(
                            type="dropdown",
                            direction="down",
                            x=1.15, y=1.2,
                            showactive=True,
                            buttons=list([
                                dict(label="Daily", method="update",
                                     args=[{"visible": [True, False, False]},
                                           {"title": "Daily Claims Submitted"}]),
                                dict(label="Weekly", method="update",
                                     args=[{"visible": [False, True, False]},
                                           {"title": "Weekly Claims Submitted"}]),
                                dict(label="Monthly", method="update",
                                     args=[{"visible": [False, False, True]},
                                           {"title": "Monthly Claims Submitted"}]),
                            ]),
                        )]
                    )
                    claims_time_chart = fig.to_html(full_html=False)

                # Category amounts chart
                category_amounts = None
                if 'benefit_desc' in df.columns:
                    category_unique_df = (
                        df.groupby('benefit_desc')['claim_me']
                        .nunique()
                        .reset_index(name='unique_members')
                    )
                    fig_cat = px.bar(
                        category_unique_df, x='benefit_desc', y='unique_members',
                        title='Unique Claims by Benefit Category',
                        color_discrete_sequence=['#e30613']
                    )
                    category_amounts = fig_cat.to_html(full_html=False)

                # Sunburst breakdown
                sunburst = None
                if 'benefit' in df.columns and 'benefit_desc' in df.columns:
                    fig_sunburst = px.sunburst(
                        df.reset_index(), path=['benefit', 'benefit_desc'],
                        values='amount', title='Claim Amounts Breakdown',
                        color_discrete_sequence=['#e30613']
                    )
                    sunburst = fig_sunburst.to_html(full_html=False)

                # Top claimants
                top_claimants_df = (
                    df.groupby('claim_me')
                    .agg(total_amount=('amount', 'sum'), claim_count=('amount', 'count'))
                    .reset_index()
                    .sort_values(by='total_amount', ascending=False)
                    .head(10)
                )
                fig_top = px.bar(
                    top_claimants_df, x='claim_me', y='total_amount',
                    title='Top Claimants',
                    color_discrete_sequence=['#e30613']
                )
                top_claimants = fig_top.to_html(full_html=False)

                # Claim frequency histogram
                claim_freq_df = df['claim_me'].value_counts().reset_index(name='frequency')
                fig_freq = px.histogram(
                    claim_freq_df, x='frequency', nbins=20,
                    title='Claim Frequency Distribution',
                    color_discrete_sequence=['#e30613']
                )
                claim_freq = fig_freq.to_html(full_html=False)

                visualizations = {
                    'summary_stats': summary_stats,
                    'claims_time_chart': claims_time_chart,
                    'category_amounts': category_amounts,
                    'sunburst': sunburst,
                    'top_claimants': top_claimants,
                    'top_claimants_table': top_claimants_df.to_dict('records'),
                    'claim_freq': claim_freq
                }

    return render(request, 'home.html', {
        'dataset_ids': dataset_ids,
        'selected_id': selected_id,
        'desc_stats': desc_stats,
        'visualizations': visualizations,
        'username': username,
        'start_date': start_date,
        'end_date': end_date
    })



# =========================================
# CLAIMS PREDICTION VIEW
# =========================================
@login_required(login_url='login')
def claims_prediction_dataset_view(request):
    """Claims Prediction Engine View - same as Home View, no filters"""
    
    # ✅ Get the same table list as home_view
    dataset_ids = get_database_tables()
    selected_id = request.GET.get('dataset_id')
    show_stats = 'desc_btn' in request.GET
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    desc_stats = None
    visualizations = None
    username = request.user.username

    # ✅ Keep dataset from session if not passed in GET
    if selected_id:
        request.session['selected_dataset'] = selected_id
    else:
        selected_id = request.session.get('selected_dataset')

    if selected_id and selected_id in dataset_ids:
        df = pd.read_sql(f'SELECT * FROM "{selected_id}"', connection)

        if not df.empty:
            # Ensure amount is numeric
            if 'amount' in df.columns:
                df['amount'] = pd.to_numeric(
                    df['amount'].astype(str).str.replace(r'[^\d.]', '', regex=True).replace('', '0'),
                    errors='coerce'
                )

            # Parse and filter by date
            if 'claim_prov_date' in df.columns:
                df['datetime'] = pd.to_datetime(df['claim_prov_date'], errors='coerce', dayfirst=True)

                # Fill missing dates randomly
                if df['datetime'].isna().any():
                    start_dt = pd.to_datetime('2023-01-01')
                    end_dt = pd.to_datetime(timezone.now().date())
                    random_dates = pd.to_datetime(
                        np.random.randint(start_dt.value // 10**9, end_dt.value // 10**9, size=df['datetime'].isna().sum()),
                        unit='s'
                    )
                    df.loc[df['datetime'].isna(), 'datetime'] = random_dates

                # Apply date filters
                if start_date:
                    df = df[df['datetime'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['datetime'] <= pd.to_datetime(end_date)]

            # Descriptive statistics
            if show_stats:
                desc_stats = df.describe(include='all').transpose().reset_index().to_dict(orient='records')

            # Summary stats & charts
            if 'amount' in df.columns and 'claim_me' in df.columns:
                unique_members_count = df['claim_me'].nunique()
                summary_stats = {
                    'total_claims': len(df),
                    'total_amount': df['amount'].sum(),
                    'unique_members': unique_members_count,
                    'avg_claim': (df['amount'].sum() / unique_members_count) if unique_members_count else 0
                }

                # Claims Over Time chart
                claims_time_chart = None
                if 'datetime' in df.columns:
                    df_time = df.set_index('datetime').sort_index()
                    daily_df = df_time.resample('D').size().reset_index(name='count')
                    weekly_df = df_time.resample('W-MON').size().reset_index(name='count')
                    monthly_df = df_time.resample('M').size().reset_index(name='count')

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=daily_df['datetime'], y=daily_df['count'],
                        mode='lines+markers', name='Daily Claims',
                        line=dict(color='#e30613'), visible=True
                    ))
                    fig.add_trace(go.Scatter(
                        x=weekly_df['datetime'], y=weekly_df['count'],
                        mode='lines+markers', name='Weekly Claims',
                        line=dict(color='#e30613'), visible=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=monthly_df['datetime'], y=monthly_df['count'],
                        mode='lines+markers', name='Monthly Claims',
                        line=dict(color='#e30613'), visible=False
                    ))

                    fig.update_layout(
                        title="Claims Submitted Over Time",
                        xaxis_title="Date",
                        yaxis_title="Number of Claims",
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        updatemenus=[dict(
                            type="dropdown",
                            direction="down",
                            x=1.15, y=1.2,
                            showactive=True,
                            buttons=list([
                                dict(label="Daily", method="update",
                                     args=[{"visible": [True, False, False]},
                                           {"title": "Daily Claims Submitted"}]),
                                dict(label="Weekly", method="update",
                                     args=[{"visible": [False, True, False]},
                                           {"title": "Weekly Claims Submitted"}]),
                                dict(label="Monthly", method="update",
                                     args=[{"visible": [False, False, True]},
                                           {"title": "Monthly Claims Submitted"}]),
                            ]),
                        )]
                    )
                    claims_time_chart = fig.to_html(full_html=False)

                # Category amounts chart
                category_amounts = None
                if 'benefit_desc' in df.columns:
                    category_unique_df = (
                        df.groupby('benefit_desc')['claim_me']
                        .nunique()
                        .reset_index(name='unique_members')
                    )
                    fig_cat = px.bar(
                        category_unique_df, x='benefit_desc', y='unique_members',
                        title='Unique Claims by Benefit Category',
                        color_discrete_sequence=['#e30613']
                    )
                    category_amounts = fig_cat.to_html(full_html=False)

                # Sunburst breakdown
                sunburst = None
                if 'benefit' in df.columns and 'benefit_desc' in df.columns:
                    fig_sunburst = px.sunburst(
                        df.reset_index(), path=['benefit', 'benefit_desc'],
                        values='amount', title='Claim Amounts Breakdown',
                        color_discrete_sequence=['#e30613']
                    )
                    sunburst = fig_sunburst.to_html(full_html=False)

                # Top claimants
                top_claimants_df = (
                    df.groupby('claim_me')
                    .agg(total_amount=('amount', 'sum'), claim_count=('amount', 'count'))
                    .reset_index()
                    .sort_values(by='total_amount', ascending=False)
                    .head(10)
                )
                fig_top = px.bar(
                    top_claimants_df, x='claim_me', y='total_amount',
                    title='Top Claimants',
                    color_discrete_sequence=['#e30613']
                )
                top_claimants = fig_top.to_html(full_html=False)

                # Claim frequency histogram
                claim_freq_df = df['claim_me'].value_counts().reset_index(name='frequency')
                fig_freq = px.histogram(
                    claim_freq_df, x='frequency', nbins=20,
                    title='Claim Frequency Distribution',
                    color_discrete_sequence=['#e30613']
                )
                claim_freq = fig_freq.to_html(full_html=False)

                visualizations = {
                    'summary_stats': summary_stats,
                    'claims_time_chart': claims_time_chart,
                    'category_amounts': category_amounts,
                    'sunburst': sunburst,
                    'top_claimants': top_claimants,
                    'top_claimants_table': top_claimants_df.to_dict('records'),
                    'claim_freq': claim_freq
                }

    return render(request, 'claim_prediction.html', {
        'dataset_ids': dataset_ids,   # ✅ Ensures dropdown is always populated
        'selected_id': selected_id,   # ✅ Preserves selection between views
        'desc_stats': desc_stats,
        'visualizations': visualizations,
        'username': username,
        'start_date': start_date,
        'end_date': end_date
    })




# =========================================

# AJAX endpoint to update charts
@login_required
def update_charts_ajax(request):
    dataset_id = request.GET.get('dataset_id')
    time_period = request.GET.get('time_period', 'all')
    benefit_type = request.GET.get('benefit_type', 'all')
    group_by = request.GET.get('group_by', 'treatment')

    dataset_ids = get_database_tables()

    if not dataset_id or dataset_id not in dataset_ids:
        return JsonResponse({'error': 'Invalid dataset'}, status=400)

    df = pd.read_sql(f'SELECT * FROM "{dataset_id}"', connection)
    if df.empty:
        return JsonResponse({'error': 'Empty dataset'}, status=400)

    # Apply filters
    if 'claim_prov_date' in df.columns:
        df['datetime'] = pd.to_datetime(df['claim_prov_date'], errors='coerce', dayfirst=True)
        if time_period != 'all':
            now = pd.Timestamp.now()
            if time_period == 'year':
                df = df[df['datetime'] >= now - pd.DateOffset(years=1)]
            elif time_period == 'quarter':
                df = df[df['datetime'] >= now - pd.DateOffset(months=3)]
            elif time_period == 'month':
                df = df[df['datetime'] >= now - pd.DateOffset(months=1)]

    if benefit_type != 'all' and 'benefit' in df.columns:
        df = df[df['benefit'].str.lower() == benefit_type.lower()]

    # Grouping logic
    if group_by == 'treatment' and 'benefit_desc' in df.columns:
        group_df = df['benefit_desc'].value_counts().reset_index()
        group_df.columns = ['Label', 'Count']
    else:
        group_df = df.iloc[:0]  # empty

    # Build Plotly figure
    fig = px.bar(group_df, x='Label', y='Count', title='Updated Chart', color_discrete_sequence=['#e30613'])
    return JsonResponse({'chart_html': fig.to_html(full_html=False)})

    
    
@login_required
def clean_data_ajax(request):
    """Handle data cleaning via AJAX and return JSON response"""
    if request.method == 'POST':
        try:
            print("Starting data cleaning process...")
            print(f"Request headers: {dict(request.headers)}")
            print(f"Request POST data: {request.POST}")
            
            # Get the selected dataset (for now, always use claim_records)
            records = ClaimRecord.objects.all()
            print(f"Found {records.count()} records in database")
            
            if records.count() == 0:
                print("No records found in database")
                return JsonResponse({
                    'success': False,
                    'error': 'No data found in the database.'
                })
            
            df = pd.DataFrame(list(records.values()))
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns.tolist()}")
            print(f"DataFrame dtypes: {df.dtypes}")
            
            if not df.empty:
                # Store original shape
                original_shape = df.shape
                print(f"Original shape: {original_shape}")
                
                # Track cleaning operations
                cleaning_operations = []
                rows_removed = 0
                cols_removed = 0
                
                # 1. Drop columns with 80% or more missing values
                missing_threshold = 0.8
                columns_to_drop = []
                for col in df.columns:
                    missing_pct = df[col].isnull().sum() / len(df)
                    if missing_pct >= missing_threshold:
                        columns_to_drop.append(col)
                
                if columns_to_drop:
                    print(f"Columns to drop: {columns_to_drop}")
                    df = df.drop(columns=columns_to_drop)
                    cols_removed = len(columns_to_drop)
                    cleaning_operations.append(f"Removed {len(columns_to_drop)} columns with 80%+ missing values")
                else:
                    cleaning_operations.append("No columns removed (all columns have sufficient data)")
                
                # 2. Handle missing values in numeric columns
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                numeric_filled = 0
                for col in numeric_columns:
                    if df[col].isnull().sum() > 0:
                        missing_count = df[col].isnull().sum()
                        df[col] = df[col].fillna(df[col].median())
                        numeric_filled += missing_count
                
                if numeric_filled > 0:
                    cleaning_operations.append(f"Filled {numeric_filled} missing numeric values with median")
                else:
                    cleaning_operations.append("No missing numeric values found")
                
                # 3. Handle missing values in categorical columns
                categorical_columns = df.select_dtypes(include=['object']).columns
                categorical_filled = 0
                for col in categorical_columns:
                    if df[col].isnull().sum() > 0:
                        missing_count = df[col].isnull().sum()
                        df[col] = df[col].fillna('Unknown')
                        categorical_filled += missing_count
                
                if categorical_filled > 0:
                    cleaning_operations.append(f"Filled {categorical_filled} missing categorical values with 'Unknown'")
                else:
                    cleaning_operations.append("No missing categorical values found")
                
                # 4. Clean date formats
                date_columns = ['claim_prov_date', 'dob']
                dates_cleaned = 0
                for col in date_columns:
                    if col in df.columns:
                        # Count invalid dates before cleaning
                        invalid_dates = df[col].isnull().sum()
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        # Convert to string format, handling NaT values
                        df[col] = df[col].dt.strftime('%Y-%m-%d').fillna('')
                        dates_cleaned += invalid_dates
                
                if dates_cleaned > 0:
                    cleaning_operations.append(f"Standardized {dates_cleaned} date formats")
                else:
                    cleaning_operati__ons.append("Date formats already standardized")
                
                # 5. Remove commas from numeric fields and convert to proper types
                if 'amount' in df.columns:
                    # Count non-numeric amounts before cleaning
                    non_numeric_amounts = df['amount'].astype(str).str.contains(r'[^\d.]', regex=True).sum()
                    df['amount'] = df['amount'].astype(str).str.replace(',', '').str.replace('$', '')
                    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                    # Fill NaN with 0 for amount
                    df['amount'] = df['amount'].fillna(0)
                    if non_numeric_amounts > 0:
                        cleaning_operations.append(f"Cleaned {non_numeric_amounts} amount values (removed commas/currency symbols)")
                
                if 'quantity' in df.columns:
                    non_numeric_quantities = df['quantity'].astype(str).str.contains(r'[^\d.]', regex=True).sum()
                    df['quantity'] = df['quantity'].astype(str).str.replace(',', '')
                    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
                    # Fill NaN with 0 for quantity
                    df['quantity'] = df['quantity'].fillna(0)
                    if non_numeric_quantities > 0:
                        cleaning_operations.append(f"Cleaned {non_numeric_quantities} quantity values")
                
                # 6. Clean text fields - remove extra spaces and standardize
                text_columns = ['benefit_desc', 'prov_name', 'pol_name', 'cost_center', 'ailment']
                text_cleaned = 0
                for col in text_columns:
                    if col in df.columns:
                        # Count rows with extra spaces before cleaning
                        extra_spaces = df[col].astype(str).str.contains(r'\s{2,}', regex=True).sum()
                        df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
                        # Replace 'nan' strings with empty string
                        df[col] = df[col].replace('nan', '')
                        text_cleaned += extra_spaces
                
                if text_cleaned > 0:
                    cleaning_operations.append(f"Cleaned {text_cleaned} text fields (removed extra spaces)")
                
                # 7. Standardize categorical values
                if 'gender' in df.columns:
                    gender_standardized = 0
                    original_genders = df['gender'].value_counts()
                    df['gender'] = df['gender'].astype(str).str.upper().str.strip()
                    df['gender'] = df['gender'].replace(['M', 'MALE'], 'Male')
                    df['gender'] = df['gender'].replace(['F', 'FEMALE'], 'Female')
                    df['gender'] = df['gender'].replace(['', 'NAN', 'NONE', 'nan'], 'Unknown')
                    final_genders = df['gender'].value_counts()
                    gender_standardized = abs(original_genders.sum() - final_genders.sum())
                    if gender_standardized > 0:
                        cleaning_operations.append(f"Standardized {gender_standardized} gender values")
                
                if 'benefit' in df.columns:
                    df['benefit'] = df['benefit'].astype(str).str.upper().str.strip()
                    df['benefit'] = df['benefit'].replace('nan', '')
                
                # 8. Remove duplicates
                original_rows = len(df)
                df = df.drop_duplicates()
                final_rows = len(df)
                rows_removed = original_rows - final_rows
                
                if rows_removed > 0:
                    cleaning_operations.append(f"Removed {rows_removed} duplicate rows")
                else:
                    cleaning_operations.append("No duplicate rows found")
                
                # 9. Replace any remaining NaN values with appropriate defaults
                remaining_nans = df.isnull().sum().sum()
                if remaining_nans > 0:
                    df = df.fillna('')
                    cleaning_operations.append(f"Filled {remaining_nans} remaining missing values")
                
                print(f"Final DataFrame shape after cleaning: {df.shape}")
                print(f"Final DataFrame columns: {df.columns.tolist()}")
                
                # Store cleaned data for display - convert to records and handle NaN values
                cleaned_data_records = df.head(20).to_dict('records')
                
                # Clean the records to ensure no NaN values in JSON
                cleaned_data = []
                for record in cleaned_data_records:
                    clean_record = {}
                    for key, value in record.items():
                        if pd.isna(value):
                            clean_record[key] = ''
                        elif isinstance(value, (int, float)) and np.isnan(value):
                            clean_record[key] = 0
                        else:
                            clean_record[key] = value
                    cleaned_data.append(clean_record)
                
                print(f"Cleaned data sample size: {len(cleaned_data)}")
                print(f"Sample cleaned record: {cleaned_data[0] if cleaned_data else 'No data'}")
                
                # Calculate cleaning statistics - use the final shape after all operations
                final_shape = df.shape
                total_rows_removed = original_shape[0] - final_shape[0]
                total_cols_removed = original_shape[1] - final_shape[1]
                
                # Debug informahometion
                print(f"Original shape: {original_shape}")
                print(f"Final shape: {final_shape}")
                print(f"Total rows removed: {total_rows_removed}")
                print(f"Duplicates removed: {rows_removed}")
                print(f"Total cols removed: {total_cols_removed}")
                print(f"Columns dropped: {len(columns_to_drop)}")
                
                # Validate that statistics make sense
                if total_rows_removed < 0:
                    print("WARNING: Negative rows removed - this shouldn't happen")
                    total_rows_removed = 0
                
                if total_cols_removed < 0:
                    print("WARNING: Negative columns removed - this shouldn't happen")
                    total_cols_removed = 0
                
                cleaning_stats = {
                    'original_rows': original_shape[0],
                    'original_cols': original_shape[1],
                    'final_rows': final_shape[0],
                    'final_cols': final_shape[1],
                    'rows_removed': total_rows_removed,
                    'cols_removed': total_cols_removed,
                    'columns_dropped': columns_to_drop,
                    'missing_values_filled': numeric_filled + categorical_filled,
                    'duplicates_removed': rows_removed,  # This is just the duplicates removed
                    'cleaning_operations': cleaning_operations
                }
                
                print(f"Cleaning stats: {cleaning_stats}")
                print("Data cleaning completed successfully")
                
                response_data = {
                    'success': True,
                    'cleaned_data': cleaned_data,
                    'cleaning_stats': cleaning_stats,
                    'columns': df.columns.tolist() if not df.empty else []
                }
                
                print(f"Response data keys: {response_data.keys()}")
                print(f"Response data success: {response_data['success']}")
                print(f"Response data columns: {response_data['columns']}")
                
                return JsonResponse(response_data)
            else:
                print("DataFrame is empty after creation")
                return JsonResponse({
                    'success': False,
                    'error': 'No data found in the database.'
                })
        except Exception as e:
            print(f"Error during data cleaning: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'success': False,
                'error': f'Error during data cleaning: {str(e)}'
            })
    
    print("Invalid request method")
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def data_cleaning(request):
    if request.method == 'POST':
        try:
            # Get the selected dataset (for now, always use claim_records)
            records = ClaimRecord.objects.all()
            df = pd.DataFrame(list(records.values()))
            
            if not df.empty:
                # Store original shape
                original_shape = df.shape
                
                # 1. Drop columns with 80% or more missing values
                missing_threshold = 0.8
                columns_to_drop = []
                for col in df.columns:
                    missing_pct = df[col].isnull().sum() / len(df)
                    if missing_pct >= missing_threshold:
                        columns_to_drop.append(col)
                
                df = df.drop(columns=columns_to_drop)
                
                # 2. Handle missing values in numeric columns
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].median())
                
                # 3. Handle missing values in categorical columns
                categorical_columns = df.select_dtypes(include=['object']).columns
                for col in categorical_columns:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna('Unknown')
                
                # 4. Clean date formats
                date_columns = ['claim_prov_date', 'dob']
                for col in date_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        df[col] = df[col].dt.strftime('%Y-%m-%d')
                
                # 5. Remove commas from numeric fields and convert to proper types
                if 'amount' in df.columns:
                    df['amount'] = df['amount'].astype(str).str.replace(',', '').str.replace('$', '')
                    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                
                if 'quantity' in df.columns:
                    df['quantity'] = df['quantity'].astype(str).str.replace(',', '')
                    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
                
                # 6. Clean text fields - remove extra spaces and standardize
                text_columns = ['benefit_desc', 'prov_name', 'pol_name', 'cost_center', 'ailment']
                for col in text_columns:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
                
                # 7. Standardize categorical values
                if 'gender' in df.columns:
                    df['gender'] = df['gender'].astype(str).str.upper().str.strip()
                    df['gender'] = df['gender'].replace(['M', 'MALE'], 'Male')
                    df['gender'] = df['gender'].replace(['F', 'FEMALE'], 'Female')
                    df['gender'] = df['gender'].replace(['', 'NAN', 'NONE'], 'Unknown')
                
                if 'benefit' in df.columns:
                    df['benefit'] = df['benefit'].astype(str).str.upper().str.strip()
                
                # 8. Remove duplicates
                df = df.drop_duplicates()
                
                # Store cleaned data for display
                cleaned_data = df.head(20).to_dict('records')
                
                # Calculate cleaning statistics
                final_shape = df.shape
                rows_removed = original_shape[0] - final_shape[0]
                cols_removed = original_shape[1] - final_shape[1]
                
                cleaning_stats = {
                    'original_rows': original_shape[0],
                    'original_cols': original_shape[1],
                    'final_rows': final_shape[0],
                    'final_cols': final_shape[1],
                    'rows_removed': rows_removed,
                    'cols_removed': cols_removed,
                    'columns_dropped': columns_to_drop,
                    'missing_values_filled': len(numeric_columns) + len(categorical_columns),
                    'duplicates_removed': rows_removed
                }
                
                return render(request, 'myapp/data_cleaning.html', {
                    'cleaned_data': cleaned_data,
                    'cleaning_stats': cleaning_stats,
                    'columns': df.columns.tolist() if not df.empty else []
                })
            else:
                return render(request, 'myapp/data_cleaning.html', {
                    'error': 'No data found in the database.'
                })
        except Exception as e:
            return render(request, 'myapp/data_cleaning.html', {
                'error': f'Error during data cleaning: {str(e)}'
            })
    
    return render(request, 'myapp/data_cleaning.html')

@login_required
def logout_view(request):
    logout(request)
    return redirect('landing')

@login_required
def claim_prediction(request):
    return render(request, 'myapp/claim_prediction.html')

@login_required
def fraud_detection(request):
    return render(request, 'myapp/fraud_detection.html')

@login_required
def client_management(request):
    return render(request, 'myapp/client_management.html')

@login_required
def reports(request):
    return render(request, 'myapp/reports.html')

@login_required
def exploratory_analysis(request):
    return render(request, 'myapp/exploratory_analysis.html')

@login_required
def model_training(request):
    return render(request, 'myapp/model_training.html')

@login_required
def make_predictions(request):
    return render(request, 'myapp/make_predictions.html')

@login_required
def impact_analysis(request):
    return render(request, 'myapp/impact_analysis.html')

@login_required
def agentic_ai(request):
    return render(request, 'myapp/agentic_ai.html')

@login_required
def temporal_analysis(request):
    return render(request, 'myapp/temporal_analysis.html')

###################
#################
############
#################
###################
############
@login_required
def safaricom_home(request):
    return render(request, 'myapp/safaricom_report.html', {
        'active_tab': 'claim_distribution',
        'visualizations': get_visualizations_data(request)
    })

@login_required
def claim_distribution(request):
    return render(request, 'myapp/safaricom_report.html', {
        'active_tab': 'claim_distribution',
        'visualizations': get_visualizations_data(request)
    })

@login_required
def temporal_analysis(request):
    return render(request, 'myapp/safaricom_report.html', {
        'active_tab': 'temporal_analysis',
        'visualizations': get_visualizations_data(request)
    })

@login_required
def safaricom_home(request):
    return render(request, 'myapp/safaricom_report.html', {
        'active_tab': 'claim_distribution',
        'visualizations': get_visualizations_data(request)
    })

@login_required
def claim_distribution(request):
    return render(request, 'myapp/safaricom_report.html', {
        'active_tab': 'claim_distribution',
        'visualizations': get_visualizations_data(request)
    })

@login_required
def temporal_analysis(request):
    return render(request, 'myapp/safaricom_report.html', {
        'active_tab': 'temporal_analysis',
        'visualizations': get_visualizations_data(request)
    })

@login_required
def provider_efficiency(request):
    return render(request, 'myapp/safaricom_report.html', {
        'active_tab': 'provider_efficiency',
        'visualizations': get_visualizations_data(request)
    })

@login_required
def diagnosis_patterns(request):
    return render(request, 'myapp/safaricom_report.html', {
        'active_tab': 'diagnosis_patterns',
        'visualizations': get_visualizations_data(request)
    })

def advanced_analysis(request):
    return render(request, 'myapp/safaricom_report.html', {
        'active_tab': 'claim_distribution',  # Default tab
        'visualizations': get_visualizations_data(request)
    })
    
    
    
    ###
    ###
    ###
    ###
    ###
    ###
@login_required
def get_visualizations_data(request):
    try:
        # Get the data - make sure this returns a DataFrame, not a JsonResponse
        df = get_claim_data(request)
        
        if df is None or df.empty:
            return {
                'summary_stats': {
                    'total_claims': 0,
                    'total_amount': 0,
                    'avg_claim': 0,
                    'unique_members': 0,
                    'unique_providers': 0,
                    'claims_per_member': 0
                },
                'benefit_types': [],
                'providers': [],
                'cost_centers': [],
                # Add empty visualizations for all expected fields
            }
        
        # Generate summary stats
        summary_stats = {
            'total_claims': len(df),
            'total_amount': float(df['amount'].sum()) if 'amount' in df.columns else 0,
            'avg_claim': float(df['amount'].mean()) if 'amount' in df.columns else 0,
            'unique_members': df['member_id'].nunique() if 'member_id' in df.columns else 0,
            'unique_providers': df['provider_id'].nunique() if 'provider_id' in df.columns else 0,
            'claims_per_member': len(df) / df['member_id'].nunique() if 'member_id' in df.columns and df['member_id'].nunique() > 0 else 0
        }
        
        # Generate visualizations
        visualizations = {
            'summary_stats': summary_stats,
            'benefit_types': df['benefit_type'].unique().tolist() if 'benefit_type' in df.columns else [],
            'providers': df['provider_id'].unique().tolist() if 'provider_id' in df.columns else [],
            'cost_centers': df['cost_center'].unique().tolist() if 'cost_center' in df.columns else [],
            'cost_percentiles': generate_cost_percentiles(df),
            'member_segmentation': generate_member_segmentation(df),
            # Add all other visualizations here
        }
        
        return visualizations
        
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        return {
            'summary_stats': {
                'total_claims': 0,
                'total_amount': 0,
                'avg_claim': 0,
                'unique_members': 0,
                'unique_providers': 0,
                'claims_per_member': 0
            },
            'error': str(e)
        }

@login_required
def generate_cost_percentiles(df):
    # Example visualization generation
    fig = px.box(df, y='amount', title='Cost Distribution by Percentile')
    return plot(fig, output_type='div')

@login_required
def generate_member_segmentation(df):
    # Example visualization generation
    member_spending = df.groupby('member_id')['amount'].sum().reset_index()
    fig = px.histogram(member_spending, x='amount', nbins=20, title='Member Spending Segments')
    return plot(fig, output_type='div')

# ... include all other visualization generation functions

@login_required
def get_claim_data(request):
    # Implement your data fetching logic here
    # This could be from a database, API, or other source
    # Return a pandas DataFrame
    pass



@login_required
def safaricom_reports(request):
    return render(request, 'myapp/safaricom_report.html')



@login_required
def get_cleaned_data():
    """Get cleaned data for EDA analysis"""
    try:
        records = ClaimRecord.objects.all()
        df = pd.DataFrame(list(records.values()))
        
        if df.empty:
            return None
            
        # Apply the same cleaning logic as in clean_data_ajax
        # 1. Drop columns with 80% or more missing values
        missing_threshold = 0.8
        columns_to_drop = []
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct >= missing_threshold:
                columns_to_drop.append(col)
        
        df = df.drop(columns=columns_to_drop)
        
        # 2. Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna('Unknown')
        
        # 3. Clean date formats
        date_columns = ['claim_prov_date', 'dob']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].dt.strftime('%Y-%m-%d').fillna('')
        
        # 4. Clean numeric fields
        if 'amount' in df.columns:
            df['amount'] = df['amount'].astype(str).str.replace(',', '').str.replace('$', '')
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        
        if 'quantity' in df.columns:
            df['quantity'] = df['quantity'].astype(str).str.replace(',', '')
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
        
        # 5. Clean text fields
        text_columns = ['benefit_desc', 'prov_name', 'pol_name', 'cost_center', 'ailment']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
                df[col] = df[col].replace('nan', '')
        
        # 6. Standardize categorical values
        if 'gender' in df.columns:
            df['gender'] = df['gender'].astype(str).str.upper().str.strip()
            df['gender'] = df['gender'].replace(['M', 'MALE'], 'Male')
            df['gender'] = df['gender'].replace(['F', 'FEMALE'], 'Female')
            df['gender'] = df['gender'].replace(['', 'NAN', 'NONE', 'nan'], 'Unknown')
        
        if 'benefit' in df.columns:
            df['benefit'] = df['benefit'].astype(str).str.upper().str.strip()
            df['benefit'] = df['benefit'].replace('nan', '')
        
        # 7. Remove duplicates
        df = df.drop_duplicates()
        
        # 8. Fill remaining NaN values
        df = df.fillna('')
        
        return df
    except Exception as e:
        print(f"Error getting cleaned data: {str(e)}")
        return None

@login_required
def claims_overview_ajax(request):
    """Generate claims overview visualizations"""
    if request.method == 'POST':
        try:
            print("Starting claims overview analysis...")
            df = get_cleaned_data()
            if df is None or df.empty:
                print("No data available for claims overview")
                return JsonResponse({'success': False, 'error': 'No data available'})
            
            print(f"Claims overview data shape: {df.shape}")
            print(f"Available columns: {df.columns.tolist()}")
            
            # Basic statistics
            total_claims = len(df)
            if 'amount' in df.columns:
                # Convert decimal types to float for calculations
                df['amount'] = df['amount'].astype(float)
                total_amount = df['amount'].sum()
                avg_amount = df['amount'].mean()
                print(f"Amount statistics - Total: {total_amount}, Avg: {avg_amount}")
            else:
                total_amount = 0
                avg_amount = 0
                print("Amount column not found")
            
            # Claims by benefit type
            if 'benefit' in df.columns:
                benefit_stats = df.groupby('benefit').agg({
                    'id': 'count',
                    'amount': 'sum'
                }).reset_index()
                benefit_stats.columns = ['benefit_type', 'claim_count', 'total_amount']
                # Convert to regular Python types for JSON serialization
                benefit_stats = benefit_stats.to_dict('records')
                for stat in benefit_stats:
                    stat['total_amount'] = float(stat['total_amount'])
                print(f"Benefit stats generated: {len(benefit_stats)} categories")
            else:
                benefit_stats = []
                print("Benefit column not found")
            
            # Top claimants
            if 'pol_name' in df.columns and 'amount' in df.columns:
                top_claimants = df.groupby('pol_name').agg({
                    'id': 'count',
                    'amount': 'sum'
                }).reset_index()
                top_claimants.columns = ['policy_holder', 'claim_count', 'total_amount']
                top_claimants = top_claimants.sort_values('total_amount', ascending=False).head(10)
                # Convert to regular Python types for JSON serialization
                top_claimants = top_claimants.to_dict('records')
                for claimant in top_claimants:
                    claimant['total_amount'] = float(claimant['total_amount'])
                print(f"Top claimants generated: {len(top_claimants)} claimants")
            else:
                top_claimants = []
                print("Policy name or amount column not found for top claimants")
            
            response_data = {
                'success': True,
                'total_claims': total_claims,
                'total_amount': total_amount,
                'avg_amount': avg_amount,
                'benefit_stats': benefit_stats,
                'top_claimants': top_claimants
            }
            
            print("Claims overview analysis completed successfully")
            return JsonResponse(response_data)
            
        except Exception as e:
            print(f"Error in claims overview: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def fraud_detection_ajax(request):
    """Generate fraud detection visualizations"""
    if request.method == 'POST':
        try:
            print("Starting fraud detection analysis...")
            df = get_cleaned_data()
            if df is None or df.empty:
                print("No data available for fraud detection")
                return JsonResponse({'success': False, 'error': 'No data available'})
            
            print(f"Fraud detection data shape: {df.shape}")
            
            # Simple fraud detection based on amount outliers
            if 'amount' in df.columns:
                # Convert decimal types to float for calculations
                df['amount'] = df['amount'].astype(float)
                
                Q1 = df['amount'].quantile(0.25)
                Q3 = df['amount'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                print(f"Outlier bounds - Lower: {lower_bound}, Upper: {upper_bound}")
                
                # Mark outliers as potential fraud
                df['is_outlier'] = (df['amount'] < lower_bound) | (df['amount'] > upper_bound)
                df['fraud_risk'] = df['is_outlier'].astype(int)
                
                # Create visualizations
                charts = {}
                
                # 1. Fraud Risk Distribution Pie Chart
                fraud_counts = df['fraud_risk'].value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Normal Claims', 'Suspicious Claims'],
                    values=[fraud_counts.get(0, 0), fraud_counts.get(1, 0)],
                    hole=0.5,
                    marker_colors=['#2E8B57', '#DC143C'],
                    textinfo='percent+label',
                    textfont=dict(size=14, color='white'),
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )])
                fig_pie.update_layout(
                    title=dict(
                        text='Fraud Risk Distribution',
                        x=0.5,
                        font=dict(size=18, color='#222b45')
                    ),
                    height=350,
                    width=400,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=20, r=20, t=60, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                charts['fraud_pie'] = plotly.utils.PlotlyJSONEncoder().encode(fig_pie)
                print("Generated fraud pie chart")
                
                # 2. Amount Distribution Histogram
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=df[df['fraud_risk'] == 0]['amount'],
                    name='Normal Claims',
                    nbinsx=30,
                    marker_color='#2E8B57',
                    opacity=0.8,
                    hovertemplate='<b>Normal Claims</b><br>Amount: %{x}<br>Count: %{y}<extra></extra>'
                ))
                fig_hist.add_trace(go.Histogram(
                    x=df[df['fraud_risk'] == 1]['amount'],
                    name='Suspicious Claims',
                    nbinsx=30,
                    marker_color='#DC143C',
                    opacity=0.8,
                    hovertemplate='<b>Suspicious Claims</b><br>Amount: %{x}<br>Count: %{y}<extra></extra>'
                ))
                fig_hist.update_layout(
                    title=dict(
                        text='Claim Amount Distribution by Risk Level',
                        x=0.5,
                        font=dict(size=18, color='#222b45')
                    ),
                    xaxis_title='Amount (KES)',
                    yaxis_title='Frequency',
                    height=350,
                    width=500,
                    barmode='overlay',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=60, r=20, t=60, b=60),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#e3e8ee', zeroline=False),
                    yaxis=dict(gridcolor='#e3e8ee', zeroline=False)
                )
                charts['amount_hist'] = plotly.utils.PlotlyJSONEncoder().encode(fig_hist)
                print("Generated amount histogram")
                
                # 3. Top Suspicious Claims Bar Chart
                suspicious_claims = df[df['is_outlier']].nlargest(8, 'amount')[
                    ['id', 'amount', 'pol_name', 'benefit_desc']
                ]
                fig_suspicious = go.Figure(data=[go.Bar(
                    x=suspicious_claims['amount'],
                    y=[f"Claim {row['id']}" for _, row in suspicious_claims.iterrows()],
                    orientation='h',
                    marker_color='#DC143C',
                    text=suspicious_claims['amount'].apply(lambda x: f'KES {x:,.0f}'),
                    textposition='auto',
                    hovertemplate='<b>Claim %{y}</b><br>Amount: %{x:,.0f} KES<extra></extra>'
                )])
                fig_suspicious.update_layout(
                    title=dict(
                        text='Top 8 Suspicious Claims by Amount',
                        x=0.5,
                        font=dict(size=18, color='#222b45')
                    ),
                    xaxis_title='Amount (KES)',
                    yaxis_title='Claim ID',
                    height=350,
                    width=500,
                    margin=dict(l=80, r=20, t=60, b=60),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#e3e8ee', zeroline=False),
                    yaxis=dict(gridcolor='#e3e8ee', zeroline=False)
                )
                charts['suspicious_bar'] = plotly.utils.PlotlyJSONEncoder().encode(fig_suspicious)
                print("Generated suspicious claims bar chart")
                
                # 4. Provider Risk Analysis Bar Chart
                if 'prov_name' in df.columns:
                    provider_risk = df.groupby('prov_name').agg({
                        'fraud_risk': 'mean',
                        'amount': 'sum',
                        'id': 'count'
                    }).reset_index()
                    provider_risk.columns = ['provider', 'fraud_risk_avg', 'total_amount', 'claim_count']
                    top_suspicious_providers = provider_risk.nlargest(8, 'fraud_risk_avg')
                    
                    fig_providers = go.Figure(data=[go.Bar(
                        x=top_suspicious_providers['fraud_risk_avg'] * 100,
                        y=top_suspicious_providers['provider'],
                        orientation='h',
                        marker_color='#FF6B6B',
                        text=[f"{val:.1f}%" for val in top_suspicious_providers['fraud_risk_avg'] * 100],
                        textposition='auto',
                        hovertemplate='<b>%{y}</b><br>Risk: %{x:.1f}%<extra></extra>'
                    )])
                    fig_providers.update_layout(
                        title=dict(
                            text='Top 8 Suspicious Providers by Fraud Risk',
                            x=0.5,
                            font=dict(size=18, color='#222b45')
                        ),
                        xaxis_title='Average Fraud Risk (%)',
                        yaxis_title='Provider',
                        height=350,
                        width=500,
                        margin=dict(l=80, r=20, t=60, b=60),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='#e3e8ee', zeroline=False),
                        yaxis=dict(gridcolor='#e3e8ee', zeroline=False)
                    )
                    charts['providers_bar'] = plotly.utils.PlotlyJSONEncoder().encode(fig_providers)
                    print("Generated providers risk bar chart")
                
                # 5. Benefit Type Fraud Risk Analysis
                if 'benefit' in df.columns:
                    benefit_risk = df.groupby('benefit').agg({
                        'fraud_risk': 'mean',
                        'amount': 'sum',
                        'id': 'count'
                    }).reset_index()
                    benefit_risk.columns = ['benefit_type', 'fraud_risk_avg', 'total_amount', 'claim_count']
                    benefit_risk = benefit_risk.sort_values('fraud_risk_avg', ascending=False).head(8)
                    
                    fig_benefit = go.Figure(data=[go.Bar(
                        x=benefit_risk['benefit_type'],
                        y=benefit_risk['fraud_risk_avg'] * 100,
                        marker_color='#FF8C00',
                        text=[f"{val:.1f}%" for val in benefit_risk['fraud_risk_avg'] * 100],
                        textposition='auto',
                        hovertemplate='<b>%{x}</b><br>Risk: %{y:.1f}%<extra></extra>'
                    )])
                    fig_benefit.update_layout(
                        title=dict(
                            text='Fraud Risk by Benefit Type',
                            x=0.5,
                            font=dict(size=18, color='#222b45')
                        ),
                        xaxis_title='Benefit Type',
                        yaxis_title='Average Fraud Risk (%)',
                        height=350,
                        width=500,
                        xaxis_tickangle=-45,
                        margin=dict(l=60, r=20, t=60, b=80),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='#e3e8ee', zeroline=False),
                        yaxis=dict(gridcolor='#e3e8ee', zeroline=False)
                    )
                    charts['benefit_risk_bar'] = plotly.utils.PlotlyJSONEncoder().encode(fig_benefit)
                    print("Generated benefit risk bar chart")
                
                # 6. Amount vs Risk Scatter Plot
                sample_df = df.sample(min(800, len(df)))  # Sample for performance
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=sample_df[sample_df['fraud_risk'] == 0]['amount'],
                    y=sample_df[sample_df['fraud_risk'] == 0].index,
                    mode='markers',
                    name='Normal Claims',
                    marker=dict(color='#2E8B57', size=6, opacity=0.6),
                    hovertemplate='<b>Normal Claim</b><br>Amount: %{x:,.0f} KES<br>Index: %{y}<extra></extra>'
                ))
                fig_scatter.add_trace(go.Scatter(
                    x=sample_df[sample_df['fraud_risk'] == 1]['amount'],
                    y=sample_df[sample_df['fraud_risk'] == 1].index,
                    mode='markers',
                    name='Suspicious Claims',
                    marker=dict(color='#DC143C', size=8, opacity=0.8),
                    hovertemplate='<b>Suspicious Claim</b><br>Amount: %{x:,.0f} KES<br>Index: %{y}<extra></extra>'
                ))
                fig_scatter.update_layout(
                    title=dict(
                        text='Claim Amount vs Risk Level (Sample)',
                        x=0.5,
                        font=dict(size=18, color='#222b45')
                    ),
                    xaxis_title='Amount (KES)',
                    yaxis_title='Claim Index',
                    height=350,
                    width=500,
                    margin=dict(l=60, r=20, t=60, b=60),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#e3e8ee', zeroline=False),
                    yaxis=dict(gridcolor='#e3e8ee', zeroline=False),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                charts['amount_scatter'] = plotly.utils.PlotlyJSONEncoder().encode(fig_scatter)
                print("Generated amount scatter plot")
                
                # 7. Monthly Fraud Trend (if date available)
                if 'claim_prov_date' in df.columns:
                    df['claim_prov_date'] = pd.to_datetime(df['claim_prov_date'], errors='coerce')
                    df = df.dropna(subset=['claim_prov_date'])
                    
                    monthly_fraud = df.groupby(df['claim_prov_date'].dt.to_period('M')).agg({
                        'fraud_risk': 'mean',
                        'id': 'count'
                    }).reset_index()
                    monthly_fraud['month'] = monthly_fraud['claim_prov_date'].astype(str)
                    
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=monthly_fraud['month'],
                        y=monthly_fraud['fraud_risk'] * 100,
                        mode='lines+markers',
                        name='Fraud Risk %',
                        line=dict(color='#DC143C', width=3),
                        marker=dict(size=8, color='#DC143C'),
                        hovertemplate='<b>%{x}</b><br>Fraud Risk: %{y:.1f}%<extra></extra>'
                    ))
                    fig_trend.update_layout(
                        title=dict(
                            text='Monthly Fraud Risk Trend',
                            x=0.5,
                            font=dict(size=18, color='#222b45')
                        ),
                        xaxis_title='Month',
                        yaxis_title='Average Fraud Risk (%)',
                        height=350,
                        width=500,
                        margin=dict(l=60, r=20, t=60, b=60),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='#e3e8ee', zeroline=False),
                        yaxis=dict(gridcolor='#e3e8ee', zeroline=False)
                    )
                    charts['fraud_trend'] = plotly.utils.PlotlyJSONEncoder().encode(fig_trend)
                    print("Generated fraud trend chart")
                
                # 8. Risk Level Distribution by Gender (if available)
                if 'gender' in df.columns:
                    gender_risk = df.groupby('gender').agg({
                        'fraud_risk': 'mean',
                        'amount': 'sum',
                        'id': 'count'
                    }).reset_index()
                    gender_risk.columns = ['gender', 'fraud_risk_avg', 'total_amount', 'claim_count']
                    
                    fig_gender = go.Figure(data=[go.Bar(
                        x=gender_risk['gender'],
                        y=gender_risk['fraud_risk_avg'] * 100,
                        marker_color=['#FF69B4', '#4169E1', '#808080'],
                        text=[f"{val:.1f}%" for val in gender_risk['fraud_risk_avg'] * 100],
                        textposition='auto',
                        hovertemplate='<b>%{x}</b><br>Risk: %{y:.1f}%<extra></extra>'
                    )])
                    fig_gender.update_layout(
                        title=dict(
                            text='Fraud Risk by Gender',
                            x=0.5,
                            font=dict(size=18, color='#222b45')
                        ),
                        xaxis_title='Gender',
                        yaxis_title='Average Fraud Risk (%)',
                        height=350,
                        width=400,
                        margin=dict(l=60, r=20, t=60, b=60),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='#e3e8ee', zeroline=False),
                        yaxis=dict(gridcolor='#e3e8ee', zeroline=False)
                    )
                    charts['gender_risk_bar'] = plotly.utils.PlotlyJSONEncoder().encode(fig_gender)
                    print("Generated gender risk bar chart")
                
                # Convert data for tables
                suspicious_claims = suspicious_claims.to_dict('records')
                for claim in suspicious_claims:
                    claim['amount'] = float(claim['amount'])
                
                top_suspicious_providers = top_suspicious_providers.to_dict('records')
                for provider in top_suspicious_providers:
                    provider['total_amount'] = float(provider['total_amount'])
                    provider['fraud_risk_avg'] = float(provider['fraud_risk_avg'])
                    provider['claim_count'] = int(provider['claim_count'])  # Convert int64 to int
                
                response_data = {
                    'success': True,
                    'charts': charts,
                    'suspicious_claims': suspicious_claims,
                    'top_suspicious_providers': top_suspicious_providers,
                    'outlier_thresholds': {
                        'lower': float(lower_bound),
                        'upper': float(upper_bound)
                    },
                    'fraud_stats': {
                        'total_claims': int(len(df)),  # Convert int64 to int
                        'suspicious_claims': int(fraud_counts.get(1, 0)),  # Convert int64 to int
                        'normal_claims': int(fraud_counts.get(0, 0)),  # Convert int64 to int
                        'fraud_percentage': float((fraud_counts.get(1, 0) / len(df)) * 100)  # Convert to float
                    }
                }
                
                print("Fraud detection analysis completed successfully")
                return JsonResponse(response_data)
            else:
                print("Amount column not available for fraud detection")
                return JsonResponse({'success': False, 'error': 'Amount column not available'})
                
        except Exception as e:
            print(f"Error in fraud detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def exploratory_analysis_ajax(request):
    """Main EDA endpoint that returns all visualizations"""
    if request.method == 'POST':
        try:
            print("Starting comprehensive EDA analysis...")
            
            # Get claims overview data
            claims_response = claims_overview_ajax(request)
            claims_data = json.loads(claims_response.content)
            
            # Get fraud detection data
            fraud_response = fraud_detection_ajax(request)
            fraud_data = json.loads(fraud_response.content)
            
            if claims_data['success'] and fraud_data['success']:
                response_data = {
                    'success': True,
                    'claims_overview': claims_data,
                    'fraud_detection': fraud_data
                }
                print("Comprehensive EDA analysis completed successfully")
                return JsonResponse(response_data)
            else:
                error_msg = f"Claims success: {claims_data.get('success', False)}, Fraud success: {fraud_data.get('success', False)}"
                print(f"Failed to generate visualizations: {error_msg}")
                return JsonResponse({
                    'success': False,
                    'error': 'Failed to generate visualizations'
                })
                
        except Exception as e:
            print(f"Error in exploratory analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def safaricom_dashboard_ajax(request):
    """Generate visualizations for the Safaricom dashboard."""
    if request.method == 'POST':
        try:
            df = get_cleaned_data()
            if df is None or df.empty:
                return JsonResponse({'success': False, 'error': 'No data available for analysis.'})
 
            # --- Data Preparation ---
            if 'amount' not in df.columns:
                return JsonResponse({'success': False, 'error': "'amount' column not found."})
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
 
            if 'claim_prov_date' not in df.columns:
                return JsonResponse({'success': False, 'error': "'claim_prov_date' column not found."})
            df['claim_prov_date'] = pd.to_datetime(df['claim_prov_date'], errors='coerce')
            df.dropna(subset=['claim_prov_date'], inplace=True)

            # --- Date Range Filter ---
            start_date_str = request.POST.get('start_date')
            end_date_str = request.POST.get('end_date')

            if start_date_str:
                start_date = pd.to_datetime(start_date_str, errors='coerce')
                if pd.notna(start_date):
                    df = df[df['claim_prov_date'] >= start_date]

            if end_date_str:
                end_date = pd.to_datetime(end_date_str, errors='coerce')
                if pd.notna(end_date):
                    # Add a day to the end date to make the range inclusive
                    df = df[df['claim_prov_date'] < end_date + pd.Timedelta(days=1)]

            if df.empty:
                return JsonResponse({'success': False, 'error': 'No data available for the selected date range.'})

            # --- 1. Summary Metrics ---
            summary_metrics = {
                'total_claims': int(len(df)),
                'total_amount': f"KES {float(df['amount'].sum()):,.2f}",
                'unique_claimants': int(df['pol_name'].nunique()) if 'pol_name' in df.columns else 0
            }
 
            charts = {}
            tables = {}
 
            # --- 2. Total Claims Submitted (Trends) ---
            period = request.POST.get('period', 'monthly').lower()
            period_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'M'}
            resample_code = period_map.get(period, 'M')
            title_period = period.capitalize()
 
            df_trend = df.set_index('claim_prov_date')
            trend_data = df_trend.resample(resample_code).agg(total_claims=('id', 'count'), total_amount=('amount', 'sum')).reset_index()
            
            fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
            fig_trend.add_trace(go.Scatter(x=trend_data['claim_prov_date'], y=trend_data['total_amount'], name='Total Amount', mode='lines+markers', line=dict(color='#e30613')), secondary_y=False)
            fig_trend.add_trace(go.Bar(x=trend_data['claim_prov_date'], y=trend_data['total_claims'], name='Claim Count', marker_color='#ff9800', opacity=0.6), secondary_y=True)
            fig_trend.update_layout(
                title_text=f"{title_period} Claims Trend", 
                xaxis_title="Date", 
                height=350, 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#222b45')
            )
            fig_trend.update_yaxes(title_text="Total Amount (KES)", secondary_y=False, gridcolor='#e3e8ee')
            fig_trend.update_yaxes(title_text="Number of Claims", secondary_y=True, gridcolor='#e3e8ee')
            charts['claims_trend'] = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig_trend))
 
            # --- 3. Claim Amounts by Category (Sunburst Chart) ---
            if 'benefit' in df.columns:
                category_claims = df.groupby('benefit')['amount'].sum().reset_index()
                category_claims = category_claims[category_claims['amount'] > 0]
                
                fig_category = px.sunburst(
                    category_claims,
                    path=['benefit'],
                    values='amount',
                    title="Claim Amounts by Category",
                    color='amount',
                    color_continuous_scale=px.colors.sequential.RdBu,
                    height=400
                )
                fig_category.update_traces(textinfo="label+percent entry")
                fig_category.update_layout(
                    margin=dict(t=40, l=0, r=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                charts['category_distribution'] = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig_category))
 
            # --- 4. Top Claimants (Ranked Table & Bar Chart) ---
            if 'pol_name' in df.columns:
                top_claimants_df = df.groupby('pol_name').agg(
                    total_amount=('amount', 'sum'), 
                    claim_count=('id', 'count')
                ).sort_values('total_amount', ascending=False).head(10).reset_index()
                
                # Format for table display
                top_claimants_table = top_claimants_df.copy()
                top_claimants_table['total_amount'] = top_claimants_table['total_amount'].apply(lambda x: f"KES {x:,.2f}")
                tables['top_claimants'] = top_claimants_table.to_dict('records')
 
                # Chart
                fig_top_claimants = px.bar(
                    top_claimants_df.sort_values('total_amount', ascending=True), 
                    y='pol_name', 
                    x='total_amount', 
                    orientation='h', 
                    title="Top 10 Claimants by Total Amount", 
                    labels={'pol_name': 'Claimant', 'total_amount': 'Total Amount (KES)'}, 
                    color_discrete_sequence=['#2E8B57'],
                    text='total_amount'
                )
                fig_top_claimants.update_traces(texttemplate='KES %{text:,.0f}', textposition='outside')
                fig_top_claimants.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#222b45'),
                    xaxis=dict(gridcolor='#e3e8ee'),
                    yaxis=dict(showticklabels=True)
                )
                charts['top_claimants'] = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig_top_claimants))
 
            # --- 5. Claim Frequency Distribution (Histogram) ---
            if 'pol_name' in df.columns:
                claim_frequency = df.groupby('pol_name')['id'].count()
                fig_freq_dist = px.histogram(
                    claim_frequency, 
                    x=claim_frequency.values, 
                    title="Claim Frequency Distribution per Member", 
                    labels={'x': 'Number of Claims per Member', 'count': 'Number of Members'}, 
                    nbins=20, 
                    color_discrete_sequence=['#ff9800']
                )
                fig_freq_dist.update_layout(
                    height=350,
                    yaxis_title="Number of Members",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#222b45'),
                    xaxis=dict(gridcolor='#e3e8ee'),
                    yaxis=dict(gridcolor='#e3e8ee')
                )
                charts['claim_frequency'] = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig_freq_dist))
 
            return JsonResponse({
                'success': True,
                'summary_metrics': summary_metrics,
                'charts': charts,
                'tables': tables
            })
 
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'error': f'An error occurred: {str(e)}'})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def advanced_analysis_ajax(request):
    """AJAX endpoint for advanced analysis: data profile, claim distribution, temporal, provider, diagnosis."""
    import plotly.graph_objects as go
    from plotly.utils import PlotlyJSONEncoder
    if request.method == 'POST':
        try:
            df = get_cleaned_data()
            if df is None or df.empty:
                return JsonResponse({'success': False, 'error': 'No data available'})

            # --- 1. Automated Data Profile ---
            profile = {}
            desc = df.describe(include='all').transpose().reset_index()
            desc = desc.fillna('')
            profile['summary'] = desc.to_dict('records')
            profile['columns'] = list(df.columns)
            profile['missing'] = df.isnull().sum().to_dict()
            profile['unique'] = df.nunique().to_dict()

            # --- 2. Claim Distribution ---
            groupby = request.POST.get('groupby', 'treatment')
            metric = request.POST.get('metric', 'amount')
            aggfunc = request.POST.get('aggfunc', 'sum')
            filter_col = request.POST.get('filter_col', None)
            filter_val = request.POST.get('filter_val', None)
            claim_dist = {}
            dfg = df.copy()
            if filter_col and filter_val and filter_col in dfg.columns:
                dfg = dfg[dfg[filter_col] == filter_val]
            if groupby in dfg.columns and metric in dfg.columns:
                if aggfunc == 'sum':
                    grouped = dfg.groupby(groupby)[metric].sum().sort_values(ascending=False)
                elif aggfunc == 'count':
                    grouped = dfg.groupby(groupby)[metric].count().sort_values(ascending=False)
                elif aggfunc == 'mean':
                    grouped = dfg.groupby(groupby)[metric].mean().sort_values(ascending=False)
                else:
                    grouped = dfg.groupby(groupby)[metric].sum().sort_values(ascending=False)
                claim_dist['table'] = grouped.reset_index().to_dict('records')
                fig = go.Figure([go.Bar(x=grouped.index.astype(str), y=grouped.values, marker_color='#e30613')])
                fig.update_layout(title=f'Claim Distribution by {groupby}', xaxis_title=groupby.title(), yaxis_title=aggfunc.title() + ' of ' + metric.title(), height=350)
                claim_dist['chart'] = PlotlyJSONEncoder().encode(fig)
            else:
                claim_dist['table'] = []
                claim_dist['chart'] = None

            # --- 3. Temporal Analysis ---
            temporal = {}
            if 'claim_prov_date' in df.columns:
                dft = df.copy()
                dft['claim_prov_date'] = pd.to_datetime(dft['claim_prov_date'], errors='coerce')
                dft = dft.dropna(subset=['claim_prov_date'])
                dft['month'] = dft['claim_prov_date'].dt.to_period('M').astype(str)
                temp_metric = request.POST.get('temporal_metric', 'amount')
                if temp_metric not in dft.columns:
                    temp_metric = 'amount'
                temp_agg = request.POST.get('temporal_agg', 'sum')
                if temp_agg == 'sum':
                    grouped = dft.groupby('month')[temp_metric].sum()
                elif temp_agg == 'count':
                    grouped = dft.groupby('month')[temp_metric].count()
                elif temp_agg == 'mean':
                    grouped = dft.groupby('month')[temp_metric].mean()
                else:
                    grouped = dft.groupby('month')[temp_metric].sum()
                temporal['table'] = grouped.reset_index().to_dict('records')
                fig = go.Figure([go.Scatter(x=grouped.index, y=grouped.values, mode='lines+markers', line=dict(color='#e30613'))])
                fig.update_layout(title='Claims Over Time', xaxis_title='Month', yaxis_title=temp_agg.title() + ' of ' + temp_metric.title(), height=350)
                temporal['chart'] = PlotlyJSONEncoder().encode(fig)
            else:
                temporal['table'] = []
                temporal['chart'] = None

            # --- 4. Provider Efficiency ---
            provider = {}
            if 'prov_name' in df.columns and 'amount' in df.columns:
                prov = df.groupby('prov_name').agg({'amount': ['sum', 'count', 'mean']})
                prov.columns = ['total_amount', 'claim_count', 'avg_amount']
                prov = prov.sort_values('total_amount', ascending=False).head(20)
                provider['table'] = prov.reset_index().to_dict('records')
                fig = go.Figure([go.Bar(x=prov.index.astype(str), y=prov['total_amount'], marker_color='#2E8B57')])
                fig.update_layout(title='Provider Total Amounts', xaxis_title='Provider', yaxis_title='Total Amount', height=350)
                provider['chart'] = PlotlyJSONEncoder().encode(fig)
            else:
                provider['table'] = []
                provider['chart'] = None

            # --- 5. Diagnosis Patterns ---
            diagnosis = {}
            if 'ailment' in df.columns:
                diag = df['ailment'].value_counts().head(20)
                diagnosis['table'] = diag.reset_index().rename(columns={'index': 'ailment', 'ailment': 'count'}).to_dict('records')
                fig = go.Figure([go.Bar(x=diag.index.astype(str), y=diag.values, marker_color='#ff9800')])
                fig.update_layout(title='Top Diagnosis Patterns', xaxis_title='Ailment', yaxis_title='Count', height=350)
                diagnosis['chart'] = PlotlyJSONEncoder().encode(fig)
            else:
                diagnosis['table'] = []
                diagnosis['chart'] = None

            return JsonResponse({
                'success': True,
                'profile': profile,
                'claim_distribution': claim_dist,
                'temporal': temporal,
                'provider': provider,
                'diagnosis': diagnosis
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid request method'})


#### 
#####
####
###   Safaricom page 
@login_required
def safaricom_home(request):
    username = request.user.username
    context = {
        'username': username,
        'visualizations': {
            'summary_stats': {
                'total_claims': 0,
                'total_amount': 0.0,
                'avg_claim': 0.0,
                'unique_members': 0,
            },
            'debug': {
                'cleaned_records': 0,
                'min_date': None,
                'max_date': None,
                'error': None,
                'raw_count': 0,
                'date_format_issues': 0,
                'date_samples': [],
                'used_fallback_dates': False
            }
        }
    }

    try:
        # Load all records
        claims = ClaimRecord.objects.order_by('-claim_prov_date').values(
            'amount', 'claim_prov_date', 'benefit', 'benefit_desc', 'claim_me'
        )

        raw_count = ClaimRecord.objects.count()
        context['visualizations']['debug']['raw_count'] = raw_count
        context['visualizations']['summary_stats']['total_claims'] = raw_count

        if raw_count == 0:
            context['visualizations']['debug']['error'] = "No records found in database"
            return render(request, 'myapp/safaricom_home.html', context)

        # Convert to DataFrame
        df = pd.DataFrame.from_records(claims)
        context['visualizations']['debug']['date_samples'] = df['claim_prov_date'].head(3).tolist()

        # Clean amount
        df['amount'] = pd.to_numeric(
            df['amount'].astype(str)
            .str.replace(r'[^\d.]', '', regex=True)
            .replace('', '0'),
            errors='coerce'
        )

        # Parse and clean dates
        if df['claim_prov_date'].isna().all():
            start_date = pd.to_datetime('2023-01-01')
            end_date = pd.to_datetime(timezone.now().date())
            df['datetime'] = pd.to_datetime(np.random.randint(
                start_date.value // 10**9,
                end_date.value // 10**9,
                size=len(df)
            ), unit='s')
            context['visualizations']['debug']['used_fallback_dates'] = True
        else:
            df['datetime'] = pd.to_datetime(
                df['claim_prov_date'],
                errors='coerce',
                format='mixed',
                dayfirst=True
            )
            if df['datetime'].isna().any():
                valid_count = df['datetime'].notna().sum()
                start_date = pd.to_datetime('2023-01-01')
                end_date = pd.to_datetime(timezone.now().date())
                random_dates = pd.to_datetime(np.random.randint(
                    start_date.value // 10**9,
                    end_date.value // 10**9,
                    size=len(df) - valid_count
                ), unit='s')
                df.loc[df['datetime'].isna(), 'datetime'] = random_dates
                context['visualizations']['debug']['used_fallback_dates'] = True

        # Drop invalid
        df = df.dropna(subset=['datetime', 'amount'])
        context['visualizations']['debug']['cleaned_records'] = len(df)
        df = df.set_index('datetime').sort_index()

        # Date range
        min_date, max_date = df.index.min(), df.index.max()
        context['visualizations']['debug'].update({
            'min_date': min_date,
            'max_date': max_date,
            'date_format_issues': raw_count - len(df)
        })

        # Summary stats
        total_amount = ClaimRecord.objects.aggregate(total=Sum('amount'))['total'] or 0
        unique_members = ClaimRecord.objects.values('claim_me').distinct().count()
        avg_claim = total_amount / unique_members if unique_members > 0 else 0
        context['visualizations']['summary_stats'].update({
            'total_amount': float(total_amount),
            'avg_claim': float(avg_claim),
            'unique_members': unique_members
        })

        # ===== SINGLE INTERACTIVE CHART =====
        if not df.empty:
            daily_df = df.resample('D').size().reset_index(name='count')
            weekly_df = df.resample('W-MON').size().reset_index(name='count')
            monthly_df = df.resample('M').size().reset_index(name='count')

            fig = go.Figure()

            # Daily trace
            fig.add_trace(go.Scatter(
                x=daily_df['datetime'], y=daily_df['count'],
                mode='lines+markers', name='Daily Claims', visible=True,
                line=dict(color='blue')
            ))

            # Weekly trace
            fig.add_trace(go.Scatter(
                x=weekly_df['datetime'], y=weekly_df['count'],
                mode='lines+markers', name='Weekly Claims', visible=False,
                line=dict(color='orange')
            ))

            # Monthly trace
            fig.add_trace(go.Scatter(
                x=monthly_df['datetime'], y=monthly_df['count'],
                mode='lines+markers', name='Monthly Claims', visible=False,
                line=dict(color='green')
            ))

            # Dropdown for switching
            fig.update_layout(
                title="Claims Submitted Over Time",
                xaxis_title="Date",
                yaxis_title="Number of Claims",
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                updatemenus=[dict(
                    type="dropdown",
                    direction="down",
                    x=1.15, y=1.2,
                    showactive=True,
                    buttons=list([
                        dict(label="Daily",
                             method="update",
                             args=[{"visible": [True, False, False]},
                                   {"title": "Daily Claims Submitted"}]),
                        dict(label="Weekly",
                             method="update",
                             args=[{"visible": [False, True, False]},
                                   {"title": "Weekly Claims Submitted"}]),
                        dict(label="Monthly",
                             method="update",
                             args=[{"visible": [False, False, True]},
                                   {"title": "Monthly Claims Submitted"}]),
                    ]),
                )]
            )

            # Remove mini range slider and quick buttons
            fig.update_xaxes(
                rangeselector=None,
                rangeslider=dict(visible=False),
                type="date"
            )

            context['visualizations']['claims_time_chart'] = fig.to_html(full_html=False)

        # ===== OTHER CHARTS REMAIN UNCHANGED =====
        if 'benefit' in df.columns:
            benefit_amount = df.groupby('benefit')['amount'].sum().reset_index()
            if not benefit_amount.empty:
                benefit_fig = px.bar(
                    benefit_amount.sort_values('amount', ascending=False).head(10),
                    x='benefit', y='amount',
                    title='Top Benefit Categories by Amount',
                    labels={'benefit': 'Category', 'amount': 'Total Amount (KES)'},
                    color_discrete_sequence=['#1BB64F']
                )
                benefit_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20),
                                          paper_bgcolor='rgba(0,0,0,0)',
                                          plot_bgcolor='rgba(0,0,0,0)')
                context['visualizations']['category_amounts'] = benefit_fig.to_html(full_html=False)

            if 'benefit_desc' in df.columns and not benefit_amount.empty:
                top_benefits = benefit_amount.nlargest(10, 'amount')['benefit'].tolist()
                sun_df = df[df['benefit'].isin(top_benefits)]
                if not sun_df.empty:
                    sun_fig = px.sunburst(
                        sun_df.reset_index(),
                        path=['benefit', 'benefit_desc'],
                        values='amount',
                        title='Claims Breakdown by Benefit',
                        color_discrete_sequence=px.colors.sequential.Tealgrn
                    )
                    sun_fig.update_layout(margin=dict(t=50, l=20, r=20, b=20), height=500)
                    context['visualizations']['sunburst'] = sun_fig.to_html(full_html=False)

        if 'claim_me' in df.columns:
            top_claimants = df.groupby('claim_me').agg(
                total_amount=('amount', 'sum'),
                claim_count=('amount', 'count')
            ).nlargest(10, 'total_amount').reset_index()

            if not top_claimants.empty:
                claimants_fig = px.bar(
                    top_claimants, x='claim_me', y='total_amount',
                    title='Top Claimants by Amount',
                    labels={'claim_me': 'Member ID', 'total_amount': 'Total Amount (KES)'},
                    color_discrete_sequence=['#1BB64F']
                )
                claimants_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20),
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)')
                context['visualizations']['top_claimants'] = claimants_fig.to_html(full_html=False)
                context['visualizations']['top_claimants_table'] = top_claimants.to_dict('records')

                freq = df['claim_me'].value_counts().reset_index()
                freq.columns = ['claim_me', 'count']
                freq_fig = px.histogram(
                    freq, x='count',
                    title='Claim Frequency Distribution',
                    labels={'count': 'Number of Claims'},
                    color_discrete_sequence=['#1BB64F']
                )
                freq_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20),
                                       paper_bgcolor='rgba(0,0,0,0)',
                                       plot_bgcolor='rgba(0,0,0,0)')
                context['visualizations']['claim_freq'] = freq_fig.to_html(full_html=False)

        return render(request, 'myapp/safaricom_home.html', context)

    except Exception as e:
        import traceback
        traceback.print_exc()
        context['visualizations']['debug']['error'] = f"An error occurred: {str(e)}"
        return render(request, 'myapp/safaricom_home.html', context)
    

    
    
    
@login_required  
def create_visualizations(claims_df):
    visualizations = {}
    
    # Ensure we have a valid datetime column
    if 'claim_prov_date' not in claims_df.columns or not pd.api.types.is_datetime64_any_dtype(claims_df['claim_prov_date']):
        visualizations['error'] = "Valid date column not available for time series analysis"
        return visualizations

    try:
        # Set datetime as index for resampling
        claims_df = claims_df.set_index('claim_prov_date').sort_index()
        
        # 1. Time Series Charts - with proper error handling
        try:
            # Daily claims
            daily_claims = claims_df.resample('D').size().reset_index(name='count')
            fig_daily = px.line(daily_claims, x='claim_prov_date', y='count',
                              title='Daily Claims Submitted', 
                              labels={'count': 'Number of Claims'})
            visualizations['daily_claims'] = plotly_plot(fig_daily, output_type='div')

            # Weekly claims
            weekly_claims = claims_df.resample('W').size().reset_index(name='count')
            fig_weekly = px.area(weekly_claims, x='claim_prov_date', y='count',
                               title='Weekly Claims Submitted', 
                               labels={'count': 'Number of Claims'})
            visualizations['weekly_claims'] = plotly_plot(fig_weekly, output_type='div')

            # Monthly claims
            monthly_claims = claims_df.resample('M').size().reset_index(name='count')
            fig_monthly = px.line(monthly_claims, x='claim_prov_date', y='count',
                                title='Monthly Claims Submitted', 
                                labels={'count': 'Number of Claims'})
            visualizations['monthly_claims'] = plotly_plot(fig_monthly, output_type='div')
        except Exception as e:
            print(f"Error generating time series charts: {e}")
            visualizations['time_series_error'] = f"Could not generate time series charts: {str(e)}"

        # 2. Category Analysis
        if 'benefit' in claims_df.columns and 'amount' in claims_df.columns:
            try:
                category_amounts = claims_df.reset_index().groupby('benefit')['amount'].sum().reset_index()
                fig_category = px.bar(category_amounts, x='benefit', y='amount',
                                    title='Claim Amounts by Benefit Category',
                                    labels={'amount': 'Total Amount (KES)', 'benefit': 'Benefit Category'})
                visualizations['category_amounts'] = plotly_plot(fig_category, output_type='div')

                if 'benefit_desc' in claims_df.columns:
                    fig_sunburst = px.sunburst(
                        claims_df.reset_index(), 
                        path=['benefit', 'benefit_desc'], 
                        values='amount',
                        title='Claim Amounts by Benefit Category (Sunburst)'
                    )
                    visualizations['sunburst'] = plotly_plot(fig_sunburst, output_type='div')
            except Exception as e:
                print(f"Error generating category charts: {e}")

        # 3. Top Claimants
        if 'claim_me' in claims_df.columns and 'amount' in claims_df.columns:
            try:
                top_claimants = claims_df.reset_index().groupby('claim_me').agg(
                    total_amount=('amount', 'sum'),
                    claim_count=('amount', 'count')
                ).nlargest(10, 'total_amount').reset_index()

                fig_top_claimants = px.bar(
                    top_claimants, 
                    x='claim_me', 
                    y='total_amount',
                    title='Top 10 Claimants by Total Amount',
                    labels={'total_amount': 'Total Claim Amount (KES)', 'claim_me': 'Member ID'}
                )
                visualizations['top_claimants'] = plotly_plot(fig_top_claimants, output_type='div')
                visualizations['top_claimants_table'] = top_claimants.to_dict('records')
            except Exception as e:
                print(f"Error generating top claimants chart: {e}")

        # 4. Claim Frequency Distribution
        if 'claim_me' in claims_df.columns:
            try:
                claim_freq = claims_df.reset_index()['claim_me'].value_counts().reset_index()
                claim_freq.columns = ['claim_me', 'count']

                fig_freq = px.histogram(
                    claim_freq, 
                    x='count',
                    title='Claim Frequency Distribution',
                    labels={'count': 'Number of Claims per Member'}
                )
                visualizations['claim_freq'] = plotly_plot(fig_freq, output_type='div')
            except Exception as e:
                print(f"Error generating frequency chart: {e}")

        # Summary statistics
        summary_stats = {
            'total_claims': len(claims_df),
            'total_amount': claims_df['amount'].sum() if 'amount' in claims_df.columns else 0,
            'avg_claim': claims_df['amount'].mean() if 'amount' in claims_df.columns else 0,
            'unique_members': claims_df['claim_me'].nunique() if 'claim_me' in claims_df.columns else 0,
        }
        visualizations['summary_stats'] = summary_stats

    except Exception as e:
        print(f"Error in visualization generation: {e}")
        visualizations['error'] = f"Visualization error: {str(e)}"

    return visualizations


@login_required
def get_claim_data(request):
    # Get all claims data
    claims = ClaimRecord.objects.all()
    
    # Convert to DataFrame for analysis
    claims_df = pd.DataFrame.from_records(claims.values())
    
    # Clean amount column
    if 'amount' in claims_df.columns:
        claims_df['amount'] = claims_df['amount'].astype(str).str.replace(',', '').astype(float)
    
    # Return as JSON
    return JsonResponse(claims_df.to_dict('records'), safe=False)


################
###############

##Safaricom advanced analysis page

@login_required
def advanced_analysis(request):
    claims = ClaimRecord.objects.all()

    # === FILTERING ===
    time_period = request.GET.get('time_period', 'all')
    benefit_type = request.GET.get('benefit_type', 'all')
    provider = request.GET.get('provider', 'all')
    cost_center = request.GET.get('cost_center', 'all')

    if benefit_type != 'all':
        claims = claims.filter(benefit=benefit_type)
    if provider != 'all':
        claims = claims.filter(prov_name=provider)
    if cost_center != 'all':
        claims = claims.filter(cost_center=cost_center)
    if time_period != 'all':
        today = datetime.today().date()
        if time_period == '3m':
            claims = claims.filter(claim_prov_date__gte=today - timedelta(days=90))
        elif time_period == '6m':
            claims = claims.filter(claim_prov_date__gte=today - timedelta(days=180))
        elif time_period == '12m':
            claims = claims.filter(claim_prov_date__gte=today - timedelta(days=365))

    # === GET UNIQUE VALUES FOR DROPDOWNS ===
    benefit_types = ClaimRecord.objects.values_list('benefit', flat=True).distinct().order_by('benefit').exclude(benefit__isnull=True).exclude(benefit__exact='')
    providers = ClaimRecord.objects.values_list('prov_name', flat=True).distinct().order_by('prov_name').exclude(prov_name__isnull=True).exclude(prov_name__exact='')
    cost_centers = ClaimRecord.objects.values_list('cost_center', flat=True).distinct().order_by('cost_center').exclude(cost_center__isnull=True).exclude(cost_center__exact='')

    # === 1. Summary Statistics ===
    summary_stats = {
        'total_claims': claims.count(),
        'total_amount': claims.aggregate(Sum('amount'))['amount__sum'] or 0,
        'avg_claim': claims.aggregate(Avg('amount'))['amount__avg'] or 0,
        'unique_members': claims.values('claim_me').distinct().count(),
        'unique_providers': claims.values('claim_pr').distinct().count(),
        'claims_per_member': claims.count() / max(1, claims.values('claim_me').distinct().count()),
    }

    # === 2. Temporal Patterns ===
    try:
        hourly_claims = claims.annotate(
            claim_date_dt=Cast('claim_prov_date', output_field=DateTimeField())
        ).annotate(
            hour=ExtractHour('claim_date_dt'),
            day_part=Case(
                When(hour__gte=6, hour__lt=12, then=Value('Morning')),
                When(hour__gte=12, hour__lt=17, then=Value('Afternoon')),
                When(hour__gte=17, hour__lt=21, then=Value('Evening')),
                default=Value('Night'),
                output_field=CharField()
            )
        ).values('day_part').annotate(
            count=Count('admit_id'),
            amount=Sum('amount')
        ).order_by('day_part')
    except Exception:
        hourly_claims = []

    # === 3. Cost Distribution Percentiles ===
    amounts = list(claims.values_list('amount', flat=True))
    amounts = [float(a) for a in amounts if a is not None]
    percentiles = []
    if amounts:
        percentiles = [
            {'percentile': '50th', 'amount': np.percentile(amounts, 50)},
            {'percentile': '75th', 'amount': np.percentile(amounts, 75)},
            {'percentile': '90th', 'amount': np.percentile(amounts, 90)},
            {'percentile': '95th', 'amount': np.percentile(amounts, 95)},
            {'percentile': '99th', 'amount': np.percentile(amounts, 99)},
        ]

    # === 4. Member Segmentation ===
    member_stats = claims.values('claim_me').annotate(
        claim_count=Count('admit_id'),
        total_amount=Sum('amount'),
        avg_amount=Avg('amount')
    ).order_by('-total_amount')

    # === 5. Provider Network Analysis ===
    provider_stats = claims.values('prov_name').annotate(
        claim_count=Count('admit_id'),
        total_amount=Sum('amount'),
        member_count=Count('claim_me', distinct=True)
    ).order_by('-total_amount')[:20]

    # === 6. Age-Service Matrix ===
    try:
        claims_with_age = claims.annotate(
            age=ExtractYear(datetime.now().date()) - ExtractYear('dob')
        ).filter(age__gte=0, age__lte=100)
    except Exception as e:
        print("Error generating age data:", e)
        claims_with_age = ClaimRecord.objects.none()

    # === 7. Day of Week Analysis ===
    day_of_week_claims = claims.annotate(
        day_of_week=ExtractWeekDay('claim_prov_date')
    ).annotate(
        day_name=Case(
            When(day_of_week=1, then=Value('Sunday')),
            When(day_of_week=2, then=Value('Monday')),
            When(day_of_week=3, then=Value('Tuesday')),
            When(day_of_week=4, then=Value('Wednesday')),
            When(day_of_week=5, then=Value('Thursday')),
            When(day_of_week=6, then=Value('Friday')),
            When(day_of_week=7, then=Value('Saturday')),
            output_field=CharField()
        )
    ).values('day_name').annotate(
        claim_count=Count('admit_id'),
        total_amount=Sum('amount')
    ).order_by('day_of_week')

    # === 8. Additional Analyses ===
    # Gender Analysis
    gender_stats = claims.values('gender').annotate(
        claim_count=Count('admit_id'),
        total_amount=Sum('amount'),
        avg_amount=Avg('amount')
    ).order_by('-total_amount')

    # Monthly Trend
    monthly_trend = claims.annotate(
        month=ExtractMonth('claim_prov_date')
    ).values('month').annotate(
        claim_count=Count('admit_id'),
        total_amount=Sum('amount')
    ).order_by('month')

    # Top Ailments
    top_ailments = claims.values('ailment').annotate(
        claim_count=Count('admit_id'),
        total_amount=Sum('amount')
    ).order_by('-claim_count')[:10]

    # Cost Center Analysis
    cost_center_analysis = claims.values('cost_center').annotate(
        claim_count=Count('admit_id'),
        total_amount=Sum('amount')
    ).order_by('-total_amount')[:10]

    # === Chart Visualizations ===
    visualizations = {
        'summary_stats': summary_stats,
        'hourly_claims': generate_plotly_chart(hourly_claims, 'bar', 'day_part', 'amount', 'Claims by Time of Day'),
        'cost_percentiles': generate_plotly_chart(percentiles, 'line', 'percentile', 'amount', 'Cost Distribution by Percentile'),
        'member_segmentation': generate_member_segmentation_chart(member_stats),
        'provider_network': generate_provider_network_chart(provider_stats),
        'age_service_matrix': generate_age_service_matrix(claims_with_age),
        'gender_stats': generate_plotly_chart(gender_stats, 'pie', 'gender', 'total_amount', 'Claims by Gender'),
        'monthly_trend': generate_plotly_chart(monthly_trend, 'line', 'month', 'total_amount', 'Monthly Claims Trend'),
        'top_ailments': generate_plotly_chart(top_ailments, 'bar', 'ailment', 'claim_count', 'Top 10 Ailments by Claim Count'),
        'cost_center_analysis': generate_plotly_chart(cost_center_analysis, 'bar', 'cost_center', 'total_amount', 'Top 10 Cost Centers by Total Amount'),
        'day_of_week_analysis': generate_plotly_chart(day_of_week_claims, 'bar', 'day_name', 'total_amount', 'Claims by Day of Week'),
        'correlation_matrix': generate_correlation_matrix(claims_with_age),
        'amount_distribution': generate_amount_distribution(claims),
        'provider_efficiency': generate_provider_efficiency(claims),
        'time_series': generate_time_series(claims),
        'benefit_types': benefit_types,
        'providers': providers,
        'cost_centers': cost_centers,
    }

    return render(request, 'safaricom_report.html', {
        'visualizations': visualizations,
        'username': request.user.username
    })

@login_required
def generate_plotly_chart(data, chart_type, x, y, title):
    df = pd.DataFrame(data)
    if df.empty or x not in df.columns or y not in df.columns:
        return f"<div class='no-data'><strong>No data available for: {title}</strong></div>"

    if chart_type == 'bar':
        fig = px.bar(df, x=x, y=y, title=title)
    elif chart_type == 'line':
        fig = px.line(df, x=x, y=y, title=title)
    elif chart_type == 'pie':
        fig = px.pie(df, names=x, values=y, title=title)
    else:
        fig = px.scatter(df, x=x, y=y, title=title)

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return plot(fig, output_type='div', include_plotlyjs=False)

@login_required
def generate_member_segmentation_chart(data):
    df = pd.DataFrame(list(data))
    if not df.empty and 'total_amount' in df.columns:
        df['total_amount'] = df['total_amount'].astype(float)
        df['segment'] = pd.qcut(df['total_amount'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        fig = px.box(df, x='segment', y='total_amount', title='Member Segmentation by Spending')
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return plot(fig, output_type='div', include_plotlyjs=False)
    return "<div class='no-data'><strong>No data available for: Member Segmentation</strong></div>"

@login_required
def generate_provider_network_chart(data):
    df = pd.DataFrame(list(data))
    if not df.empty:
        fig = px.scatter(df, x='member_count', y='total_amount', size='claim_count',
                         hover_name='prov_name', title='Provider Network Analysis',
                         labels={'member_count': 'Unique Members', 'total_amount': 'Total Amount', 'claim_count': 'Claim Count'})
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return plot(fig, output_type='div', include_plotlyjs=False)
    return "<div class='no-data'><strong>No data available for: Provider Network</strong></div>"

@login_required
def generate_age_service_matrix(data):
    try:
        if isinstance(data, list):
            return "<div class='no-data'><strong>No data available for: Age-Service Matrix</strong></div>"
        df = pd.DataFrame(list(data.values('age', 'service_code', 'amount')))
        if not df.empty:
            pivot = df.groupby(['age', 'service_code'])['amount'].sum().unstack().fillna(0)
            fig = px.imshow(pivot, title='Age-Service Heatmap')
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            return plot(fig, output_type='div', include_plotlyjs=False)
    except Exception as e:
        return f"<div class='no-data'><strong>Error generating age-service matrix: {e}</strong></div>"
    return "<div class='no-data'><strong>No data available for: Age-Service Matrix</strong></div>"

@login_required
def generate_correlation_matrix(data):
    try:
        df = pd.DataFrame(list(data.values('amount', 'age', 'quantity')))
        if not df.empty:
            corr = df.corr()
            fig = px.imshow(corr, 
                          text_auto=True,
                          color_continuous_scale='Viridis',
                          title='Claims Data Correlation Matrix')
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            return plot(fig, output_type='div', include_plotlyjs=False)
    except Exception as e:
        return f"<div class='no-data'><strong>Error generating correlation matrix: {e}</strong></div>"
    return "<div class='no-data'><strong>No data available for correlation matrix</strong></div>"

@login_required
def generate_amount_distribution(data):
    try:
        amounts = list(data.values_list('amount', flat=True))
        if amounts:
            fig = px.histogram(x=amounts, nbins=50, 
                             title='Claim Amount Distribution',
                             labels={'x': 'Claim Amount', 'y': 'Count'})
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            return plot(fig, output_type='div', include_plotlyjs=False)
    except Exception as e:
        return f"<div class='no-data'><strong>Error generating distribution: {e}</strong></div>"
    return "<div class='no-data'><strong>No data available for amount distribution</strong></div>"

@login_required
def generate_provider_efficiency(claims):
    df = pd.DataFrame.from_records(claims.values())
    
    if df.empty or 'SERVICE_DESCRIPTION' not in df.columns:
        return {}

    # Group by service provider (replace with actual column for provider if different)
    provider_grouped = df.groupby('SERVICE_DESCRIPTION').agg(
        claim_count=('ADMIT_ID', 'count'),
        total_amount=('AMOUNT', 'sum'),
        avg_amount=('AMOUNT', 'mean')
    ).reset_index()

    provider_grouped.rename(columns={'SERVICE_DESCRIPTION': 'provider_name'}, inplace=True)

    # Ensure numeric values and handle invalid types
    provider_grouped['claim_count'] = pd.to_numeric(provider_grouped['claim_count'], errors='coerce')
    provider_grouped['total_amount'] = pd.to_numeric(provider_grouped['total_amount'], errors='coerce')
    provider_grouped['avg_amount'] = pd.to_numeric(provider_grouped['avg_amount'], errors='coerce')

    provider_grouped = provider_grouped.dropna(subset=['claim_count', 'total_amount', 'avg_amount'])

    # Create scatter plot
    fig = px.scatter(
        provider_grouped,
        x='claim_count',
        y='avg_amount',
        size='total_amount',
        hover_name='provider_name',
        title='Provider Efficiency: Claims vs. Avg Amount',
        labels={
            'claim_count': 'Number of Claims',
            'avg_amount': 'Average Claim Amount',
            'total_amount': 'Total Claim Amount',
            'provider_name': 'Provider'
        },
        size_max=60,
        template='plotly_white'
    )

    return fig.to_html(full_html=False)

@login_required
def generate_time_series(data):
    daily_claims = data.annotate(
        date=Cast('claim_prov_date', output_field=DateField())
    ).values('date').annotate(
        claim_count=Count('admit_id'),
        total_amount=Sum('amount')
    ).order_by('date')
    
    df = pd.DataFrame(list(daily_claims))
    if not df.empty:
        fig = px.line(df, x='date', y='total_amount',
                     title='Daily Claims Amount Over Time',
                     labels={'date': 'Date', 'total_amount': 'Total Amount'})
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return plot(fig, output_type='div', include_plotlyjs=False)
    return "<div class='no-data'><strong>No data available for time series</strong></div>"








####################

#################


##############







##############

import pandas as pd
import plotly.express as px
from django.shortcuts import render
from myapp.models import ClaimRecord
from django.core.cache import cache
from datetime import datetime, timedelta
from django.template.defaultfilters import register

@login_required
def claim_distribution(request):
    # Get filter parameters from request
    time_period = request.GET.get('time_period', 'all')
    benefit_type = request.GET.get('benefit_type', 'all')
    provider = request.GET.get('provider', 'all')
    cost_center = request.GET.get('cost_center', 'all')
    category = request.GET.get('category', 'benefit')
    metric = request.GET.get('metric', 'Sum of Amount')
    filter_col = request.GET.get('filter_col', 'None')
    filter_values = request.GET.getlist('filter_values')
    
    # Generate cache key based on all parameters
    cache_key = f"claim_dist_{request.GET.urlencode()}"
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    # Query data from database
    queryset = ClaimRecord.objects.all()
    
    # Apply time period filter
    if time_period != 'all':
        now = datetime.now()
        if time_period == '3m':
            cutoff_date = now - timedelta(days=90)
        elif time_period == '6m':
            cutoff_date = now - timedelta(days=180)
        elif time_period == '12m':
            cutoff_date = now - timedelta(days=365)
        queryset = queryset.filter(claim_prov_date__gte=cutoff_date)
    
    # Apply other filters
    if benefit_type != 'all':
        queryset = queryset.filter(benefit=benefit_type)
    if provider != 'all':
        queryset = queryset.filter(prov_name=provider)
    if cost_center != 'all':
        queryset = queryset.filter(cost_center=cost_center)
    
    # Convert to DataFrame
    data = pd.DataFrame.from_records(queryset.values())
    
    # Get categorical columns
    cat_cols = [col for col in data.columns if data[col].dtype in ['object', 'category'] or data[col].nunique() < 20]
    
    # Prepare categorical values dictionary
    categorical_values = {}
    for col in cat_cols:
        try:
            unique_vals = data[col].unique().tolist()
            # Convert to string and remove None/NaN values
            unique_vals = [str(x) for x in unique_vals if pd.notna(x)]
            categorical_values[col] = sorted(unique_vals)
        except:
            continue
    
    # Apply additional filters if specified
    if filter_col != 'None' and filter_values:
        data = data[data[filter_col].astype(str).isin(filter_values)]
    
    # Prepare visualizations dictionary
    visualizations = {
        'summary_stats': {
            'total_claims': len(data),
            'total_amount': data['amount'].sum() if 'amount' in data.columns else 0,
            'avg_claim': data['amount'].mean() if 'amount' in data.columns else 0,
            'unique_members': data['claim_me'].nunique() if 'claim_me' in data.columns else 0,
            'unique_providers': data['prov_name'].nunique() if 'prov_name' in data.columns else 0,
            'claims_per_member': len(data)/data['claim_me'].nunique() if 'claim_me' in data.columns and data['claim_me'].nunique() > 0 else 0,
        },
        'benefit_types': sorted(data['benefit'].unique().tolist()) if 'benefit' in data.columns else [],
        'providers': sorted(data['prov_name'].unique().tolist()) if 'prov_name' in data.columns else [],
        'cost_centers': sorted(data['cost_center'].unique().tolist()) if 'cost_center' in data.columns else [],
        'current_category': category,
        'current_metric': metric,
        'current_filter_col': filter_col,
        'current_filter_values': filter_values,
        'categorical_columns': cat_cols,
        'categorical_values': categorical_values,
    }
    
    # Generate the claim distribution plot if data exists
    if not data.empty and cat_cols:
        # Prepare data based on selections
        if metric == 'Count':
            dist_data = data[category].value_counts().reset_index()
            dist_data.columns = [category, 'Count']
            y_metric = 'Count'
        else:
            amount_col = 'amount'
            if amount_col in data.columns:
                data[amount_col] = pd.to_numeric(data[amount_col], errors='coerce')
                
                if metric == 'Sum of Amount':
                    dist_data = data.groupby(category)[amount_col].sum().reset_index()
                    dist_data.columns = [category, 'Total Amount']
                    y_metric = 'Total Amount'
                elif metric == 'Average Amount':
                    dist_data = data.groupby(category)[amount_col].mean().reset_index()
                    dist_data.columns = [category, 'Average Amount']
                    y_metric = 'Average Amount'
            else:
                dist_data = pd.DataFrame()
        
        if not dist_data.empty:
            # Create visualization
            fig = px.bar(
                dist_data,
                x=category,
                y=y_metric,
                title=f"Claim Distribution by {category}",
                hover_data=[y_metric],
                color=category
            )
            
            if 'Amount' in metric and 'amount' in data.columns:
                avg_value = data['amount'].mean() if metric == 'Average Amount' else data['amount'].sum()/len(dist_data)
                fig.add_hline(
                    y=avg_value,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Overall {'Average' if metric == 'Average Amount' else 'Mean per Category'}",
                    annotation_position="top left"
                )
            
            visualizations['claim_distribution'] = fig.to_html(full_html=False)
    
    # Generate other visualizations
    if not data.empty and 'amount' in data.columns:
        try:
            data['amount'] = pd.to_numeric(data['amount'], errors='coerce')
            
            # Cost percentiles
            percentiles = data['amount'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).reset_index()
            percentiles.columns = ['Percentile', 'Value']
            fig_percentiles = px.line(
                percentiles,
                x='Percentile',
                y='Value',
                title='Cost Distribution by Percentile'
            )
            visualizations['cost_percentiles'] = fig_percentiles.to_html(full_html=False)
            
            # Member segmentation
            member_spending = data.groupby('claim_me')['amount'].sum().reset_index()
            member_spending['Segment'] = pd.qcut(member_spending['amount'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
            fig_segments = px.pie(
                member_spending,
                names='Segment',
                title='Member Spending Segments'
            )
            visualizations['member_segmentation'] = fig_segments.to_html(full_html=False)
            
            # Amount distribution
            fig_dist = px.histogram(
                data,
                x='amount',
                nbins=50,
                title='Claim Amount Distribution'
            )
            visualizations['amount_distribution'] = fig_dist.to_html(full_html=False)
            
            # Top cost centers
            if 'cost_center' in data.columns:
                top_centers = data.groupby('cost_center')['amount'].sum().nlargest(10).reset_index()
                fig_centers = px.bar(
                    top_centers,
                    x='cost_center',
                    y='amount',
                    title='Top 10 Cost Centers'
                )
                visualizations['cost_center_analysis'] = fig_centers.to_html(full_html=False)
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    response = render(request, 'myapp/safaricom_report.html', {
        'active_tab': 'claim_distribution',
        'visualizations': visualizations
    })
    
    # Cache the response for 15 minutes
    cache.set(cache_key, response, timeout=60*15)
    return response



######

##### Temporal analysis 

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from django.shortcuts import render
from myapp.models import ClaimRecord
from django.core.cache import cache
from datetime import datetime, timedelta

@login_required
def temporal_analysis(request):
    from django.utils import timezone
    from datetime import timedelta
    import pandas as pd
    import plotly.express as px
    import numpy as np
    
    username = request.user.username
    context = {
        'username': username,
        'time_unit': request.GET.get('time_unit', 'month'),
        'metric': request.GET.get('metric', 'Total Amount'),
        'benefit_types': ClaimRecord.objects.values_list('benefit', flat=True).distinct().order_by('benefit'),
        'providers': ClaimRecord.objects.values_list('prov_name', flat=True).distinct().order_by('prov_name'),
        'cost_centers': ClaimRecord.objects.values_list('cost_center', flat=True).distinct().order_by('cost_center'),
        'error_message': None,
        'temporal_chart': None,
        'numerical_chart': None,
        'categorical_chart': None,
    }

    try:
        # Get filter parameters
        time_period = request.GET.get('time_period', 'all')
        benefit_type = request.GET.get('benefit_type', 'all')
        provider = request.GET.get('provider', 'all')
        cost_center = request.GET.get('cost_center', 'all')

        # Start with base query
        claims = ClaimRecord.objects.all()

        # Apply filters FIRST
        if time_period != 'all':
            today = timezone.now().date()
            if time_period == '3m':
                start_date = today - timedelta(days=90)
            elif time_period == '6m':
                start_date = today - timedelta(days=180)
            elif time_period == '12m':
                start_date = today - timedelta(days=365)
            claims = claims.filter(claim_prov_date__gte=start_date)

        if benefit_type != 'all':
            claims = claims.filter(benefit=benefit_type)
        if provider != 'all':
            claims = claims.filter(prov_name=provider)
        if cost_center != 'all':
            claims = claims.filter(cost_center=cost_center)

        # Only select needed fields and limit AFTER applying filters
        claims = claims.values(
            'id', 'claim_prov_date', 'amount', 'claim_me', 'prov_name', 'benefit'
        )[:10000]  # Limit to 10,000 records for performance

        # Convert to DataFrame
        data = pd.DataFrame.from_records(claims)

        # Check if we have data
        if data.empty:
            context['error_message'] = "No claims found matching your filters"
            return render(request, 'myapp/temporal_analysis.html', context)

        # Handle amount conversion - fixed regex
        data['amount'] = pd.to_numeric(
            data['amount'].astype(str)
            .str.replace(r'[^\d.]', '', regex=True)  # Fixed regex escape
            .replace('', '0'),
            errors='coerce'
        )

        # Create datetime column - handle missing dates
        if data['claim_prov_date'].isna().all():
            # For filtered datasets, use random dates within a reasonable range
            start_date = pd.to_datetime('2023-01-01')
            end_date = pd.to_datetime('2024-01-01')
            data['datetime'] = pd.to_datetime(np.random.randint(
                start_date.value // 10**9,
                end_date.value // 10**9,
                size=len(data)
            ), unit='s')
            context['error_message'] = "Using randomly generated dates as no valid dates were found"
        else:
            # Try to parse existing dates
            data['datetime'] = pd.to_datetime(
                data['claim_prov_date'],
                errors='coerce',
                format='mixed',
                dayfirst=True
            )
            # Fill any remaining NaT with random dates
            if data['datetime'].isna().any():
                valid_count = data['datetime'].notna().sum()
                start_date = pd.to_datetime('2023-01-01')
                end_date = pd.to_datetime('2024-01-01')
                random_dates = pd.to_datetime(np.random.randint(
                    start_date.value // 10**9,
                    end_date.value // 10**9,
                    size=len(data) - valid_count
                ), unit='s')
                data.loc[data['datetime'].isna(), 'datetime'] = random_dates
                context['error_message'] = f"Used random dates for {len(data) - valid_count} records with invalid dates"

        # Set datetime index
        df = data.set_index('datetime').sort_index()

        # Temporal aggregation
        period_map = {
            'day': 'D',
            'week': 'W-MON',
            'month': 'ME',
            'quarter': 'QE',
            'year': 'YE'
        }
        time_unit = request.GET.get('time_unit', 'month')
        resample_period = period_map.get(time_unit, 'ME')

        temporal_data = df.resample(resample_period).agg({
            'amount': ['sum', 'count', 'mean']
        })
        temporal_data.columns = ['total_amount', 'claim_count', 'avg_amount']
        temporal_data = temporal_data.reset_index().rename(columns={'index': 'datetime'})

        # Create visualization
        metric = request.GET.get('metric', 'Total Amount')
        if metric == 'Claim Count':
            fig = px.line(
                temporal_data,
                x='datetime',
                y='claim_count',
                title=f"Claim Count by {time_unit.capitalize()}",
                labels={'claim_count': 'Number of Claims'},
                template='plotly_white'
            )
        elif metric == 'Total Amount':
            fig = px.line(
                temporal_data,
                x='datetime',
                y='total_amount',
                title=f"Total Claim Amount by {time_unit.capitalize()}",
                labels={'total_amount': 'Total Amount (KES)'},
                template='plotly_white'
            )
        else:  # Average Amount
            fig = px.line(
                temporal_data,
                x='datetime',
                y='avg_amount',
                title=f"Average Claim Amount by {time_unit.capitalize()}",
                labels={'avg_amount': 'Average Amount (KES)'},
                template='plotly_white'
            )

        fig.update_traces(line_color='#1BB64F')
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        context['temporal_chart'] = fig.to_html(full_html=False)

        # Quick Insights Charts
        monthly_data = df.resample('ME')['amount'].sum().reset_index().rename(columns={'index': 'datetime'})
        numerical_fig = px.line(
            monthly_data,
            x='datetime',
            y='amount',
            title='Monthly Claim Amount Trend',
            labels={'amount': 'Total Amount (KES)'},
            template='plotly_white'
        )
        numerical_fig.update_traces(line_color='#1BB64F')
        context['numerical_chart'] = numerical_fig.to_html(full_html=False)

        # For categorical chart
        categorical_fig = px.histogram(
            df.reset_index(),
            x='datetime',
            color='benefit',
            title='Claims by Benefit Type Over Time',
            labels={'datetime': 'Date', 'count': 'Number of Claims'},
            template='plotly_white'
        )
        categorical_fig.update_layout(barmode='stack')
        context['categorical_chart'] = categorical_fig.to_html(full_html=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        context['error_message'] = f"An error occurred: {str(e)}"

    return render(request, 'myapp/temporal_analysis.html', context)



#####
####
#### Provider efficiency

@login_required
def provider_efficiency(request):
    username = request.user.username
    context = {
        'username': username,
        'visualizations': {
            'provider_efficiency': None,
            'top_providers': [],
            'bottom_providers': [],
            'debug': {
                'error': None,
                'providers_analyzed': 0
            }
        },
        'fixed_cost': request.GET.get('fixed_cost', 50000),
        'variable_rate': request.GET.get('variable_rate', 60),
        # Initialize these lists to prevent template errors
        'benefit_types': list(ClaimRecord.objects.values_list('benefit', flat=True).distinct()),
        'providers': list(ClaimRecord.objects.values_list('prov_name', flat=True).distinct())
    }

    try:
        # Get filter parameters from request
        time_period = request.GET.get('time_period', 'all')
        benefit_type = request.GET.get('benefit_type', 'all')
        fixed_cost = float(context['fixed_cost'])
        variable_rate = float(context['variable_rate']) / 100
        
        # Query data from database
        queryset = ClaimRecord.objects.all()
        
        # Apply time period filter
        if time_period != 'all':
            now = datetime.now()
            if time_period == '3m':
                cutoff_date = now - timedelta(days=90)
            elif time_period == '6m':
                cutoff_date = now - timedelta(days=180)
            elif time_period == '12m':
                cutoff_date = now - timedelta(days=365)
            queryset = queryset.filter(claim_prov_date__gte=cutoff_date)
        
        # Apply benefit type filter
        if benefit_type != 'all':
            queryset = queryset.filter(benefit=benefit_type)
        
        # Convert to DataFrame
        data = pd.DataFrame.from_records(queryset.values(
            'prov_name', 'amount', 'claim_prov_date', 'benefit'
        ))
        
        if data.empty:
            context['visualizations']['debug']['error'] = "No claims data found matching your filters"
            return render(request, 'myapp/provider_efficiency.html', context)
        
        # Clean and prepare data
        data['amount'] = pd.to_numeric(data['amount'], errors='coerce')
        data = data.dropna(subset=['amount', 'prov_name'])
        
        if data.empty:
            context['visualizations']['debug']['error'] = "No valid provider/amount data found"
            return render(request, 'myapp/provider_efficiency.html', context)
        
        # Calculate provider statistics
        provider_stats = data.groupby('prov_name').agg({
            'amount': ['sum', 'count', 'mean', 'median'],
        }).reset_index()
        
        # Flatten multi-index columns
        provider_stats.columns = [
            'prov_name',
            'total_amount',
            'total_claims',
            'avg_amount',
            'median_amount'
        ]
        
        # Calculate efficiency metrics
        overall_avg = data['amount'].mean()
        provider_stats['efficiency_score'] = 1 / (provider_stats['avg_amount'] / overall_avg)
        
        # Calculate break-even points
        provider_stats['break_even_claims'] = np.ceil(
            fixed_cost / (provider_stats['avg_amount'] * (1 - variable_rate)))
        provider_stats['profitability'] = np.where(
            provider_stats['total_claims'] > provider_stats['break_even_claims'],
            'Profitable',
            'Unprofitable'
        )
        
        # Create visualization
        fig = px.scatter(
            provider_stats,
            x='total_claims',
            y='avg_amount',
            color='profitability',
            size='total_amount',
            hover_name='prov_name',
            hover_data=['break_even_claims'],
            title="Provider Cost Efficiency Analysis",
            labels={
                'total_claims': 'Number of Claims',
                'avg_amount': 'Average Claim Amount (KES)',
                'total_amount': 'Total Amount (KES)'
            }
        )
        
        # Add break-even line
        if fixed_cost > 0:
            max_claims = provider_stats['total_claims'].max()
            if max_claims > 0:
                break_even_line = fixed_cost / (np.linspace(1, max_claims, 100) * (1 - variable_rate))
                fig.add_trace(
                    go.Scatter(
                        x=np.linspace(1, max_claims, 100),
                        y=break_even_line,
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Break-Even Line'
                    )
                )
        
        context['visualizations']['provider_efficiency'] = fig.to_html(full_html=False)
        
        # Prepare top/bottom providers
        provider_stats = provider_stats.sort_values('efficiency_score', ascending=False)
        context['visualizations']['top_providers'] = provider_stats.head(5).to_dict('records')
        context['visualizations']['bottom_providers'] = provider_stats.tail(5).to_dict('records')
        context['visualizations']['debug']['providers_analyzed'] = len(provider_stats)
        
        return render(request, 'myapp/provider_efficiency.html', context)
    
    except Exception as e:
        print(f"Error in provider_efficiency: {str(e)}")
        context['visualizations']['debug']['error'] = str(e)
        return render(request, 'myapp/provider_efficiency.html', context)
    
    
    
############

##########


################



###diagnosis patterns 

@login_required
def diagnosis_patterns(request):
    username = request.user.username
    context = {
        'username': username,
        'summary': {
            'total_claims': 0,
            'total_amount': '0.00',
            'avg_claim': '0.00',
            'unique_members': 0,
            'providers': 0,
            'avg_claims_per_member': '0.0'
        },
        'benefit_types': ClaimRecord.objects.values_list('benefit', flat=True).distinct(),
        'providers': ClaimRecord.objects.values_list('prov_name', flat=True).distinct(),
        'visualizations': {
            'diagnosis_treatment_matrix': None,
            'top_ailments': None,
            'age_service': None,
            'gender_distribution': None
        }
    }

    try:
        # Get filter parameters from request
        time_period = request.GET.get('time_period', 'all')
        benefit_type = request.GET.get('benefit_type', 'all')
        provider = request.GET.get('provider', 'all')
        
        # Query data from database
        queryset = ClaimRecord.objects.all()
        
        # Apply time period filter
        if time_period != 'all':
            now = datetime.now()
            if time_period == '3m':
                cutoff_date = now - timedelta(days=90)
            elif time_period == '6m':
                cutoff_date = now - timedelta(days=180)
            elif time_period == '12m':
                cutoff_date = now - timedelta(days=365)
            queryset = queryset.filter(claim_prov_date__gte=cutoff_date)
        
        # Apply benefit type filter
        if benefit_type != 'all':
            queryset = queryset.filter(benefit=benefit_type)
            
        # Apply provider filter
        if provider != 'all':
            queryset = queryset.filter(prov_name=provider)
        
        # Calculate summary statistics
        context['summary'] = {
            'total_claims': queryset.count(),
            'total_amount': "{:,.2f}".format(queryset.aggregate(Sum('amount'))['amount__sum'] or 0),
            'avg_claim': "{:,.2f}".format(queryset.aggregate(Avg('amount'))['amount__avg'] or 0),
            'unique_members': queryset.values('pol_id').distinct().count(),
            'providers': queryset.values('prov_name').distinct().count(),
            'avg_claims_per_member': "{:.1f}".format(
                queryset.count() / max(1, queryset.values('pol_id').distinct().count())
            )
        }
        
        # Convert to DataFrame
        data = pd.DataFrame.from_records(queryset.values(
            'ailment', 'service_code', 'benefit', 'claim_prov_date', 
            'dob', 'gender', 'amount', 'prov_name'
        ))
        
        if data.empty:
            return render(request, 'myapp/diagnosis_patterns.html', context)
        
        # Clean and prepare data
        data = data.rename(columns={
            'ailment': 'diagnosis',
            'service_code': 'treatment',
            'amount': 'claim_amount'
        })
        
        # Calculate age from DOB
        if 'dob' in data.columns:
            data['age'] = (datetime.now().year - pd.to_datetime(data['dob']).dt.year)
        
        # Create diagnosis-treatment matrix
        if 'diagnosis' in data.columns and 'treatment' in data.columns:
            diag_treat_matrix = pd.crosstab(
                data['diagnosis'],
                data['treatment'],
                normalize='index'
            )
            
            # Filter to top diagnoses and treatments
            top_diag = data['diagnosis'].value_counts().head(20).index
            top_treat = data['treatment'].value_counts().head(20).index
            
            filtered_matrix = diag_treat_matrix.loc[
                top_diag.intersection(diag_treat_matrix.index), 
                top_treat.intersection(diag_treat_matrix.columns)
            ]
            
            # Create heatmap
            fig = px.imshow(
                filtered_matrix,
                labels=dict(x="Treatment", y="Diagnosis", color="Frequency"),
                x=filtered_matrix.columns,
                y=filtered_matrix.index,
                aspect="auto",
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                xaxis_title="Treatment",
                yaxis_title="Diagnosis",
                height=600
            )
            
            context['visualizations']['diagnosis_treatment_matrix'] = fig.to_html(full_html=False)
        
        # Create top ailments chart
        if 'diagnosis' in data.columns:
            top_ailments = data['diagnosis'].value_counts().head(10)
            fig = px.bar(
                top_ailments,
                orientation='h',
                labels={'value': 'Number of Claims', 'index': 'Diagnosis'},
                title='Top 10 Ailments'
            )
            fig.update_layout(height=600)
            context['visualizations']['top_ailments'] = fig.to_html(full_html=False)
        
        # Create age-service relationships
        if 'age' in data.columns and 'treatment' in data.columns:
            # Create age groups
            bins = [0, 18, 35, 50, 65, 120]
            labels = ['0-18', '19-35', '36-50', '51-65', '65+']
            data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels)
            
            # Get top services per age group
            age_service = data.groupby(['age_group', 'treatment']).size().unstack().fillna(0)
            top_services = data['treatment'].value_counts().head(10).index
            age_service = age_service[top_services.intersection(age_service.columns)]
            
            fig = px.imshow(
                age_service,
                labels=dict(x="Service", y="Age Group", color="Claims"),
                aspect="auto",
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=600)
            context['visualizations']['age_service'] = fig.to_html(full_html=False)
        
        # Create gender distribution
        if 'gender' in data.columns:
            gender_dist = data['gender'].value_counts()
            fig = px.pie(
                gender_dist,
                names=gender_dist.index,
                values=gender_dist.values,
                title='Claims by Gender'
            )
            fig.update_layout(height=600)
            context['visualizations']['gender_distribution'] = fig.to_html(full_html=False)
        
        return render(request, 'myapp/diagnosis_patterns.html', context)
    
    except Exception as e:
        print(f"Error in diagnosis_patterns: {str(e)}")
        return render(request, 'myapp/diagnosis_patterns.html', context)
    
    

######################
#####################
##################

##########

#####   claims prediction functionality
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from datetime import timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score, mean_absolute_error
from myapp.models import ClaimRecord


@login_required
def claims_prediction_home(request):
    # === Collect filter values ===
    selected_time_period = request.GET.get('time_period', 'all')
    selected_benefit_type = request.GET.get('benefit_type', 'all')
    selected_provider = request.GET.get('provider', 'all')
    selected_forecast_months = int(request.GET.get('forecast_months', 3))

    # === Base context ===
    context = {
        'username': request.user.username,
        'active_tab': 'claims-prediction',
        'visualizations': {},
        'benefit_types': sorted(ClaimRecord.objects.values_list('benefit', flat=True)
                                 .exclude(benefit__isnull=True)
                                 .exclude(benefit='')
                                 .distinct()),
        'providers': sorted(ClaimRecord.objects.values_list('prov_name', flat=True)
                             .exclude(prov_name__isnull=True)
                             .exclude(prov_name='')
                             .distinct()),
        'selected_time_period': selected_time_period,
        'selected_benefit_type': selected_benefit_type,
        'selected_provider': selected_provider,
        'selected_forecast_months': selected_forecast_months,
        'forecast_next_month': None,
        'forecast_growth': None,
        'forecast_accuracy': None,
        'forecast_r2': None,
        'forecast_mae': None
    }

    try:
        # === FILTER DATA ===
        queryset = ClaimRecord.objects.all()

        if selected_time_period != 'all':
            today = timezone.now().date()
            days_map = {'3m': 90, '6m': 180, '12m': 365}
            days = days_map.get(selected_time_period, 0)
            if days > 0:
                start_date = today - timedelta(days=days)
                queryset = queryset.filter(claim_prov_date__gte=start_date)

        if selected_benefit_type != 'all':
            queryset = queryset.filter(benefit=selected_benefit_type)

        if selected_provider != 'all':
            queryset = queryset.filter(prov_name=selected_provider)

        # === LOAD DATA ===
        df = pd.DataFrame.from_records(queryset.values('claim_prov_date', 'amount'))
        if df.empty:
            context['error'] = "No claims data found for the selected filters."
            return render(request, 'myapp/forecasted_volume.html', context)

        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df['datetime'] = pd.to_datetime(df['claim_prov_date'], errors='coerce')
        df = df.dropna(subset=['datetime'])

        if df.empty:
            context['error'] = "No valid claims data with dates found."
            return render(request, 'myapp/forecasted_volume.html', context)

        # === MONTHLY AGGREGATION ===
        monthly_data = df.groupby(pd.Grouper(key='datetime', freq='M'))['amount'].sum().reset_index()
        monthly_data.rename(columns={'datetime': 'date'}, inplace=True)

        # === DEBUG RAW DATA CHART ===
        debug_fig = px.bar(
            monthly_data, x='date', y='amount',
            title="Raw Monthly Aggregated Data",
            labels={'amount': 'Total Amount (KES)', 'date': 'Month'},
            text='amount'
        )
        debug_fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        context['visualizations']['raw_monthly_data'] = debug_fig.to_html(full_html=False)

        if len(monthly_data) < 3:
            context['error'] = "Insufficient historical data (need at least 3 months) for forecasting."
            return render(request, 'myapp/forecasted_volume.html', context)

        # === FORECASTING ===
        last_date = monthly_data['date'].max()
        forecast_months = selected_forecast_months

        try:
            model = ARIMA(monthly_data['amount'], order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_months)
            forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                           periods=forecast_months, freq='M')
            predictions = model_fit.predict(start=1, end=len(monthly_data))
        except Exception as e:
            context['error'] = f"ARIMA model failed: {str(e)}. Using simple linear regression."
            x = np.arange(len(monthly_data))
            y = monthly_data['amount'].values
            coeffs = np.polyfit(x, y, 1)
            forecast = np.polyval(coeffs, np.arange(len(monthly_data), len(monthly_data) + forecast_months))
            predictions = np.polyval(coeffs, x)
            forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                           periods=forecast_months, freq='M')

        # === METRICS (Safe Calculations) ===
        actuals = np.array(monthly_data['amount'], dtype=float)
        predictions = np.array(predictions, dtype=float)

        if np.all(actuals == actuals[0]):
            r2 = np.nan  # No variance in data
        else:
            r2 = r2_score(actuals, predictions)

        mae = mean_absolute_error(actuals, predictions)

        non_zero_mask = actuals != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask])) * 100
            confidence = max(0, min(100, 100 - mape))
        else:
            mape = np.nan
            confidence = np.nan

        context['forecast_accuracy'] = None if np.isnan(confidence) else round(confidence, 1)
        context['forecast_r2'] = None if np.isnan(r2) else round(r2, 3)
        context['forecast_mae'] = None if np.isnan(mae) else round(mae, 2)

        # === COMBINE HISTORICAL & FORECAST ===
        forecast_df = pd.DataFrame({'date': forecast_dates, 'amount': forecast, 'type': 'Forecast'})
        historical_df = monthly_data.copy()
        historical_df['type'] = 'Historical'
        combined_df = pd.concat([historical_df, forecast_df])

        # === FORECAST CHART ===
        fig = px.line(combined_df, x='date', y='amount', color='type',
                      title='Monthly Claims Volume with Forecast',
                      labels={'amount': 'Total Amount (KES)', 'date': 'Month'},
                      markers=True)

        fig.add_vrect(x0=last_date, x1=forecast_dates[-1], fillcolor="lightgray",
                      opacity=0.2, line_width=0, annotation_text="Forecast Period",
                      annotation_position="top left")

        if 'model_fit' in locals():
            conf_int = model_fit.get_forecast(steps=forecast_months).conf_int()
            fig.add_trace(go.Scatter(x=forecast_dates, y=conf_int.iloc[:, 0], fill=None,
                                     mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast_dates, y=conf_int.iloc[:, 1], fill='tonexty',
                                     mode='lines', line=dict(width=0),
                                     fillcolor='rgba(27, 182, 79, 0.2)',
                                     name='Confidence Interval'))

        context['visualizations']['forecast_volume'] = fig.to_html(full_html=False)
        context['forecast_next_month'] = float(forecast[0]) if len(forecast) > 0 else None
        if len(forecast) > 0 and monthly_data['amount'].iloc[-1] != 0:
            context['forecast_growth'] = round(((forecast[0] - monthly_data['amount'].iloc[-1]) /
                                               monthly_data['amount'].iloc[-1] * 100), 2)

    except Exception as e:
        context['error'] = f"Error processing forecast: {str(e)}"

    return render(request, 'myapp/forecasted_volume.html', context)




from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from myapp.models import ClaimRecord


@login_required
def confidence_intervals(request):
    """View for dynamic confidence intervals from DB"""
    context = {
        'active_tab': 'claims-prediction',
        'visualizations': {}
    }

    try:
        # === Get confidence level from query param ===
        selected_confidence = int(request.GET.get('confidence', 95))
        z_values = {90: 1.645, 95: 1.96, 99: 2.576}
        z_value = z_values.get(selected_confidence, 1.96)
        context['selected_confidence'] = selected_confidence

        # === Load claims data from database ===
        qs = ClaimRecord.objects.values('claim_prov_date', 'amount')
        df = pd.DataFrame.from_records(qs)

        if df.empty:
            context['error'] = "No claims data found."
            return render(request, 'confidence_interval.html', context)

        # Ensure correct data types
        df['date'] = pd.to_datetime(df['claim_prov_date'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['date', 'amount'])

        if df.empty:
            context['error'] = "No valid claims data with dates and amounts."
            return render(request, 'confidence_interval.html', context)

        # === Aggregate by month ===
        monthly_data = df.set_index('date').resample('M').agg({
            'amount': ['sum', 'count', 'mean', 'std']
        }).reset_index()

        # Flatten multi-index columns
        monthly_data.columns = [
            'date',
            'total_amount',
            'claim_count',
            'avg_amount',
            'std_amount'
        ]

        # Handle NaNs
        monthly_data['std_amount'] = monthly_data['std_amount'].fillna(0)

        # === Calculate Confidence Intervals ===
        monthly_data['ci_lower'] = monthly_data['avg_amount'] - z_value * monthly_data['std_amount'] / np.sqrt(monthly_data['claim_count'])
        monthly_data['ci_upper'] = monthly_data['avg_amount'] + z_value * monthly_data['std_amount'] / np.sqrt(monthly_data['claim_count'])

        # === Average CI Range (for stat card) ===
        monthly_data['ci_range'] = monthly_data['ci_upper'] - monthly_data['ci_lower']
        avg_ci_range = monthly_data['ci_range'].mean()
        context['avg_ci_range'] = f"{avg_ci_range:,.0f}"  # formatted KES

        # === Outlier Detection (last 12 months) ===
        last_year = monthly_data[monthly_data['date'] >= (monthly_data['date'].max() - pd.DateOffset(months=12))]
        outliers = []
        for _, row in last_year.iterrows():
            if row['avg_amount'] > row['ci_upper']:
                outliers.append({
                    'date': row['date'].strftime('%b %Y'),
                    'amount': f"+KES {row['avg_amount'] - row['ci_upper']:,.0f} above upper limit"
                })
            elif row['avg_amount'] < row['ci_lower']:
                outliers.append({
                    'date': row['date'].strftime('%b %Y'),
                    'amount': f"-KES {row['ci_lower'] - row['avg_amount']:,.0f} below lower limit"
                })

        context['outliers'] = outliers
        context['outlier_count'] = len(outliers)

        # === Create Plotly Chart ===
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=monthly_data['date'],
            y=monthly_data['total_amount'],
            mode='lines+markers',
            name='Total Claims',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=monthly_data['date'],
            y=monthly_data['ci_upper'],
            mode='lines',
            line=dict(width=0),
            name='Upper CI',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=monthly_data['date'],
            y=monthly_data['ci_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            name=f'{selected_confidence}% Confidence Interval'
        ))

        fig.update_layout(
            title=f'Monthly Claims with {selected_confidence}% Confidence Intervals',
            xaxis_title='Month',
            yaxis_title='Total Claim Amount (KES)',
            template='plotly_white'
        )

        context['visualizations']['confidence_intervals'] = fig.to_html(full_html=False)

    except Exception as e:
        context['error'] = f"Error generating confidence intervals: {str(e)}"

    return render(request, 'confidence_interval.html', context)




@login_required
def impact_simulation(request):
    qs = ClaimRecord.objects.values('amount')
    df = pd.DataFrame.from_records(qs)

    if df.empty:
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'error': 'No claims data found'}, status=400)
        return render(request, 'impact_simulation.html', {'error': 'No claims data found'})

    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').dropna()
    total_amount = df['amount'].sum()

    def simulate(copay_change, deductible_change, utilization_change):
        return total_amount * (1 + utilization_change / 100) \
                             * (1 - copay_change / 200) \
                             * (1 - deductible_change / 300)

    scenarios_list = [
        {'name': 'Current Policy', 'amount': total_amount, 'savings': 0},
        {'name': '10% Copay Increase', 'amount': simulate(10, 0, 0), 'savings': total_amount - simulate(10, 0, 0)},
        {'name': '5% Deductible Increase', 'amount': simulate(0, 5, 0), 'savings': total_amount - simulate(0, 5, 0)},
        {'name': 'Combined Changes', 'amount': simulate(10, 5, 0), 'savings': total_amount - simulate(10, 5, 0)},
    ]

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        copay_change = float(request.GET.get('copay_change', 0))
        deductible_change = float(request.GET.get('deductible_change', 0))
        utilization_change = float(request.GET.get('utilization_change', 0))

        new_total = simulate(copay_change, deductible_change, utilization_change)
        savings = total_amount - new_total
        savings_percent = (savings / total_amount * 100) if total_amount else 0

        return JsonResponse({
            'current_total': f"KES {total_amount:,.2f}",
            'projected_total': f"KES {new_total:,.2f}",
            'savings': f"KES {savings:,.2f}",
            'savings_percent': f"{savings_percent:.1f}%",
            'copay_change': copay_change,
            'deductible_change': deductible_change,
            'utilization_change': utilization_change,
            'scenarios': [
                {'name': s['name'], 'amount': f"KES {s['amount']:,.2f}", 'savings': f"KES {s['savings']:,.2f}"}
                for s in scenarios_list
            ]
        })

    context = {
        'active_tab': 'claims-prediction',
        'metrics': {
            'current_total': f"KES {total_amount:,.2f}",
            'projected_total': f"KES {total_amount:,.2f}",
            'savings_percent': "0%",
            'copay_change': 0,
            'deductible_change': 0,
            'utilization_change': 0,
            'scenarios': [
                {'name': s['name'], 'amount': f"KES {s['amount']:,.2f}", 'savings': f"KES {s['savings']:,.2f}"}
                for s in scenarios_list
            ]
        }
    }
    return render(request, 'impact_simulation.html', context)




import numpy as np
import pandas as pd
import shap
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from myapp.models import ClaimRecord
from django.utils import timezone
from datetime import timedelta

@login_required
def explainability(request):
    import traceback
    import numpy as np
    import pandas as pd
    import shap
    import plotly.express as px
    from statsmodels.tsa.arima.model import ARIMA
    from .models import ClaimRecord

    context = {
        'active_tab': 'claims-prediction',
        'visualizations': {},
        'error': None
    }

    try:
        # Load data
        df = pd.DataFrame.from_records(
            ClaimRecord.objects.values('claim_prov_date', 'amount')
        )
        if df.empty:
            context['error'] = "No claims data found."
            return render(request, 'explainability.html', context)

        # Clean data
        df['date'] = pd.to_datetime(df['claim_prov_date'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df = df.dropna(subset=['date']).sort_values('date')

        # Monthly aggregation
        full_range = pd.date_range(
            df['date'].min().replace(day=1),
            df['date'].max().replace(day=1),
            freq='MS'
        )
        monthly_data = (
            df.groupby(pd.Grouper(key='date', freq='M'))['amount']
              .sum()
              .reindex(full_range, fill_value=0)
              .reset_index()
        )
        monthly_data.columns = ['date', 'amount']

        if len(monthly_data) < 4:
            context['error'] = "Need at least 4 months of data for explainability."
            return render(request, 'explainability.html', context)

        # Lag features
        lagged_df = pd.DataFrame({'y': monthly_data['amount']})
        for lag in range(1, 4):
            lagged_df[f'lag_{lag}'] = monthly_data['amount'].shift(lag)
        lagged_df = lagged_df.dropna()

        X = lagged_df[['lag_1', 'lag_2', 'lag_3']]
        y = lagged_df['y']

        # Fit ARIMA on actual data
        model = ARIMA(y, order=(1, 1, 1))
        model_fit = model.fit()

        # Forecast next month (fixed with .iloc[0])
        forecast_next = model_fit.forecast(steps=1).iloc[0]
        context['forecasted_next_month'] = f"KES {forecast_next:,.2f}"

        # Prediction wrapper for SHAP
        def arima_predict(data_as_array):
            preds = []
            y_list = y.tolist()
            for row in data_as_array:
                try:
                    temp_series = y_list.copy()
                    # Replace last 3 lags with row values
                    for i, val in enumerate(row):
                        temp_series[-(i+1)] = val
                    pred_model = ARIMA(temp_series, order=(1, 1, 1))
                    pred_fit = pred_model.fit()
                    preds.append(pred_fit.forecast(steps=1).iloc[0])
                except Exception:
                    preds.append(np.nan)
            return np.array(preds)

        # SHAP or Fallback
        try:
            explainer = shap.KernelExplainer(arima_predict, X)
            shap_values = explainer.shap_values(X, nsamples=20)

            if shap_values is not None and not np.all(np.isnan(shap_values)):
                # SHAP plot
                shap_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Mean SHAP Value': np.nan_to_num(np.abs(shap_values)).mean(axis=0)
                }).sort_values('Mean SHAP Value', ascending=True)

                shap_fig = px.bar(
                    shap_df,
                    x='Mean SHAP Value',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance for ARIMA (Lags)',
                    color='Mean SHAP Value',
                    color_continuous_scale='Blues'
                )
                context['visualizations']['shap_values'] = shap_fig.to_html(full_html=False)

                # Partial dependence plot for top feature
                top_feature = shap_df.iloc[-1]['Feature']
                x_range = np.linspace(X[top_feature].min(), X[top_feature].max(), 20)
                pd_preds = []
                for val in x_range:
                    temp = X.copy()
                    temp[top_feature] = val
                    pd_preds.append(np.nanmean(arima_predict(temp)))

                pd_fig = px.line(
                    x=x_range,
                    y=pd_preds,
                    title=f'Partial Dependence on {top_feature}',
                    labels={'x': top_feature, 'y': 'Predicted Amount'}
                )
                context['visualizations']['partial_dependence'] = pd_fig.to_html(full_html=False)

            else:
                raise ValueError("SHAP returned empty or NaN values")

        except Exception:
            # Fallback to correlation plot
            corr_df = pd.DataFrame({
                'Feature': X.columns,
                'Correlation with Target': [np.corrcoef(X[col], y)[0, 1] for col in X.columns]
            }).sort_values('Correlation with Target', ascending=True)

            corr_fig = px.bar(
                corr_df,
                x='Correlation with Target',
                y='Feature',
                orientation='h',
                title='Correlation with Claims Amount (Fallback)',
                color='Correlation with Target',
                color_continuous_scale='Blues'
            )
            context['visualizations']['shap_values'] = corr_fig.to_html(full_html=False)

        return render(request, 'explainability.html', context)

    except Exception as e:
        traceback_str = traceback.format_exc()
        context['error'] = f"Explainability error:\n{traceback_str}"
        return render(request, 'explainability.html', context)



#################
##############
#############
########### Fraud detection
import json
import pandas as pd
import numpy as np
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .models import ClaimRecord

# ---- Simple Fraud Detection Logic ----
def detect_fraud_anomalies(df):
    """
    Detect fraud with calibrated scoring to reduce over-flagging.
    """
    # Lower base score
    df['fraud_score'] = np.random.uniform(0.0, 0.2, len(df))

    # Rule 1: High amount (above 95th percentile)
    amount_threshold = df['amount'].quantile(0.95)
    df.loc[df['amount'] > amount_threshold, 'fraud_score'] += 0.3

    # Rule 2: Duplicate provider + icd10_code + amount (only if more than 2 duplicates)
    duplicate_groups = df.groupby(['provider_name', 'icd10_code', 'amount']).size()
    common_duplicates = duplicate_groups[duplicate_groups > 2].reset_index()[['provider_name', 'icd10_code', 'amount']]
    dup_mask = df.merge(common_duplicates, on=['provider_name', 'icd10_code', 'amount'], how='left', indicator=True)['_merge'] == 'both'
    df.loc[dup_mask, 'fraud_score'] += 0.15

    # Rule 3: Suspicious diagnosis codes
    suspicious_codes = ['D01', 'D05', 'D12']
    df.loc[df['icd10_code'].isin(suspicious_codes), 'fraud_score'] += 0.2

    # Cap scores at 1.0
    df['fraud_score'] = df['fraud_score'].clip(0, 1)

    # Flag fraud – higher threshold to reduce false positives
    df['fraud_flag'] = (df['fraud_score'] > 0.6).astype(int)

    return df



@login_required
def fraud_detection_home(request):
    context = {
        'metrics': {},
        'risky_claims': [],
        'risk_distribution_data': json.dumps({'bins': [], 'counts': []}),
        'suspicious_providers': [],
        'diagnosis_patterns': [],
        'monthly_trends': []
    }

    try:
        # Load from DB
        queryset = ClaimRecord.objects.values(
            'claim_prov_date', 'amount', 'prov_name', 'icd10_code'
        )
        df = pd.DataFrame(list(queryset))

        if df.empty:
            context['error'] = "No claim data found."
            return render(request, 'risk_scores.html', context)

        # Data cleaning
        df['date'] = pd.to_datetime(df['claim_prov_date'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df['provider_name'] = df['prov_name'].fillna("Unknown")
        df['icd10_code'] = df['icd10_code'].fillna("Unknown")

        # Run fraud detection
        df = detect_fraud_anomalies(df)

        # Summary metrics
        fraud_count = int(df['fraud_flag'].sum())
        fraud_rate = fraud_count / len(df)
        fraud_amount = df.loc[df['fraud_flag'] == 1, 'amount'].sum()

        context['metrics'] = {
            'fraud_count': fraud_count,
            'fraud_rate': f"{fraud_rate:.1%}",
            'fraud_amount': f"KES {fraud_amount:,.2f}"
        }

        # Risky claims (top 20)
        context['risky_claims'] = df[df['fraud_flag'] == 1] \
            .sort_values('fraud_score', ascending=False) \
            .head(20).to_dict('records')

        # Suspicious providers
        provider_fraud = df.groupby('provider_name').agg(
            total_amount=('amount', 'sum'),
            fraud_count=('fraud_flag', 'sum'),
            total_claims=('fraud_flag', 'count')
        ).reset_index()
        provider_fraud['fraud_rate'] = provider_fraud['fraud_count'] / provider_fraud['total_claims']
        context['suspicious_providers'] = provider_fraud.sort_values('fraud_count', ascending=False).head(10).to_dict('records')

        # Diagnosis patterns
        diagnosis_fraud = df.groupby('icd10_code').agg(
            total_amount=('amount', 'sum'),
            fraud_count=('fraud_flag', 'sum'),
            total_claims=('fraud_flag', 'count')
        ).reset_index()
        diagnosis_fraud['fraud_rate'] = diagnosis_fraud['fraud_count'] / diagnosis_fraud['total_claims']
        context['diagnosis_patterns'] = diagnosis_fraud[diagnosis_fraud['total_claims'] > 10].sort_values(
            'fraud_rate', ascending=False
        ).head(10).to_dict('records')

        # Monthly trends
        monthly_trends = df.set_index('date').resample('M')['fraud_flag'].sum().reset_index()
        context['monthly_trends'] = monthly_trends.to_dict('records')

        # Risk Score Distribution
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        df['score_bin'] = pd.cut(df['fraud_score'], bins=bins, labels=bin_labels, include_lowest=True)
        bin_counts = df['score_bin'].value_counts().reindex(bin_labels, fill_value=0).tolist()
        context['risk_distribution_data'] = json.dumps({'bins': bin_labels, 'counts': bin_counts})

    except Exception as e:
        context['error'] = f"Error: {str(e)}"

    return render(request, 'risk_scores.html', context)







@login_required
def suspicious_providers(request):
    """Suspicious providers dashboard (DB-driven)"""
    import plotly.express as px

    sort_by = request.GET.get('sort', 'count')  # Default sort by fraud count
    context = {'active_tab': 'fraud-detection', 'visualizations': {}, 'sort_by': sort_by}

    try:
        queryset = ClaimRecord.objects.values(
            'claim_prov_date', 'amount', 'prov_name', 'icd10_code'
        )
        df = pd.DataFrame(list(queryset))

        if df.empty:
            context['error'] = "No claims data found."
            return render(request, 'suspicious_providers.html', context)

        # Prepare data
        df['date'] = pd.to_datetime(df['claim_prov_date'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df['provider_name'] = df['prov_name'].fillna("Unknown")
        df['diagnosis'] = df['icd10_code'].fillna("Unknown")

        # Detect fraud
        df = detect_fraud_anomalies(df)

        # Aggregate provider stats
        provider_fraud = df.groupby('provider_name').agg(
            Total_Amount=('amount', 'sum'),
            Fraud_Count=('fraud_flag', 'sum'),
            Total_Claims=('fraud_flag', 'count')
        ).reset_index()
        provider_fraud['Fraud_Rate'] = provider_fraud['Fraud_Count'] / provider_fraud['Total_Claims']

        # Sort based on filter
        if sort_by == 'count':
            top_fraud = provider_fraud.sort_values('Fraud_Count', ascending=False).head(10)
        elif sort_by == 'rate':
            top_fraud = provider_fraud.sort_values('Fraud_Rate', ascending=False).head(10)
        elif sort_by == 'amount':
            top_fraud = provider_fraud.sort_values('Total_Amount', ascending=False).head(10)
        else:
            top_fraud = provider_fraud.sort_values('Fraud_Count', ascending=False).head(10)

        # Chart
        fig = px.bar(
            top_fraud,
            x='provider_name',
            y='Fraud_Count',
            color='Fraud_Rate',
            title=f"Top Providers by {sort_by.capitalize()}",
            hover_data=['Total_Amount', 'Total_Claims']
        )
        context['visualizations']['providers'] = fig.to_html(full_html=False)

        # Provider data for template
        context['provider_data'] = top_fraud.rename(columns={'provider_name': 'Provider'}).to_dict('records')

        # Data for comparison chart
        context['provider_names'] = list(top_fraud['provider_name'])
        context['provider_rates'] = list((top_fraud['Fraud_Rate'] * 100).round(1))
        context['avg_rate'] = round(provider_fraud['Fraud_Rate'].mean() * 100, 1)

    except Exception as e:
        context['error'] = f"Error: {str(e)}"

    return render(request, 'suspicious_providers.html', context)




@login_required
def diagnosis_patterns(request):
    """View for diagnosis patterns analysis (DB-driven)"""
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from myapp.models import ClaimRecord

    context = {
        'active_tab': 'fraud-detection',
        'visualizations': {},
        'diagnosis_data': None
    }

    try:
        # 1. Get min_claims from query param (default 5)
        min_claims = int(request.GET.get('min_claims', 5))

        # 2. Pull data from DB
        queryset = ClaimRecord.objects.values(
            'icd10_code', 'amount', 'claim_prov_date', 'prov_name'
        )
        df = pd.DataFrame(list(queryset))

        if df.empty:
            context['error'] = "No claims data available."
            return render(request, 'diagnosis_patterns1.html', context)

        # 3. Add simulated fraud flag (replace with real detection later)
        df['fraud_flag'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])

        # 4. Clean column names
        df.rename(columns={'icd10_code': 'Diagnosis'}, inplace=True)

        # 5. Group & aggregate
        diagnosis_fraud = df.groupby('Diagnosis').agg({
            'amount': 'sum',
            'fraud_flag': ['sum', 'count']
        }).reset_index()

        diagnosis_fraud.columns = ['Diagnosis', 'Total Amount', 'Fraud Count', 'Total Claims']
        diagnosis_fraud['Fraud Rate'] = diagnosis_fraud['Fraud Count'] / diagnosis_fraud['Total Claims']

        # 6. Filter by min claims
        high_fraud_diag = diagnosis_fraud[diagnosis_fraud['Total Claims'] >= min_claims] \
            .sort_values('Fraud Rate', ascending=False).head(10)

        if not high_fraud_diag.empty:
            fig = px.bar(
                high_fraud_diag,
                x='Diagnosis',
                y='Fraud Rate',
                color='Total Amount',
                title="Diagnoses with Highest Fraud Rates",
                hover_data=['Total Claims', 'Fraud Count']
            )
            context['visualizations']['diagnosis'] = fig.to_html(full_html=False)
            context['diagnosis_data'] = high_fraud_diag.to_dict('records')

    except Exception as e:
        context['error'] = str(e)

    return render(request, 'diagnosis_patterns1.html', context)




from .views import detect_fraud_anomalies  # reuse your existing fraud detection

@login_required
def monthly_trends(request):
    """Monthly fraud trends from DB using real fraud detection"""
    context = {
        'active_tab': 'fraud-detection',
        'visualizations': {},
        'trend_data': None
    }

    try:
        # 1. Pull claims from DB
        queryset = ClaimRecord.objects.values(
            'claim_prov_date', 'amount', 'prov_name', 'icd10_code'
        )
        df = pd.DataFrame(list(queryset))

        if df.empty:
            context['error'] = "No claims data available."
            return render(request, 'monthly_trends.html', context)

        # 2. Data cleaning
        df['date'] = pd.to_datetime(df['claim_prov_date'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df['provider_name'] = df['prov_name'].fillna("Unknown")
        df['icd10_code'] = df['icd10_code'].fillna("Unknown")

        # 3. Fraud detection
        df = detect_fraud_anomalies(df)

        # 4. Monthly aggregation
        fraud_over_time = (
            df.set_index('date')
              .resample('M')['fraud_flag']
              .sum()
              .reset_index()
        )

        # 5. Plotly line chart
        fig = px.line(
            fraud_over_time,
            x='date',
            y='fraud_flag',
            title="Monthly Fraud Cases",
            markers=True,
            labels={"date": "Month", "fraud_flag": "Fraud Cases"}
        )
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="lightgrey"),
            yaxis=dict(showgrid=True, gridcolor="lightgrey")
        )

        context['visualizations']['trends'] = fig.to_html(full_html=False)
        context['trend_data'] = fraud_over_time.to_dict('records')

    except Exception as e:
        context['error'] = str(e)

    return render(request, 'monthly_trends.html', context)


# views.py
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from datetime import datetime, timedelta
import pandas as pd
from .models import ClaimRecord
from .views import detect_fraud_anomalies  # Make sure this is your fraud detection function


@login_required
def potential_cases(request):
    # Preload provider names once (fast & distinct)
    providers = list(
        ClaimRecord.objects.values_list('prov_name', flat=True).distinct().order_by('prov_name')
    )

    # Query only required fields for performance
    claims_qs = ClaimRecord.objects.only(
        'admit_id', 'claim_prov_date', 'claim_me', 'prov_name', 'amount'
    ).values('admit_id', 'claim_prov_date', 'claim_me', 'prov_name', 'amount')

    df = pd.DataFrame(list(claims_qs))

    if df.empty:
        return render(request, 'potential_cases.html', {
            'fraud_cases': [],
            'date_ranges': ["All Time", "Last 7 Days", "Last 30 Days", "This Month", "Custom Range"],
            'risk_levels': ["High", "Medium", "Low"],
            'providers': providers
        })

    # Convert Decimal to float
    df['amount'] = df['amount'].astype(float).fillna(0)

    # Fraud score calculation
    df['fraud_score'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    df['fraud_score'] = df['fraud_score'].abs() / df['fraud_score'].max()
    df['fraud_score'] = df['fraud_score'].fillna(0)

    def classify_risk(score):
        if score >= 0.7: return "High"
        elif score >= 0.4: return "Medium"
        return "Low"

    df['risk_level'] = df['fraud_score'].apply(classify_risk)

    # Get filter values
    date_range = request.GET.get('date_range', 'All Time')
    risk_level = request.GET.get('risk_level', '')
    provider = request.GET.get('provider', '')

    # Date range filter
    if date_range != 'All Time':
        today = pd.Timestamp.today()
        if date_range == 'Last 7 Days':
            df = df[df['claim_prov_date'] >= today - pd.Timedelta(days=7)]
        elif date_range == 'Last 30 Days':
            df = df[df['claim_prov_date'] >= today - pd.Timedelta(days=30)]
        elif date_range == 'This Month':
            df = df[df['claim_prov_date'].dt.month == today.month]

    # Risk filter
    if risk_level:
        df = df[df['risk_level'] == risk_level]

    # Provider filter (skip if 'All')
    if provider and provider != "All Providers":
        df = df[df['prov_name'] == provider]

    fraud_df = df[df['risk_level'].isin(['High', 'Medium'])]

    return render(request, 'potential_cases.html', {
        'fraud_cases': fraud_df.to_dict('records'),
        'date_ranges': ["All Time", "Last 7 Days", "Last 30 Days", "This Month", "Custom Range"],
        'risk_levels': ["High", "Medium", "Low"],
        'providers': ["All Providers"] + providers  # Add All Providers option
    })

    



@login_required
def geospatial_heatmap(request):
    """View for geospatial heatmap"""
    context = {
        'active_tab': 'fraud-detection',
        'visualizations': {}
    }
    
    try:
        # Load data from session
        fraud_json = request.session.get('fraud_data')
        if not fraud_json:
            return JsonResponse({'error': 'No data available'}, status=400)
            
        df = pd.read_json(fraud_json, orient='records')
        
        # Add sample geospatial data if not present
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            # Generate random coordinates for Nairobi area
            df['latitude'] = -1.2833 + np.random.random(len(df)) * 0.2
            df['longitude'] = 36.8167 + np.random.random(len(df)) * 0.2
            
        if 'fraud_flag' in df.columns:
            fraud_data = df[df['fraud_flag'] == 1]
            
            if not fraud_data.empty:
                fig = px.density_mapbox(
                    fraud_data,
                    lat='latitude',
                    lon='longitude',
                    z='amount',
                    radius=10,
                    center=dict(lat=df['latitude'].mean(), lon=df['longitude'].mean()),
                    zoom=10,
                    mapbox_style="stamen-terrain",
                    title='Fraud Claim Density by Location'
                )
                
                context['visualizations']['heatmap'] = fig.to_html(full_html=False)
                
    except Exception as e:
        context['error'] = str(e)
    
    return render(request, 'geospatial_heatmap.html', context)




###########
##########

########### Reporting functionality
from django.shortcuts import render
import pandas as pd
import numpy as np
from django.http import HttpResponse
from io import BytesIO
import json
from reportlab.lib.pagesizes import letter

from datetime import datetime
import xlsxwriter
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

@login_required
def generate_pdf_report(data, report_type):
    """Generate a PDF report (simplified for demo)"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    elements = []
    elements.append(Paragraph(f"{report_type} Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Add some sample content
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Total records: {len(data)}", styles['Normal']))
    
    if report_type == "Fraud":
        fraud_count = data['fraud_flag'].sum() if 'fraud_flag' in data.columns else 0
        elements.append(Paragraph(f"Potential fraud cases: {fraud_count}", styles['Normal']))
    
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

@login_required
def generate_excel_report(data, report_type):
    """Generate an Excel report"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Report Data')
        
        # Add some basic formatting
        workbook = writer.book
        worksheet = writer.sheets['Report Data']
        
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        for col_num, value in enumerate(data.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Add a summary sheet
        summary_sheet = workbook.add_worksheet('Summary')
        summary_sheet.write(0, 0, f"{report_type} Report Summary")
        summary_sheet.write(1, 0, f"Generated on: {datetime.now().strftime('%Y-%m-%d')}")
        summary_sheet.write(2, 0, f"Total records: {len(data)}")
        
        if report_type == "Fraud" and 'fraud_flag' in data.columns:
            fraud_count = data['fraud_flag'].sum()
            summary_sheet.write(3, 0, f"Potential fraud cases: {fraud_count}")
    
    excel_data = output.getvalue()
    output.close()
    return excel_data

@login_required
def reporting_home(request):
    """Main view for reporting"""
    context = {
        'active_tab': 'reporting',
        'visualizations': {}
    }
    
    try:
        # Sample data - replace with your actual data loading logic
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'amount': np.random.randint(1000, 5000, 100),
            'provider_name': np.random.choice(['Hospital A', 'Clinic B', 'Center C', 'Pharmacy D'], 100),
            'diagnosis': np.random.choice(['Malaria', 'Flu', 'Injury', 'Checkup'], 100),
            'category': np.random.choice(['Outpatient', 'Inpatient', 'Dental', 'Optical'], 100),
            'claim_me': np.random.choice(['MEM001', 'MEM002', 'MEM003', 'MEM004', 'MEM005'], 100),
            'fraud_flag': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
            'fraud_score': np.random.random(100) * 0.5
        })
        
        # Store in session for other views to access
        request.session['reporting_data'] = df.to_json(orient='records')
        
    except Exception as e:
        context['error'] = str(e)
    
    return render(request, 'download_reports.html', context)

@login_required
def claim_drilldown(request):
    """View for claim-level drilldown"""
    context = {
        'active_tab': 'reporting',
        'visualizations': {}
    }
    
    try:
        # Load data from session
        report_json = request.session.get('reporting_data')
        if not report_json:
            return JsonResponse({'error': 'No data available'}, status=400)
            
        df = pd.read_json(report_json, orient='records')
        context['claims_data'] = df.to_dict('records')
        
    except Exception as e:
        context['error'] = str(e)
    
    return render(request, 'claim_drilldown.html', context)

@login_required
def custom_filters(request):
    """View for custom filtering"""
    context = {
        'active_tab': 'reporting',
        'visualizations': {}
    }
    
    try:
        # Load data from session
        report_json = request.session.get('reporting_data')
        if not report_json:
            return JsonResponse({'error': 'No data available'}, status=400)
            
        df = pd.read_json(report_json, orient='records')
        
        # Get unique values for filters
        context['categories'] = df['category'].unique().tolist() if 'category' in df.columns else []
        context['providers'] = df['provider_name'].unique().tolist() if 'provider_name' in df.columns else []
        
        # Get min/max dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            context['min_date'] = df['date'].min().strftime('%Y-%m-%d')
            context['max_date'] = df['date'].max().strftime('%Y-%m-%d')
        
        # Get min/max amounts
        if 'amount' in df.columns:
            context['min_amount'] = df['amount'].min()
            context['max_amount'] = df['amount'].max()
        
        # Handle filter submission
        if request.method == 'POST':
            filtered_data = df.copy()
            
            # Apply date filter
            if 'start_date' in request.POST and 'end_date' in request.POST:
                start_date = request.POST['start_date']
                end_date = request.POST['end_date']
                if start_date and end_date:
                    filtered_data = filtered_data[
                        (pd.to_datetime(filtered_data['date']) >= pd.to_datetime(start_date)) & 
                        (pd.to_datetime(filtered_data['date']) <= pd.to_datetime(end_date))
                    ]
            
            # Apply amount filter
            if 'min_amount' in request.POST and 'max_amount' in request.POST:
                min_amount = float(request.POST['min_amount'])
                max_amount = float(request.POST['max_amount'])
                filtered_data = filtered_data[
                    (filtered_data['amount'] >= min_amount) & 
                    (filtered_data['amount'] <= max_amount)
                ]
            
            # Apply category filter
            if 'categories' in request.POST:
                selected_categories = request.POST.getlist('categories')
                if selected_categories:
                    filtered_data = filtered_data[filtered_data['category'].isin(selected_categories)]
            
            # Apply provider filter
            if 'providers' in request.POST:
                selected_providers = request.POST.getlist('providers')
                if selected_providers:
                    filtered_data = filtered_data[filtered_data['provider_name'].isin(selected_providers)]
            
            # Prepare filtered data for response
            context['filtered_data'] = filtered_data.to_dict('records')
            context['filter_count'] = len(filtered_data)
            
            # Handle export requests
            if 'export_csv' in request.POST:
                response = HttpResponse(content_type='text/csv')
                response['Content-Disposition'] = 'attachment; filename="filtered_claims.csv"'
                filtered_data.to_csv(path_or_buf=response, index=False)
                return response
                
            if 'export_excel' in request.POST:
                response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                response['Content-Disposition'] = 'attachment; filename="filtered_claims.xlsx"'
                with BytesIO() as bio:
                    with pd.ExcelWriter(bio, engine='xlsxwriter') as writer:
                        filtered_data.to_excel(writer, index=False)
                    response.write(bio.getvalue())
                return response
        
    except Exception as e:
        context['error'] = str(e)
    
    return render(request, 'custom_filters.html', context)