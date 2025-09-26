from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.db.models import Sum, Count
from matplotlib.style import context
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
from django.http import HttpResponse

def health_check(request):
    print("[DEBUG] /healthz endpoint called")
    return HttpResponse("OK", content_type="text/plain")



# Removed os.makedirs from global scope to avoid blocking Gunicorn workers
# templates_dir = os.path.join(settings.BASE_DIR, 'myapp', 'templates', 'myapp')
# os.makedirs(templates_dir, exist_ok=True)

# Create your views here.

def landing(request):
    print(f"[DEBUG] landing view called. User authenticated: {request.user.is_authenticated}")
    if request.user.is_authenticated:
        return redirect('home')  # Optional: skip landing for logged-in users
    return render(request, 'myapp/landing.html')



def login_view(request):
    if request.user.is_authenticated:
        # Users who are already logged in are redirected.
        return redirect('claims_upload_landing') 

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
                    return redirect('claims_upload_landing')
            except UserProfile.DoesNotExist:
                # A sensible default for users without a profile.
                return redirect('claims_upload_landing')
    else:
        # For a GET request, create a new, blank form.
        form = AuthenticationForm()

    # Render the login page with the form.
    return render(request, 'myapp/login.html', {'form': form})


def get_database_tables():
    """Return all non-internal tables from the database (works with SQLite & PostgreSQL)."""
    vendor = connection.vendor  # 'sqlite', 'postgresql', 'mysql', etc.

    with connection.cursor() as cursor:
        if vendor == 'sqlite':
            cursor.execute("""
                SELECT name 
                FROM sqlite_master 
                WHERE type='table' 
                AND name NOT LIKE 'sqlite_%'
                AND name NOT LIKE 'django_%'
                AND name NOT LIKE 'auth_%'
                AND name NOT LIKE 'sessions%'
            """)
        elif vendor == 'postgresql':
            cursor.execute("""
                SELECT tablename 
                FROM pg_catalog.pg_tables 
                WHERE schemaname='public'
                AND tablename NOT LIKE 'django_%'
                AND tablename NOT LIKE 'auth_%'
                AND tablename NOT LIKE 'sessions%'
            """)
        else:
            raise NotImplementedError(f"Database vendor '{vendor}' not supported yet.")

        return [row[0] for row in cursor.fetchall()]
    

def claims_upload_landing(request):
    """View for the claims upload landing page"""
    return render(request, 'claims_upload_landing.html')

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

    df_head = None
    desc_stats = None
    visualizations = None
    username = request.user.username

    # Persist selected dataset across requests
    if selected_id:
        request.session['selected_dataset'] = selected_id
    else:
        selected_id = request.session.get('selected_dataset')

    if selected_id and selected_id in dataset_ids:
        df = pd.read_sql(f'SELECT * FROM "{selected_id}"', connection)

        if not df.empty:
            # Preview first 10 rows
            df_head = df.head(10).to_dict(orient='records')

            # Ensure amount is numeric
            if 'amount' in df.columns:
                df['amount'] = pd.to_numeric(
                    df['amount'].astype(str).str.replace(r'[^\d.]', '', regex=True).replace('', '0'),
                    errors='coerce'
                )

            # Parse claim_prov_date → datetime
            if 'claim_prov_date' in df.columns:
                df['datetime'] = pd.to_datetime(df['claim_prov_date'], errors='coerce', dayfirst=True)

                # Fill missing dates with random range values
                if df['datetime'].isna().any():
                    start_dt = pd.to_datetime('2023-01-01')
                    end_dt = pd.to_datetime(timezone.now().date())
                    random_dates = pd.to_datetime(
                        np.random.randint(start_dt.value // 10**9, end_dt.value // 10**9,
                                          size=df['datetime'].isna().sum()),
                        unit='s'
                    )
                    df.loc[df['datetime'].isna(), 'datetime'] = random_dates

                # Apply user date filters
                if start_date:
                    df = df[df['datetime'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['datetime'] <= pd.to_datetime(end_date)]

            # Optional descriptive stats
            if show_stats:
                desc_stats = df.describe(include='all').transpose().reset_index().to_dict(orient='records')

            # ==============================
            # 🔹 Summary stats + charts
            # ==============================
            if {'amount', 'claim_me', 'claim_ce'}.issubset(df.columns):

                # ✅ Business-friendly calculations
                total_invoices = len(df)                   # total rows
                total_claims = df['claim_ce'].nunique()    # unique claims
                total_amount = df['amount'].sum()
                unique_members = df['claim_me'].nunique()

                summary_stats = {
                    'total_invoices': int(total_invoices),
                    'total_claims': int(total_claims),
                    'total_amount': float(total_amount),
                    'unique_members': int(unique_members),
                    'avg_claim': float(total_amount / unique_members) if unique_members else 0.0
                }

                # ---- 🔹 Claims Over Time ----
                claims_time_chart = None
                if 'datetime' in df.columns:
                    df_time = df.set_index('datetime').sort_index()
                    daily_df = df_time.resample('D').size().reset_index(name='count')
                    weekly_df = df_time.resample('W-MON').size().reset_index(name='count')
                    monthly_df = df_time.resample('M').size().reset_index(name='count')

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=daily_df['datetime'], y=daily_df['count'],
                                             mode='lines+markers', name='Daily Claims',
                                             line=dict(color='#e30613'), visible=True))
                    fig.add_trace(go.Scatter(x=weekly_df['datetime'], y=weekly_df['count'],
                                             mode='lines+markers', name='Weekly Claims',
                                             line=dict(color='#e30613'), visible=False))
                    fig.add_trace(go.Scatter(x=monthly_df['datetime'], y=monthly_df['count'],
                                             mode='lines+markers', name='Monthly Claims',
                                             line=dict(color='#e30613'), visible=False))

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
                            buttons=[
                                dict(label="Daily", method="update", args=[{"visible": [True, False, False]}, {"title": "Daily Claims Submitted"}]),
                                dict(label="Weekly", method="update", args=[{"visible": [False, True, False]}, {"title": "Weekly Claims Submitted"}]),
                                dict(label="Monthly", method="update", args=[{"visible": [False, False, True]}, {"title": "Monthly Claims Submitted"}]),
                            ]
                        )]
                    )
                    claims_time_chart = fig.to_html(full_html=False)

                # ---- 🔹 Unique Claims by Benefit ----
                category_amounts = None
                if 'benefit_desc' in df.columns:
                    category_unique_df = (
                        df.groupby('benefit_desc')['claim_me']
                        .nunique()
                        .reset_index(name='unique_members')
                    )
                    fig_cat = px.bar(category_unique_df, x='benefit_desc', y='unique_members',
                                     title='Unique Members by Benefit Category',
                                     color_discrete_sequence=['#e30613'])
                    category_amounts = fig_cat.to_html(full_html=False)

                # ---- 🔹 Sunburst Breakdown ----
                sunburst = None
                if {'benefit', 'benefit_desc'}.issubset(df.columns):
                    fig_sunburst = px.sunburst(df.reset_index(),
                                               path=['benefit', 'benefit_desc'],
                                               values='amount',
                                               title='Claim Amounts Breakdown',
                                               color_discrete_sequence=['#e30613'])
                    sunburst = fig_sunburst.to_html(full_html=False)

                # ---- 🔹 Top Claimants ----
                top_claimants_df = (
                    df.groupby('claim_me')
                    .agg(total_amount=('amount', 'sum'),
                         claim_count=('amount', 'count'))
                    .reset_index()
                    .rename(columns={'claim_me': 'Member ID'})
                    .sort_values(by='total_amount', ascending=False)
                    .head(10)
                )
                fig_top = px.bar(top_claimants_df, x='Member ID', y='total_amount',
                                 title='Top Claimants',
                                 color_discrete_sequence=['#e30613'])
                top_claimants = fig_top.to_html(full_html=False)

                # ---- 🔹 Claim Frequency Distribution ----
                claim_freq_df = (
                    df['claim_me']
                    .value_counts()
                    .reset_index(name='frequency')
                    .rename(columns={'index': 'Member ID'})
                )
                fig_freq = px.histogram(claim_freq_df, x='frequency', nbins=20,
                                        title='Claim Frequency Distribution',
                                        color_discrete_sequence=['#e30613'])
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

    return render(request, 'myapp/home.html', {
        'dataset_ids': dataset_ids,
        'selected_id': selected_id,
        'df_head': df_head,
        'desc_stats': desc_stats,
        'visualizations': visualizations,
        'username': username,
        'start_date': start_date,
        'end_date': end_date
    })


    

# =========================================
# CLAIMS PREDICTION VIEW
# =========================================
# views.py

import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from django.shortcuts import render
from django.http import JsonResponse
from django.db import connection
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from django.views.decorators.csrf import csrf_exempt
from functools import lru_cache
import time
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_GET
from django.apps import apps
from django.core.exceptions import ObjectDoesNotExist
import random
import colorsys

# Cache dataset metadata
@lru_cache(maxsize=32)
def get_database_tables_cached():
    """Get all available database tables (cached, works with SQLite & PostgreSQL)."""
    try:
        vendor = connection.vendor  # 'sqlite', 'postgresql', 'mysql', etc.

        with connection.cursor() as cursor:
            if vendor == 'sqlite':
                cursor.execute("""
                    SELECT name 
                    FROM sqlite_master 
                    WHERE type='table' 
                    AND name NOT LIKE 'sqlite_%' 
                    AND name NOT LIKE 'django_%'
                    AND name NOT LIKE 'auth_%'
                    AND name NOT LIKE 'sessions%'
                """)
            elif vendor == 'postgresql':
                cursor.execute("""
                    SELECT tablename 
                    FROM pg_catalog.pg_tables 
                    WHERE schemaname='public'
                    AND tablename NOT LIKE 'django_%'
                    AND tablename NOT LIKE 'auth_%'
                    AND tablename NOT LIKE 'sessions%'
                """)
            else:
                raise NotImplementedError(f"Database vendor '{vendor}' not supported yet.")

            return [row[0] for row in cursor.fetchall()]

    except Exception as e:
        print(f"Error getting database tables: {e}")
        return []

# Cache dataset
@lru_cache(maxsize=8)
def load_dataset_cached(table_name: str):
    """Load dataset from database table (cached)"""
    try:
        with connection.cursor() as cursor:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)
        return df
    except Exception as e:
        print(f"Error loading dataset {table_name}: {e}")
        return pd.DataFrame()

def get_unique_values(df, column_name, limit=100):
    """Get unique values from a column for filter dropdowns"""
    if column_name not in df.columns:
        return []
    
    try:
        unique_vals = df[column_name].dropna().unique().tolist()
        return sorted([str(val) for val in unique_vals[:limit]])
    except Exception as e:
        print(f"Error getting unique values for {column_name}: {e}")
        return []

def prepare_data(df):
    """Clean and prepare data for analysis"""
    if df.empty:
        return df
        
    data = df.copy()
    
    try:
        # Amount conversion
        if 'amount' in data.columns:
            data['amount'] = data['amount'].astype(str).str.replace(",", "", regex=False)
            data['amount'] = pd.to_numeric(data['amount'], errors='coerce').fillna(0)
        else:
            data['amount'] = 0.0

        # Quantity conversion
        if 'quantity' in data.columns:
            data['quantity'] = pd.to_numeric(data['quantity'], errors='coerce').fillna(0).astype(int)

        # Date conversion - try all possible date columns
        date_columns = ['claim_prov_date', 'date', 'service_date', 'admission_date']
        for col in date_columns:
            if col in data.columns:
                data['datetime'] = pd.to_datetime(data[col], errors='coerce')
                data = data[data['datetime'].notna()]
                break
        else:
            # If no date column found, create a dummy datetime column
            data['datetime'] = pd.Timestamp('today')
            print("Warning: No date column found, using current date")

        # Age calculation from DOB
        if 'dob' in data.columns:
            data['dob'] = pd.to_datetime(data['dob'], errors='coerce')
            data['age'] = (pd.to_datetime('today') - data['dob']).dt.days // 365
            data['age'] = data['age'].fillna(0).astype(int)
            data['age_group'] = pd.cut(data['age'], bins=[0, 18, 30, 45, 60, 100], 
                                      labels=['0-18', '19-30', '31-45', '46-60', '60+'])

        return data
    except Exception as e:
        print(f"Error preparing data: {e}")
        return df

def apply_filters(df, request_params):
    """Apply all filters to the dataset"""
    if df.empty:
        return df
        
    filtered_data = df.copy()
    
    try:
        # Amount filter
        min_amount = float(request_params.get('min_amount', 0))
        max_amount = float(request_params.get('max_amount', 1000000))
        filtered_data = filtered_data[(filtered_data['amount'] >= min_amount) & 
                                     (filtered_data['amount'] <= max_amount)]

        # Quantity filter
        if 'quantity' in filtered_data.columns:
            min_quantity = int(request_params.get('min_quantity', 0))
            max_quantity = int(request_params.get('max_quantity', 1000))
            filtered_data = filtered_data[(filtered_data['quantity'] >= min_quantity) & 
                                         (filtered_data['quantity'] <= max_quantity)]

        # Date range filter
        start_date = request_params.get('start_date')
        end_date = request_params.get('end_date')
        if start_date and 'datetime' in filtered_data.columns and filtered_data['datetime'].notna().any():
            filtered_data = filtered_data[filtered_data['datetime'] >= pd.to_datetime(start_date)]
        if end_date and 'datetime' in filtered_data.columns and filtered_data['datetime'].notna().any():
            filtered_data = filtered_data[filtered_data['datetime'] <= pd.to_datetime(end_date)]

        # Time period filter
        time_period = request_params.get('time_period', 'all')
        if time_period != 'all' and 'datetime' in filtered_data.columns and filtered_data['datetime'].notna().any():
            now = pd.Timestamp.now()
            if time_period == 'year':
                filtered_data = filtered_data[filtered_data['datetime'] >= now - pd.DateOffset(years=1)]
            elif time_period == 'quarter':
                filtered_data = filtered_data[filtered_data['datetime'] >= now - pd.DateOffset(months=3)]
            elif time_period == 'month':
                filtered_data = filtered_data[filtered_data['datetime'] >= now - pd.DateOffset(months=1)]
            elif time_period == 'week':
                filtered_data = filtered_data[filtered_data['datetime'] >= now - pd.DateOffset(weeks=1)]

        # Column-based filters
        filter_columns = [
            'gender', 'benefit', 'age_group', 'cost_center', 
            'dependent_type', 'prov_name', 'ailment',
            'claim_pod', 'benefit_desc'
        ]
        
        for col in filter_columns:
            value = request_params.get(col, 'all')
            if value != 'all' and col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col].astype(str).str.strip().str.lower() == value.lower()]

        # Multi-level filters
        for i in range(1, 4):
            filter_column = request_params.get(f'filter_column_{i}', 'None')
            filter_values = request_params.getlist(f'filter_values_{i}')
            
            if filter_column != 'None' and filter_column in filtered_data.columns and filter_values:
                filtered_data = filtered_data[filtered_data[filter_column].astype(str).isin(filter_values)]

        # Drilldown filter
        drilldown_column = request_params.get('drilldown_column')
        drilldown_values = request_params.getlist('drilldown_value')
        
        if drilldown_column and drilldown_values and drilldown_column in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[drilldown_column].astype(str).isin(drilldown_values)]

        return filtered_data
    except Exception as e:
        print(f"Error applying filters: {e}")
        return df

def generate_random_colors(n):
    """Generate n distinct random colors"""
    colors = []
    for i in range(n):
        # Generate colors with good contrast
        hue = i / n
        saturation = 0.7 + random.random() * 0.3
        value = 0.5 + random.random() * 0.5
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})')
    return colors

def create_comparison_charts(df, comparison_column, comparison_values, x_axis):
    """Create comparison charts for multiple values"""
    charts = []
    
    if not comparison_column or not comparison_values or comparison_column not in df.columns:
        return charts
    
    # Filter data for each comparison value
    for value in comparison_values:
        filtered_df = df[df[comparison_column].astype(str) == value]
        
        if filtered_df.empty:
            continue
            
        # Group by x_axis and calculate total amount
        if x_axis in filtered_df.columns:
            amount_by_x = filtered_df.groupby(x_axis)['amount'].sum().reset_index()
            
            # Randomize order (don't sort by amount)
            amount_by_x = amount_by_x.sample(frac=1, random_state=42).head(20)
            
            # Generate random colors
            colors = generate_random_colors(len(amount_by_x))
            
            # Create bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=amount_by_x[x_axis],
                y=amount_by_x['amount'],
                name=value,
                marker_color=colors,
                text=amount_by_x['amount'],
                texttemplate='%{text:.2s}',
                textposition='auto'
            ))
            
            # Add average line
            avg_amount = amount_by_x['amount'].mean()
            fig.add_hline(
                y=avg_amount,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Average: {avg_amount:,.2f}",
                annotation_position="bottom right"
            )
            
            fig.update_layout(
                title=f"Amount by {x_axis.replace('_', ' ').title()} for {comparison_column}: {value}",
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title="Total Amount (KES)",
                showlegend=False,
                xaxis_tickangle=-45
            )
            
            charts.append({
                'title': f"{comparison_column}: {value}",
                'html': fig.to_html(full_html=False)
            })
    
    return charts

def create_main_visualizations(df, analysis_type, x_axis, drilldown_column=None, drilldown_values=None, 
                              comparison_column=None, comparison_values=None):
    """Create all visualizations based on analysis type"""
    visualizations = {}
    
    if df.empty:
        return visualizations
    
    # Filter for drilldown if specified
    analysis_df = df.copy()
    if drilldown_column and drilldown_values and drilldown_column in df.columns:
        analysis_df = analysis_df[analysis_df[drilldown_column].astype(str).isin(drilldown_values)]
        visualizations['drilldown_info'] = f"Drilldown: {drilldown_column} = {', '.join(drilldown_values)}"
    
    # Create comparison charts if specified
    if comparison_column and comparison_values and comparison_column in df.columns:
        comparison_charts = create_comparison_charts(analysis_df, comparison_column, comparison_values, x_axis)
        if comparison_charts:
            visualizations['comparison_charts'] = comparison_charts
            visualizations['comparison_info'] = f"Comparing {len(comparison_values)} values from {comparison_column}"
    
    # Summary statistics
    try:
        total_amount = analysis_df['amount'].sum()
        total_claims = len(analysis_df)
        avg_claim = total_amount / total_claims if total_claims > 0 else 0
        
        summary_stats = {
            'total_amount': total_amount,
            'total_claims': analysis_df['claim_ce'].nunique() if 'claim_ce' in analysis_df.columns else 0,
            'avg_claim': total_amount / analysis_df['prov_name'].nunique() if 'prov_name' in analysis_df.columns and analysis_df['prov_name'].nunique() > 0 else 0,
            'unique_members': analysis_df['claim_me'].nunique() if 'claim_me' in analysis_df.columns else 0,
            'unique_providers': analysis_df['prov_name'].nunique() if 'prov_name' in analysis_df.columns else 0,
            'unique_diagnoses': analysis_df['ailment'].nunique() if 'ailment' in analysis_df.columns else 0,
        }
        visualizations['summary_stats'] = summary_stats
    except Exception as e:
        print(f"Error calculating summary stats: {e}")
    
    import random

    # MAIN CHART: Amount by selected X-axis
    if x_axis in analysis_df.columns:
        try:
            amount_by_x = analysis_df.groupby(x_axis)['amount'].sum().reset_index()
            amount_by_x = amount_by_x.sort_values('amount', ascending=False).head(20)

            # Shuffle rows for random order
            amount_by_x = amount_by_x.sample(frac=1, random_state=None).reset_index(drop=True)

            # Generate random colors for each bar
            colors = ['#%06X' % random.randint(0, 0xFFFFFF) for _ in range(len(amount_by_x))]

            fig = px.bar(
                amount_by_x, 
                x=x_axis, 
                y='amount', 
                title=f"Total Amount by {x_axis.replace('_', ' ').title()}",
                labels={'amount': 'Total Amount (KES)', x_axis: x_axis.replace('_', ' ').title()},
                color=amount_by_x[x_axis],  # color by x values
                color_discrete_sequence=colors  # assign random colors
            )

            # Add horizontal average line
            avg_value = amount_by_x['amount'].mean()
            fig.add_hline(y=avg_value, line_dash="dash", line_color="red",
                        annotation_text="Average", annotation_position="top right")

            fig.update_layout(xaxis_tickangle=-45)
            visualizations['main_chart'] = fig.to_html(full_html=False)

        except Exception as e:
            print(f"Error creating main chart: {e}")

    
    # TIME SERIES ANALYSIS
    if analysis_type == 'temporal' and 'datetime' in analysis_df.columns and analysis_df['datetime'].notna().any():
        try:
            # Monthly trends
            ts_df = analysis_df.groupby(pd.Grouper(key='datetime', freq='M')).agg(
                total_amount=('amount', 'sum'),
                claim_count=('amount', 'size')
            ).reset_index()
            
            fig1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig1.add_trace(go.Bar(x=ts_df['datetime'], y=ts_df['total_amount']/1_000_000,
                                name='Total Amount (KES M)', marker_color='#1f77b4'),
                        secondary_y=False)
            fig1.add_trace(go.Scatter(x=ts_df['datetime'], y=ts_df['claim_count'],
                                    name='Claim Count', line=dict(color='#ff7f0e'), mode='lines+markers'),
                        secondary_y=True)
            fig1.update_layout(title="Monthly Claims Trend",
                            yaxis=dict(title='Amount (KES M)', side='left'),
                            yaxis2=dict(title='Claim Count', side='right'))
            visualizations['time_series'] = fig1.to_html(full_html=False)
            
            # Daily heatmap
            analysis_df['day_of_week'] = analysis_df['datetime'].dt.day_name()
            analysis_df['hour_of_day'] = analysis_df['datetime'].dt.hour
            heat_df = analysis_df.groupby(['day_of_week', 'hour_of_day']).size().reset_index(name='count')
            
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heat_df['day_of_week'] = pd.Categorical(heat_df['day_of_week'], categories=days, ordered=True)
            heat_df = heat_df.sort_values('day_of_week')
            
            fig2 = px.density_heatmap(heat_df, x='hour_of_day', y='day_of_week', z='count',
                                    title="Claims by Day and Hour",
                                    color_continuous_scale='Viridis')
            visualizations['heatmap'] = fig2.to_html(full_html=False)
            
        except Exception as e:
            print(f"Error creating time series charts: {e}")
    
    # PROVIDER ANALYSIS
    if analysis_type == 'provider' and 'prov_name' in analysis_df.columns:
        try:
            # Top providers
            top_providers = analysis_df.groupby('prov_name').agg(
                total_amount=('amount', 'sum'),
                claim_count=('amount', 'size')
            ).nlargest(10, 'total_amount').reset_index()
            
            fig1 = px.bar(top_providers, x='prov_name', y='total_amount',
                         title="Top Providers by Total Amount",
                         labels={'prov_name': 'Provider', 'total_amount': 'Total Amount (KES)'})
            fig1.update_layout(xaxis_tickangle=-45)
            visualizations['top_providers'] = fig1.to_html(full_html=False)
            
            # Provider efficiency (amount per claim)
            provider_stats = analysis_df.groupby('prov_name').agg(
                total_amount=('amount', 'sum'),
                claim_count=('amount', 'size')
            ).reset_index()
            provider_stats['avg_per_claim'] = provider_stats['total_amount'] / provider_stats['claim_count']
            
            fig2 = px.scatter(provider_stats, x='claim_count', y='avg_per_claim', size='total_amount',
                             hover_name='prov_name', title="Provider Efficiency Analysis",
                             labels={'claim_count': 'Number of Claims', 
                                    'avg_per_claim': 'Average Amount per Claim (KES)'})
            visualizations['provider_efficiency'] = fig2.to_html(full_html=False)
            
        except Exception as e:
            print(f"Error creating provider analysis: {e}")
    
    # DIAGNOSIS ANALYSIS
    if analysis_type == 'diagnosis' and 'ailment' in analysis_df.columns:
        try:
            # Top diagnoses
            top_diagnoses = analysis_df.groupby('ailment').agg(
                total_amount=('amount', 'sum'),
                claim_count=('amount', 'size')
            ).nlargest(10, 'total_amount').reset_index()
            
            fig1 = px.bar(top_diagnoses, x='ailment', y='total_amount',
                         title="Top Diagnoses by Total Amount",
                         labels={'ailment': 'Diagnosis', 'total_amount': 'Total Amount (KES)'})
            fig1.update_layout(xaxis_tickangle=-45)
            visualizations['top_diagnoses'] = fig1.to_html(full_html=False)
            
            # Diagnosis by age group
            if 'age_group' in analysis_df.columns:
                diagnosis_age = analysis_df.groupby(['ailment', 'age_group'])['amount'].sum().reset_index()
                top_diag = diagnosis_age.groupby('ailment')['amount'].sum().nlargest(5).index
                diagnosis_age = diagnosis_age[diagnosis_age['ailment'].isin(top_diag)]
                
                fig2 = px.bar(diagnosis_age, x='ailment', y='amount', color='age_group',
                             title="Top Diagnoses by Age Group",
                             labels={'ailment': 'Diagnosis', 'amount': 'Total Amount (KES)'})
                fig2.update_layout(xaxis_tickangle=-45)
                visualizations['diagnosis_age'] = fig2.to_html(full_html=False)
                
        except Exception as e:
            print(f"Error creating diagnosis analysis: {e}")
    
    # BENEFIT ANALYSIS
    if analysis_type == 'benefit' and 'benefit' in analysis_df.columns:
        try:
            # Top benefits
            top_benefits = analysis_df.groupby('benefit').agg(
                total_amount=('amount', 'sum'),
                claim_count=('amount', 'size')
            ).nlargest(10, 'total_amount').reset_index()
            
            fig1 = px.bar(top_benefits, x='benefit', y='total_amount',
                         title="Top Benefits by Total Amount",
                         labels={'benefit': 'Benefit Type', 'total_amount': 'Total Amount (KES)'})
            fig1.update_layout(xaxis_tickangle=-45)
            visualizations['top_benefits'] = fig1.to_html(full_html=False)
            
            # Benefit distribution
            fig2 = px.pie(top_benefits, values='total_amount', names='benefit',
                         title="Benefit Distribution by Amount")
            visualizations['benefit_distribution'] = fig2.to_html(full_html=False)
            
        except Exception as e:
            print(f"Error creating benefit analysis: {e}")
    
    # DEMOGRAPHIC ANALYSIS
    if analysis_type == 'demographic':
        try:
            # Age group analysis
            if 'age_group' in analysis_df.columns:
                age_data = analysis_df.groupby('age_group')['amount'].sum().reset_index()
                fig1 = px.bar(age_data, x='age_group', y='amount',
                             title="Total Amount by Age Group",
                             labels={'age_group': 'Age Group', 'amount': 'Total Amount (KES)'})
                visualizations['age_analysis'] = fig1.to_html(full_html=False)
            
            # Gender analysis
            if 'gender' in analysis_df.columns:
                gender_data = analysis_df.groupby('gender')['amount'].sum().reset_index()
                fig2 = px.pie(gender_data, values='amount', names='gender',
                             title="Amount Distribution by Gender")
                visualizations['gender_analysis'] = fig2.to_html(full_html=False)
            
            # Dependent type analysis
            if 'dependent_type' in analysis_df.columns:
                dep_data = analysis_df.groupby('dependent_type')['amount'].sum().reset_index()
                fig3 = px.bar(dep_data, x='dependent_type', y='amount',
                             title="Total Amount by Dependent Type",
                             labels={'dependent_type': 'Dependent Type', 'amount': 'Total Amount (KES)'})
                fig3.update_layout(xaxis_tickangle=-45)
                visualizations['dependent_analysis'] = fig3.to_html(full_html=False)
                
        except Exception as e:
            print(f"Error creating demographic analysis: {e}")
    
    # ADVANCED ANALYSIS
    if analysis_type == 'advanced':
        try:
            # Correlation heatmap
            numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1 and 'amount' in numeric_cols:
                corr_matrix = analysis_df[numeric_cols].corr()
                fig1 = px.imshow(corr_matrix, title="Correlation Matrix")
                visualizations['correlation'] = fig1.to_html(full_html=False)
            
            # Outlier detection
            if 'amount' in analysis_df.columns:
                Q1 = analysis_df['amount'].quantile(0.25)
                Q3 = analysis_df['amount'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = analysis_df[(analysis_df['amount'] < lower_bound) | (analysis_df['amount'] > upper_bound)]
                normal = analysis_df[(analysis_df['amount'] >= lower_bound) & (analysis_df['amount'] <= upper_bound)]
                
                fig2 = go.Figure()
                fig2.add_trace(go.Box(y=normal['amount'], name='Normal Claims', boxpoints=False))
                fig2.add_trace(go.Box(y=outliers['amount'], name='Outliers', boxpoints='all', jitter=0.3, pointpos=0))
                fig2.update_layout(title="Outlier Detection in Claim Amounts")
                visualizations['outliers'] = fig2.to_html(full_html=False)
            
            # Treemap of claims hierarchy
            if 'benefit' in analysis_df.columns and 'prov_name' in analysis_df.columns:
                treemap_df = analysis_df.groupby(['benefit', 'prov_name']).agg(
                    total_amount=('amount', 'sum'),
                    count=('amount', 'size')
                ).reset_index().nlargest(20, 'total_amount')
                
                fig3 = px.treemap(treemap_df, path=['benefit', 'prov_name'], values='total_amount',
                                 color='count', color_continuous_scale='Blues',
                                 title="Claims Hierarchy: Benefit → Provider")
                visualizations['treemap'] = fig3.to_html(full_html=False)
                
        except Exception as e:
            print(f"Error creating advanced analysis: {e}")
    
    return visualizations

@login_required(login_url='login')
def claims_prediction_dataset_view(request):
    try:
        # --- Step 1: Load datasets ---
        dataset_ids = get_database_tables_cached()
        if not dataset_ids:
            return render(request, 'myapp/claim_prediction.html', {
                'dataset_ids': [],
                'debug_info': 'No datasets found',
                'visualizations': {}
            })

        # --- Step 2: Get parameters ---
        selected_id = request.GET.get('dataset_id') or dataset_ids[0]
        analysis_type = request.GET.get('analysis_type', 'temporal')
        x_axis = request.GET.get('x_axis', 'benefit')

        # ✅ MULTI-SELECT DRILLDOWN FIX
        drilldown_column = request.GET.get('drilldown_column')
        drilldown_values = request.GET.getlist('drilldown_value')  # multiple allowed

        # ✅ MULTI-SELECT COMPARISON FIX
        comparison_column = request.GET.get('comparison_column')
        comparison_values = request.GET.getlist('comparison_values')  # multiple allowed

        # --- Step 3: Load and prepare dataset ---
        df = load_dataset_cached(selected_id)
        if df.empty:
            return render(request, 'myapp/claim_prediction.html', {
                'dataset_ids': dataset_ids,
                'debug_info': f"Dataset '{selected_id}' is empty",
                'visualizations': {}
            })
        prepared_df = prepare_data(df)

        # --- Step 4: Apply all filters (amount, date, custom, drilldown) ---
        filtered_df = apply_filters(prepared_df, request.GET)

        # --- Step 5: Get dropdown options ---
        all_columns = list(filtered_df.columns) if not filtered_df.empty else []
        cat_cols = [
            col for col in filtered_df.columns
            if filtered_df[col].dtype in ['object', 'category'] or filtered_df[col].nunique() < 20
        ] if not filtered_df.empty else []

        # Get comparison options if column is selected
        comparison_options = []
        if comparison_column and comparison_column in filtered_df.columns:
            comparison_options = get_unique_values(filtered_df, comparison_column)

        def get_column_options(col):
            return sorted(filtered_df[col].dropna().unique().tolist()) if col in filtered_df.columns else []

        age_groups = get_column_options('age_group')
        cost_centers = get_column_options('cost_center')
        dependent_types = get_column_options('dependent_type')
        providers = get_column_options('prov_name')
        diagnoses = get_column_options('ailment')
        benefit_types = get_column_options('benefit')

        # --- Step 6: Multi-level filter options ---
        filter_columns = ['filter_column_1', 'filter_column_2', 'filter_column_3']
        filter_values_options = {}
        for i, col_key in enumerate(filter_columns, 1):
            col_name = request.GET.get(col_key, 'None')
            filter_values_options[f'filter_values_{i}_options'] = (
                get_unique_values(filtered_df, col_name) if col_name != 'None' else []
            )

        # --- Step 7: Generate visualizations (main + drilldown + comparison) ---
        visualizations = create_main_visualizations(
            filtered_df,
            analysis_type,
            x_axis,
            drilldown_column,
            drilldown_values,
            comparison_column,
            comparison_values
        )

        # --- Step 8: Sample preview data ---
        sample_data = (
            filtered_df.head(100).replace({np.nan: None}).to_dict('records')
            if not filtered_df.empty else None
        )
        sample_columns = list(filtered_df.columns) if not filtered_df.empty else []

        # --- Step 9: Build context ---
        context = {
            'dataset_ids': dataset_ids,
            'selected_id': selected_id,
            'analysis_type': analysis_type,
            'x_axis': x_axis,
            'drilldown_column': drilldown_column,
            'drilldown_values': drilldown_values,   # ✅ multiple values preserved
            'all_columns': all_columns,
            'cat_cols': cat_cols,
            'age_groups': age_groups,
            'cost_centers': cost_centers,
            'dependent_types': dependent_types,
            'providers': providers,
            'diagnoses': diagnoses,
            'benefit_types': benefit_types,
            'sample_data': sample_data,
            'sample_columns': sample_columns,
            'visualizations': visualizations,      # ✅ contains plotly HTML
            'debug_info': f"Loaded '{selected_id}' with {len(filtered_df)} rows after filtering"
        }

        # Add multi-level filter options
        context.update(filter_values_options)

        # --- Step 10: AJAX drilldown options ---
        if request.GET.get('get_drilldown_options'):
            column_name = request.GET.get('column_name')
            if column_name and column_name in filtered_df.columns:
                values = get_unique_values(filtered_df, column_name)
                return JsonResponse({'values': values})
            return JsonResponse({'error': 'Invalid column name'})

        # --- Step 11: AJAX request (return partial content) ---
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return render(request, 'myapp/claim_prediction.html', context)

        # --- Step 12: Full page render ---
        return render(request, 'myapp/claim_prediction.html', context)

    except Exception as e:
        print(f"[ERROR] claims_prediction_dataset_view: {e}")
        return render(request, 'myapp/claim_prediction.html', {
            'dataset_ids': get_database_tables_cached(),
            'debug_info': f"Error loading dataset: {str(e)}",
            'visualizations': {}
        })


@require_GET
def get_filter_options(request):
    dataset_id = request.GET.get("dataset_id")
    column_name = request.GET.get("column_name")

    if not dataset_id:
        return HttpResponseBadRequest("Missing dataset_id")

    try:
        df = load_dataset_cached(dataset_id)
        df = prepare_data(df)  # <-- key step

        if df.empty:
            return JsonResponse({"error": f"Dataset '{dataset_id}' is empty or not found"}, status=400)

        if column_name == "__all_columns__":
            return JsonResponse({"values": list(df.columns)})

        if column_name not in df.columns:
            return JsonResponse({"error": f"Column '{column_name}' not found"}, status=400)

        # Optionally apply filters
        filtered_df = apply_filters(df, request.GET)
        values = get_unique_values(filtered_df, column_name)

        return JsonResponse({"values": values})

    except Exception as e:
        print(f"Error in get_filter_options: {e}")
        return JsonResponse({"error": f"Failed to load dataset: {str(e)}"}, status=500)


# Error handling view
def handler500(request):
    return JsonResponse({'error': 'Internal server error'}, status=500)

# Health check endpoint
@require_GET
def health_check(request):
    return JsonResponse({'status': 'ok'})


import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from django.shortcuts import render
from django.http import JsonResponse
from django.db import connection
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import calendar
from django.views.decorators.csrf import csrf_exempt
import warnings
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
warnings.filterwarnings('ignore')

@login_required(login_url='login')
def TemporalAnalysisView(request):
    """
    Main view for comprehensive temporal analysis.
    Dynamically fetches available datasets depending on the database vendor.
    Supports SQLite and PostgreSQL.
    """
    vendor = connection.vendor  # 'sqlite', 'postgresql', etc.

    with connection.cursor() as cursor:
        if vendor == 'sqlite':
            cursor.execute("""
                SELECT name 
                FROM sqlite_master 
                WHERE type='table'
                  AND name NOT LIKE 'sqlite_%'
                  AND name NOT LIKE 'django_%'
                  AND name NOT LIKE 'auth_%'
                  AND name NOT LIKE 'sessions%'
                ORDER BY name
            """)
        elif vendor == 'postgresql':
            cursor.execute("""
                SELECT tablename 
                FROM pg_catalog.pg_tables
                WHERE schemaname = 'public'
                  AND tablename NOT LIKE 'django_%'
                  AND tablename NOT LIKE 'auth_%'
                  AND tablename NOT LIKE 'sessions%'
                ORDER BY tablename
            """)
        else:
            raise NotImplementedError(
                f"Database vendor '{vendor}' is not supported yet."
            )

        dataset_ids = [row[0] for row in cursor.fetchall()]
    
    # Get parameters
    selected_id = request.GET.get('dataset_id') or (dataset_ids[0] if dataset_ids else None)
    time_unit = request.GET.get('time_unit', 'M')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    analysis_type = request.GET.get('analysis_type', 'comprehensive')
    focus_area = request.GET.get('focus_area', 'overview')
    
    # Initialize context
    context = {
        'dataset_ids': dataset_ids,
        'selected_id': selected_id,
        'time_unit': time_unit,
        'start_date': start_date,
        'end_date': end_date,
        'analysis_type': analysis_type,
        'focus_area': focus_area,
        'visualizations': {},
        'summary_stats': {},
        'temporal_insights': {},
        'error': None
    }
    
    if not selected_id or not dataset_ids:
        return render(request, 'myapp/minet_temporal_analysis.html', context)
    
    try:
        # Load dataset
        df = pd.read_sql_query(f"SELECT * FROM {selected_id}", connection)
        
        if df.empty:
            context['error'] = "Selected dataset is empty"
            return render(request, 'myapp/minet_temporal_analysis.html', context)
        
        # Clean and preprocess data
        df = clean_and_preprocess_data(df)
        
        if df.empty:
            context['error'] = "No valid data after preprocessing"
            return render(request, 'myapp/minet_temporal_analysis.html', context)
        
        # Apply date filters
        df = apply_date_filters(df, start_date, end_date)
        
        if df.empty:
            context['error'] = "No data available for the selected date range"
            return render(request, 'myapp/minet_temporal_analysis.html', context)
        
        # Calculate summary statistics
        context['summary_stats'] = calculate_summary_statistics(df)
        
        # Calculate temporal insights
        context['temporal_insights'] = calculate_temporal_insights(df, time_unit)
        
        # Generate visualizations based on analysis type and focus area
        visualizations = generate_temporal_visualizations(df, time_unit, analysis_type, focus_area)
        context['visualizations'] = visualizations
        
    except Exception as e:
        context['error'] = f"Error processing data: {str(e)}"
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
    
    return render(request, 'myapp/minet_temporal_analysis.html', context)

def clean_and_preprocess_data(df):
    """Clean and preprocess the dataset"""
    # Amount conversion
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(
            df['amount'].astype(str).str.replace(",", "").str.replace(r"[^\d.]", "", regex=True),
            errors='coerce'
        ).fillna(0)
    else:
        df['amount'] = 0.0
    
    # Date conversion - try multiple possible date columns
    date_columns = ['claim_prov_date', 'date', 'CLAIM_PROV_DATE', 'ClaimDate', 'claim_date']
    date_col = None
    for col in date_columns:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df[df['datetime'].notna()]
    else:
        # If no date column found, return empty DataFrame
        return pd.DataFrame()
    
    # Clean categorical columns
    categorical_columns = ['benefit', 'ailment', 'prov_name', 'gender', 'dependent_type']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().fillna('Unknown')
    
    # Clean member IDs
    if 'claim_me' in df.columns:
        df['claim_me'] = df['claim_me'].astype(str).str.strip().fillna('Unknown')
    
    return df

def apply_date_filters(df, start_date, end_date):
    """Apply date filters to the dataset"""
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df = df[df['datetime'] >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        df = df[df['datetime'] <= end_dt]
    
    return df

def calculate_summary_statistics(df):
    """Calculate comprehensive summary statistics"""
    total_claims = df['claim_ce'].nunique() if 'claim_ce' in df.columns else 0
    total_amount = df['amount'].sum()
    avg_claim = total_amount / total_claims if total_claims > 0 else 0
    
    # Unique counts
    unique_members = df['claim_me'].nunique() if 'claim_me' in df.columns else 0
    unique_providers = df['prov_name'].nunique() if 'prov_name' in df.columns else 0
    unique_services = df['benefit'].nunique() if 'benefit' in df.columns else 0
    unique_ailments = df['ailment'].nunique() if 'ailment' in df.columns else 0
    
    # Time-based calculations
    date_range = df['datetime'].max() - df['datetime'].min()
    days_in_period = date_range.days + 1 if date_range.days > 0 else 1
    
    claims_per_day = total_claims / days_in_period
    amount_per_day = total_amount / days_in_period
    
    # Previous period comparison (30 days prior)
    prev_start = df['datetime'].min() - timedelta(days=30)
    prev_end = df['datetime'].min() - timedelta(days=1)
    
    prev_df = df[(df['datetime'] >= prev_start) & (df['datetime'] <= prev_end)]
    prev_claims = len(prev_df)
    prev_amount = prev_df['amount'].sum() if not prev_df.empty else 0
    
    # Calculate changes
    claims_change = ((total_claims - prev_claims) / prev_claims * 100) if prev_claims > 0 else 0
    amount_change = ((total_amount - prev_amount) / prev_amount * 100) if prev_amount > 0 else 0
    
    # Top categories
    top_category = df['benefit'].value_counts().index[0] if 'benefit' in df.columns and not df['benefit'].empty else 'N/A'
    top_ailment = df['ailment'].value_counts().index[0] if 'ailment' in df.columns and not df['ailment'].empty else 'N/A'
    top_provider = df['prov_name'].value_counts().index[0] if 'prov_name' in df.columns and not df['prov_name'].empty else 'N/A'
    
    return {
        'total_claims': total_claims,
        'total_amount': total_amount,
        'avg_claim': avg_claim,
        'unique_members': unique_members,
        'unique_providers': unique_providers,
        'unique_services': unique_services,
        'unique_ailments': unique_ailments,
        'claims_per_day': claims_per_day,
        'amount_per_day': amount_per_day,
        'claims_change': claims_change,
        'amount_change': amount_change,
        'date_range_days': days_in_period,
        'start_date': df['datetime'].min().strftime('%Y-%m-%d'),
        'end_date': df['datetime'].max().strftime('%Y-%m-%d'),
        'top_category': top_category,
        'top_ailment': top_ailment,
        'top_provider': top_provider
    }

def calculate_temporal_insights(df, time_unit='M'):
    """Calculate advanced temporal insights and return as dictionary"""
    insights = {}
    
    if df.empty or 'datetime' not in df.columns:
        return insights
    
    # Resample data based on time unit
    resample_map = {
        'D': 'D',  # Daily
        'W': 'W',  # Weekly
        'M': 'M',  # Monthly
        'Q': 'Q',  # Quarterly
        'Y': 'Y'   # Yearly
    }
    
    resample_period = resample_map.get(time_unit, 'M')
    
    # Time series analysis
    time_series = df.set_index('datetime').sort_index()
    claims_count = time_series.resample(resample_period).size()
    claims_amount = time_series.resample(resample_period)['amount'].sum()
    
    # Growth rates
    claims_growth = claims_count.pct_change() * 100
    amount_growth = claims_amount.pct_change() * 100
    
    # Volatility (standard deviation of growth rates)
    insights['claims_volatility'] = claims_growth.std() if len(claims_growth) > 1 else 0
    insights['amount_volatility'] = amount_growth.std() if len(amount_growth) > 1 else 0
    
    # Seasonality analysis
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['quarter'] = df['datetime'].dt.quarter
    df['year'] = df['datetime'].dt.year
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    
    # Day of week patterns
    dow_patterns = df.groupby('day_of_week')['amount'].agg(['sum', 'count', 'mean']).reset_index()
    dow_patterns['day_name'] = dow_patterns['day_of_week'].apply(lambda x: calendar.day_name[x])
    insights['dow_patterns'] = dow_patterns.to_dict('records')
    
    # Monthly patterns
    monthly_patterns = df.groupby('month')['amount'].agg(['sum', 'count', 'mean']).reset_index()
    monthly_patterns['month_name'] = monthly_patterns['month'].apply(lambda x: calendar.month_name[x])
    insights['monthly_patterns'] = monthly_patterns.to_dict('records')
    
    # Yearly patterns
    yearly_patterns = df.groupby('year')['amount'].agg(['sum', 'count', 'mean']).reset_index()
    insights['yearly_patterns'] = yearly_patterns.to_dict('records')
    
    # Peak detection
    peak_claims_day = claims_count.idxmax() if not claims_count.empty else None
    peak_claims_value = claims_count.max() if not claims_count.empty else 0
    peak_amount_day = claims_amount.idxmax() if not claims_amount.empty else None
    peak_amount_value = claims_amount.max() if not claims_amount.empty else 0
    
    insights['peak_claims'] = {
        'date': peak_claims_day,
        'value': peak_claims_value
    }
    
    insights['peak_amount'] = {
        'date': peak_amount_day,
        'value': peak_amount_value
    }
    
    # Trend analysis (linear regression slope)
    if len(claims_count) > 1:
        x = np.arange(len(claims_count))
        y_claims = claims_count.values
        y_amount = claims_amount.values
        
        # Calculate slopes
        slope_claims = np.polyfit(x, y_claims, 1)[0] if len(y_claims) > 1 else 0
        slope_amount = np.polyfit(x, y_amount, 1)[0] if len(y_amount) > 1 else 0
        
        insights['trend_slope_claims'] = slope_claims
        insights['trend_slope_amount'] = slope_amount
        insights['trend_direction_claims'] = 'upward' if slope_claims > 0 else 'downward' if slope_claims < 0 else 'stable'
        insights['trend_direction_amount'] = 'upward' if slope_amount > 0 else 'downward' if slope_amount < 0 else 'stable'
    
    # Anomaly detection (using z-score)
    if len(claims_count) > 2:
        z_scores_claims = (claims_count - claims_count.mean()) / claims_count.std()
        anomalies_claims = claims_count[abs(z_scores_claims) > 2]
        insights['anomalies_claims'] = anomalies_claims.to_dict()
    
    if len(claims_amount) > 2:
        z_scores_amount = (claims_amount - claims_amount.mean()) / claims_amount.std()
        anomalies_amount = claims_amount[abs(z_scores_amount) > 2]
        insights['anomalies_amount'] = anomalies_amount.to_dict()
    
    # Cumulative analysis
    cumulative_claims = claims_count.cumsum()
    cumulative_amount = claims_amount.cumsum()
    insights['cumulative_claims'] = cumulative_claims.to_dict()
    insights['cumulative_amount'] = cumulative_amount.to_dict()
    
    # Statistical tests
    if len(claims_count) > 30:
        # Test for stationarity (Augmented Dickey-Fuller test)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(claims_count.dropna())
            insights['adf_statistic'] = adf_result[0]
            insights['adf_pvalue'] = adf_result[1]
            insights['is_stationary'] = adf_result[1] < 0.05
        except:
            pass
    
    # Forecasting preparation (last 20% for validation)
    if len(claims_count) > 5:
        split_idx = int(len(claims_count) * 0.8)
        insights['train_test_split'] = {
            'train_size': split_idx,
            'test_size': len(claims_count) - split_idx
        }
    
    return insights

def generate_temporal_visualizations(df, time_unit, analysis_type, focus_area):
    """Generate all temporal visualizations based on parameters"""
    visualizations = {}
    
    # Resample data based on time unit
    resample_map = {'D': 'D', 'W': 'W', 'M': 'M', 'Q': 'Q', 'Y': 'Y'}
    resample_period = resample_map.get(time_unit, 'M')
    
    time_series = df.set_index('datetime').sort_index()
    claims_count = time_series.resample(resample_period).size().reset_index(name='count')
    claims_amount = time_series.resample(resample_period)['amount'].sum().reset_index(name='total_amount')
    
    # 1. Overview visualizations (always shown)
    if focus_area in ['overview', 'all']:
        # Dual-axis time series chart
        fig_time = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_time.add_trace(
            go.Bar(
                x=claims_count['datetime'], 
                y=claims_count['count'],
                name='Claim Count',
                marker_color='#1e3a8a',
                opacity=0.7
            ),
            secondary_y=False
        )
        
        fig_time.add_trace(
            go.Scatter(
                x=claims_amount['datetime'], 
                y=claims_amount['total_amount'],
                name='Claim Amount',
                line=dict(color='#e30613', width=3),
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        fig_time.update_layout(
            title=f'Claims Trend Over Time ({time_unit})',
            xaxis_title='Date',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        fig_time.update_yaxes(title_text="Claim Count", secondary_y=False)
        fig_time.update_yaxes(title_text="Claim Amount (KES)", secondary_y=True)
        
        visualizations['time_series_combo'] = fig_time.to_html(full_html=False)
        
        # Individual amount chart
        fig_amount = px.area(
            claims_amount, 
            x='datetime', 
            y='total_amount',
            title=f'Total Claim Amount Over Time ({time_unit})',
            labels={'datetime': 'Date', 'total_amount': 'Amount (KES)'}
        )
        fig_amount.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450)
        fig_amount.update_traces(
            line=dict(color='#1e3a8a'), 
            fillcolor='rgba(30, 58, 138, 0.1)'
        )
        visualizations['amount_trend'] = fig_amount.to_html(full_html=False)
        
        # Individual count chart
        fig_count = px.bar(
            claims_count, 
            x='datetime', 
            y='count',
            title=f'Claim Count Over Time ({time_unit})',
            labels={'datetime': 'Date', 'count': 'Number of Claims'}
        )
        fig_count.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450)
        fig_count.update_traces(marker_color='#10b981')
        visualizations['count_trend'] = fig_count.to_html(full_html=False)
    
    # 2. Seasonality Analysis
    if focus_area in ['seasonality', 'all'] and analysis_type in ['comprehensive', 'advanced']:
        # Day of week analysis
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['month'] = df['datetime'].dt.month_name()
        df['year'] = df['datetime'].dt.year
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df['day_of_week'].value_counts().reindex(day_order).reset_index()
        day_counts.columns = ['day', 'count']
        
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        month_counts = df['month'].value_counts().reindex(month_order).reset_index()
        month_counts.columns = ['month', 'count']
        
        # Seasonality subplot
        fig_seasonality = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Claims by Day of Week', 'Claims by Month'),
            horizontal_spacing=0.15
        )
        
        fig_seasonality.add_trace(
            go.Bar(
                x=day_counts['day'], 
                y=day_counts['count'],
                name='Day of Week',
                marker_color='#f59e0b'
            ),
            row=1, col=1
        )
        
        fig_seasonality.add_trace(
            go.Bar(
                x=month_counts['month'], 
                y=month_counts['count'],
                name='Month',
                marker_color='#8b5cf6'
            ),
            row=1, col=2
        )
        
        fig_seasonality.update_layout(
            height=450,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        visualizations['seasonality_patterns'] = fig_seasonality.to_html(full_html=False)
        
        # Year-over-year comparison
        if 'year' in df.columns and df['year'].nunique() > 1:
            yearly_comparison = df.groupby(['year', 'month']).agg(
                total_amount=('amount', 'sum'),
                claim_count=('amount', 'count')
            ).reset_index()
            
            fig_yoy = px.line(
                yearly_comparison, 
                x='month', 
                y='total_amount',
                color='year',
                title='Year-over-Year Comparison by Month',
                labels={'month': 'Month', 'total_amount': 'Total Amount (KES)', 'year': 'Year'}
            )
            fig_yoy.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450)
            visualizations['yearly_comparison'] = fig_yoy.to_html(full_html=False)
            
            # Heatmap of claims by month and year
            heatmap_data = df.groupby([df['datetime'].dt.year, df['datetime'].dt.month]).size().unstack().fillna(0)
            fig_heatmap = px.imshow(
                heatmap_data,
                labels=dict(x="Month", y="Year", color="Claims"),
                x=[calendar.month_abbr[i] for i in range(1, 13)],
                y=heatmap_data.index,
                title="Claims Heatmap by Year and Month",
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig_heatmap.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450)
            visualizations['year_month_heatmap'] = fig_heatmap.to_html(full_html=False)
    
    # 3. Trend Analysis
    if focus_area in ['trends', 'all'] and analysis_type in ['advanced']:
        # Calculate moving averages
        daily_amount = df.set_index('datetime')['amount'].resample('D').sum()
        moving_avg_7 = daily_amount.rolling(window=7).mean()
        moving_avg_30 = daily_amount.rolling(window=30).mean()
        
        fig_moving_avg = go.Figure()
        
        fig_moving_avg.add_trace(go.Scatter(
            x=daily_amount.index, y=daily_amount.values,
            name='Daily Amount',
            line=dict(color='rgba(0,0,0,0.3)', width=1),
            opacity=0.6
        ))
        
        fig_moving_avg.add_trace(go.Scatter(
            x=moving_avg_7.index, y=moving_avg_7.values,
            name='7-Day Moving Avg',
            line=dict(color='#10b981', width=2)
        ))
        
        fig_moving_avg.add_trace(go.Scatter(
            x=moving_avg_30.index, y=moving_avg_30.values,
            name='30-Day Moving Avg',
            line=dict(color='#e30613', width=3)
        ))
        
        fig_moving_avg.update_layout(
            title='Moving Average Analysis',
            xaxis_title='Date',
            yaxis_title='Amount (KES)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450
        )
        
        visualizations['moving_averages'] = fig_moving_avg.to_html(full_html=False)
        
        # Exponential smoothing
        if len(daily_amount) > 30:
            exp_smoothing = daily_amount.ewm(span=30).mean()
            
            fig_exp_smoothing = go.Figure()
            fig_exp_smoothing.add_trace(go.Scatter(
                x=daily_amount.index, y=daily_amount.values,
                name='Daily Amount',
                line=dict(color='rgba(0,0,0,0.3)', width=1),
                opacity=0.6
            ))
            fig_exp_smoothing.add_trace(go.Scatter(
                x=exp_smoothing.index, y=exp_smoothing.values,
                name='Exponential Smoothing (30 days)',
                line=dict(color='#3b82f6', width=3)
            ))
            fig_exp_smoothing.update_layout(
                title='Exponential Smoothing Trend',
                xaxis_title='Date',
                yaxis_title='Amount (KES)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=450
            )
            visualizations['exponential_smoothing'] = fig_exp_smoothing.to_html(full_html=False)
            
            # Growth rate analysis
            growth_rate = claims_amount.set_index('datetime')['total_amount'].pct_change().dropna()
            fig_growth = px.line(
                x=growth_rate.index, 
                y=growth_rate.values,
                title='Growth Rate Over Time',
                labels={'x': 'Date', 'y': 'Growth Rate'}
            )
            fig_growth.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450)
            fig_growth.add_hline(y=0, line_dash="dash", line_color="red")
            visualizations['growth_rate'] = fig_growth.to_html(full_html=False)
    
    # 4. Category Analysis
    if focus_area in ['categories', 'all'] and analysis_type in ['comprehensive', 'advanced']:
        if 'benefit' in df.columns:
            category_df = df.groupby('benefit').agg(
                count=('benefit', 'size'),
                total_amount=('amount', 'sum')
            ).reset_index().sort_values('total_amount', ascending=False).head(10)
            
            fig_category = px.bar(
                category_df, 
                x='benefit', 
                y='total_amount',
                title='Top 10 Benefit Categories by Total Amount',
                labels={'benefit': 'Category', 'total_amount': 'Amount (KES)'},
                color='total_amount',
                color_continuous_scale='Blues'
            )
            
            fig_category.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_tickangle=-45,
                height=450
            )
            
            visualizations['category_analysis'] = fig_category.to_html(full_html=False)
        
        # Temporal patterns by category
        if 'benefit' in df.columns and analysis_type == 'advanced':
            top_categories = df['benefit'].value_counts().head(5).index.tolist()
            category_time = df[df['benefit'].isin(top_categories)].groupby(
                [pd.Grouper(key='datetime', freq=resample_period), 'benefit']
            )['amount'].sum().reset_index()
            
            fig_category_time = px.line(
                category_time,
                x='datetime',
                y='amount',
                color='benefit',
                title=f'Top 5 Categories Over Time ({time_unit})',
                labels={'datetime': 'Date', 'amount': 'Amount (KES)', 'benefit': 'Category'}
            )
            fig_category_time.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450)
            visualizations['category_trends'] = fig_category_time.to_html(full_html=False)
            
            # Sunburst chart for category hierarchy
            if 'ailment' in df.columns and 'benefit' in df.columns:
                # Sample data for sunburst to avoid performance issues
                sample_df = df.sample(min(5000, len(df)))
                fig_sunburst = px.sunburst(
                    sample_df, 
                    path=['ailment', 'benefit'], 
                    values='amount',
                    title='Claims Hierarchy: Ailment to Benefit'
                )
                fig_sunburst.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=500)
                visualizations['category_hierarchy'] = fig_sunburst.to_html(full_html=False)
    
    # 5. Member Analysis
    if focus_area in ['members', 'all'] and analysis_type in ['comprehensive', 'advanced']:
        if 'claim_me' in df.columns:
            claimant_stats = df.groupby('claim_me').agg(
                total_amount=('amount', 'sum'),
                claim_count=('claim_me', 'size'),
                avg_amount=('amount', 'mean'),
                first_claim=('datetime', 'min'),
                last_claim=('datetime', 'max')
            ).reset_index()
            
            top_claimants = claimant_stats.nlargest(10, 'total_amount')
            
            fig_claimants = px.bar(
                top_claimants, 
                x='claim_me', 
                y='total_amount',
                title='Top 10 Claimants by Total Amount',
                labels={'claim_me': 'Member ID', 'total_amount': 'Total Amount (KES)'},
                hover_data=['claim_count', 'avg_amount'],
                color='total_amount',
                color_continuous_scale='Viridis'
            )
            
            fig_claimants.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_tickangle=-45,
                height=450
            )
            
            visualizations['top_claimants'] = fig_claimants.to_html(full_html=False)
            
            # Format for table display
            top_claimants_table = top_claimants.copy()
            top_claimants_table['first_claim'] = top_claimants_table['first_claim'].dt.strftime('%Y-%m-%d')
            top_claimants_table['last_claim'] = top_claimants_table['last_claim'].dt.strftime('%Y-%m-%d')
            top_claimants_table['total_amount'] = top_claimants_table['total_amount'].round(2)
            top_claimants_table['avg_amount'] = top_claimants_table['avg_amount'].round(2)
            
            visualizations['top_claimants_table'] = top_claimants_table.to_dict('records')
        
        # Claim frequency distribution
        if 'claim_me' in df.columns and analysis_type in ['advanced']:
            claim_freq = df['claim_me'].value_counts().value_counts().sort_index()
            fig_freq = px.bar(
                x=claim_freq.index, 
                y=claim_freq.values,
                title='Claim Frequency Distribution',
                labels={'x': 'Number of Claims per Member', 'y': 'Number of Members'},
                color=claim_freq.values,
                color_continuous_scale='Reds'
            )
            
            fig_freq.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=450
            )
            
            visualizations['claim_frequency'] = fig_freq.to_html(full_html=False)
            
            # Member tenure analysis
            if 'claim_me' in df.columns:
                member_tenure = df.groupby('claim_me')['datetime'].agg(['min', 'max']).reset_index()
                member_tenure['tenure_days'] = (member_tenure['max'] - member_tenure['min']).dt.days
                
                fig_tenure = px.histogram(
                    member_tenure, 
                    x='tenure_days',
                    title='Distribution of Member Tenure (Days)',
                    labels={'tenure_days': 'Tenure (Days)', 'count': 'Number of Members'}
                )
                fig_tenure.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450)
                visualizations['member_tenure'] = fig_tenure.to_html(full_html=False)
    
    # 6. Provider Analysis
    if focus_area in ['providers', 'all'] and analysis_type in ['comprehensive', 'advanced']:
        if 'prov_name' in df.columns:
            provider_stats = df.groupby('prov_name').agg(
                total_amount=('amount', 'sum'),
                claim_count=('prov_name', 'size'),
                avg_amount=('amount', 'mean')
            ).reset_index().sort_values('total_amount', ascending=False).head(10)
            
            fig_providers = px.bar(
                provider_stats, 
                x='prov_name', 
                y='total_amount',
                title='Top 10 Providers by Total Amount',
                labels={'prov_name': 'Provider', 'total_amount': 'Total Amount (KES)'},
                hover_data=['claim_count', 'avg_amount'],
                color='total_amount',
                color_continuous_scale='Greens'
            )
            
            fig_providers.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_tickangle=-45,
                height=450
            )
            
            visualizations['top_providers'] = fig_providers.to_html(full_html=False)
            
            # Provider efficiency (amount per claim)
            provider_stats['efficiency'] = provider_stats['total_amount'] / provider_stats['claim_count']
            efficient_providers = provider_stats.nlargest(10, 'efficiency')
            
            fig_efficiency = px.bar(
                efficient_providers,
                x='prov_name',
                y='efficiency',
                title='Top 10 Providers by Efficiency (Amount per Claim)',
                labels={'prov_name': 'Provider', 'efficiency': 'Amount per Claim (KES)'},
                color='efficiency',
                color_continuous_scale='Purples'
            )
            fig_efficiency.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450)
            visualizations['provider_efficiency'] = fig_efficiency.to_html(full_html=False)
    
    # 7. Advanced Time Series Analysis
    if focus_area in ['advanced', 'all'] and analysis_type == 'advanced':
        # Only perform if we have enough data points
        if len(claims_amount) > 30:
            # Time series decomposition
            ts_data = claims_amount.set_index('datetime')['total_amount']
            
            # Determine period based on time unit
            if time_unit == 'M':
                period = 12  # Monthly data - yearly seasonality
            elif time_unit == 'Q':
                period = 4   # Quarterly data - yearly seasonality
            elif time_unit == 'W':
                period = 52  # Weekly data - yearly seasonality
            else:
                period = 7   # Default to weekly seasonality for daily data
            
            if len(ts_data) > 2 * period:  # Need at least 2 full periods
                try:
                    decomposition = seasonal_decompose(ts_data, period=period, model='additive')
                    
                    # Create subplots for decomposition
                    fig_decomp = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
                        vertical_spacing=0.08
                    )
                    
                    # Observed
                    fig_decomp.add_trace(
                        go.Scatter(x=ts_data.index, y=ts_data.values, name='Observed'),
                        row=1, col=1
                    )
                    
                    # Trend
                    fig_decomp.add_trace(
                        go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'),
                        row=2, col=1
                    )
                    
                    # Seasonal
                    fig_decomp.add_trace(
                        go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'),
                        row=3, col=1
                    )
                    
                    # Residual
                    fig_decomp.add_trace(
                        go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residual'),
                        row=4, col=1
                    )
                    
                    fig_decomp.update_layout(
                        height=600,
                        title_text=f"Time Series Decomposition ({time_unit})",
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    visualizations['time_decomposition'] = fig_decomp.to_html(full_html=False)
                except Exception as e:
                    print(f"Decomposition error: {e}")
            
            # Autocorrelation and Partial Autocorrelation
            try:
                nlags = min(40, len(ts_data) // 2)
                acf_values = acf(ts_data.dropna(), nlags=nlags)
                pacf_values = pacf(ts_data.dropna(), nlags=nlags)
                
                fig_acf = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Autocorrelation (ACF)', 'Partial Autocorrelation (PACF)'),
                    vertical_spacing=0.1
                )
                
                fig_acf.add_trace(
                    go.Bar(x=list(range(len(acf_values))), y=acf_values, name='ACF'),
                    row=1, col=1
                )
                
                fig_acf.add_trace(
                    go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name='PACF'),
                    row=2, col=1
                )
                
                # Add significance lines
                fig_acf.add_hline(y=1.96/np.sqrt(len(ts_data)), line_dash="dash", line_color="red", row=1, col=1)
                fig_acf.add_hline(y=-1.96/np.sqrt(len(ts_data)), line_dash="dash", line_color="red", row=1, col=1)
                fig_acf.add_hline(y=1.96/np.sqrt(len(ts_data)), line_dash="dash", line_color="red", row=2, col=1)
                fig_acf.add_hline(y=-1.96/np.sqrt(len(ts_data)), line_dash="dash", line_color="red", row=2, col=1)
                
                fig_acf.update_layout(
                    height=600,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                visualizations['autocorrelation'] = fig_acf.to_html(full_html=False)
            except Exception as e:
                print(f"ACF/PACF error: {e}")
            
            # Distribution analysis
            fig_dist = px.histogram(
                claims_amount, 
                x='total_amount',
                title='Distribution of Claim Amounts',
                labels={'total_amount': 'Claim Amount (KES)', 'count': 'Frequency'},
                marginal='box'
            )
            fig_dist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450)
            visualizations['amount_distribution'] = fig_dist.to_html(full_html=False)
            
            # Cumulative sum analysis
            cumulative_claims = claims_count['count'].cumsum()
            cumulative_amount = claims_amount['total_amount'].cumsum()
            
            fig_cumulative = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_cumulative.add_trace(
                go.Scatter(x=claims_count['datetime'], y=cumulative_claims, name='Cumulative Claims'),
                secondary_y=False
            )
            
            fig_cumulative.add_trace(
                go.Scatter(x=claims_amount['datetime'], y=cumulative_amount, name='Cumulative Amount'),
                secondary_y=True
            )
            
            fig_cumulative.update_layout(
                title='Cumulative Claims and Amount Over Time',
                xaxis_title='Date',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=450
            )
            
            fig_cumulative.update_yaxes(title_text="Cumulative Claims", secondary_y=False)
            fig_cumulative.update_yaxes(title_text="Cumulative Amount (KES)", secondary_y=True)
            
            visualizations['cumulative_analysis'] = fig_cumulative.to_html(full_html=False)
    
    # 8. Statistical Analysis
    if focus_area in ['statistics', 'all'] and analysis_type == 'advanced':
        # Box plots by time period
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        
        if df['year'].nunique() > 1:
            fig_box_year = px.box(
                df, 
                x='year', 
                y='amount',
                title='Claim Amount Distribution by Year',
                labels={'year': 'Year', 'amount': 'Claim Amount (KES)'}
            )
            fig_box_year.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450)
            visualizations['yearly_boxplot'] = fig_box_year.to_html(full_html=False)
        
        if df['month'].nunique() > 1:
            fig_box_month = px.box(
                df, 
                x='month', 
                y='amount',
                title='Claim Amount Distribution by Month',
                labels={'month': 'Month', 'amount': 'Claim Amount (KES)'}
            )
            fig_box_month.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450)
            visualizations['monthly_boxplot'] = fig_box_month.to_html(full_html=False)
        
        # Violin plots for distribution
        if df['year'].nunique() > 1:
            fig_violin = px.violin(
                df, 
                x='year', 
                y='amount',
                title='Claim Amount Distribution by Year (Violin Plot)',
                labels={'year': 'Year', 'amount': 'Claim Amount (KES)'},
                box=True
            )
            fig_violin.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450)
            visualizations['yearly_violin'] = fig_violin.to_html(full_html=False)
        
        # QQ plot for normality检验
        sample_amounts = df['amount'].sample(min(1000, len(df))).values
        theoretical_quantiles = stats.probplot(sample_amounts, dist="norm")
        
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theoretical_quantiles[0][0],
            y=theoretical_quantiles[0][1],
            mode='markers',
            name='Sample Quantiles'
        ))
        fig_qq.add_trace(go.Scatter(
            x=theoretical_quantiles[0][0],
            y=theoretical_quantiles[0][0] * theoretical_quantiles[1][0] + theoretical_quantiles[1][1],
            mode='lines',
            name='Theoretical Normal Distribution'
        ))
        fig_qq.update_layout(
            title='Q-Q Plot for Normality Test',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=450
        )
        visualizations['qq_plot'] = fig_qq.to_html(full_html=False)
    
    return visualizations

@csrf_exempt
@login_required(login_url='login')
def update_temporal_stats(request):
    """AJAX endpoint to update statistics based on filters"""
    if request.method == 'POST':
        try:
            dataset_id = request.POST.get('dataset_id')
            start_date = request.POST.get('start_date')
            end_date = request.POST.get('end_date')
            time_unit = request.POST.get('time_unit', 'M')
            
            if not dataset_id:
                return JsonResponse({'success': False, 'error': 'No dataset selected'})
            
            # Load and filter data
            df = pd.read_sql_query(f"SELECT * FROM {dataset_id}", connection)
            
            # Clean data
            df = clean_and_preprocess_data(df)
            
            # Apply filters
            df = apply_date_filters(df, start_date, end_date)
            
            if df.empty:
                return JsonResponse({'success': False, 'error': 'No data available for filters'})
            
            # Calculate statistics
            stats = calculate_summary_statistics(df)
            
            return JsonResponse({'success': True, 'stats': stats})
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

###########

########
#######
#####
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.db import connection
from django.http import JsonResponse
from django.contrib.humanize.templatetags.humanize import intcomma
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import math

@login_required(login_url='login')
def provider_efficiency_view(request):
    dataset_ids = get_database_tables()

    # --- Filters from GET request ---
    selected_id = request.GET.get('dataset_id')
    benefit_type = request.GET.get('benefit_type', 'all')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    provider_filter = request.GET.get('provider')
    fixed_cost = float(request.GET.get('fixed_cost', 50000))
    variable_rate = float(request.GET.get('variable_rate', 60)) / 100

    # --- Initialize chart and KPI variables ---
    chart_provider_amount = chart_processing_speed = chart_volume_avg = None
    chart_seasonality = chart_diversity_index = chart_variability = None
    chart_provider_map = chart_efficiency_matrix = chart_provider_efficiency = None
    provider_list = []
    benefit_types = []
    top_providers = []
    bottom_providers = []

    kpi_total_providers = kpi_total_claims = 0
    kpi_total_amount = kpi_avg_amount_claim = kpi_avg_amount_provider = 0.0
    kpi_avg_processing_days = kpi_fastest_days = kpi_slowest_days = 0
    kpi_providers_change = kpi_claims_change = kpi_amount_change = kpi_processing_change = 0.0

    efficiency_scores = {'cost_efficiency': 0, 'operational_efficiency': 0, 'service_quality': 0}
    efficiency_metrics = {'avg_claim_amount': 0, 'cost_per_member': 0, 'avg_processing_days': 0,
                          'claims_per_day': 0, 'service_diversity': 0, 'member_satisfaction': 0}

    if selected_id and selected_id in dataset_ids:
        try:
            df = pd.read_sql(f'SELECT * FROM "{selected_id}"', connection)
        except Exception as e:
            return render(request, 'myapp/provider_efficiency1.html', {
                'dataset_ids': dataset_ids,
                'selected_id': selected_id,
                'error': f"Error loading dataset: {e}"
            })

        # --- Standardize columns and clean data ---
        df.columns = df.columns.str.lower()
        df['amount'] = (df.get('amount', 0.0).astype(str)
                        .str.replace(r'[^\d.]', '', regex=True)
                        .replace('', np.nan)
                        .astype(float)
                        .fillna(0.0))

        date_col = next((col for col in df.columns if 'date' in col), None)
        if date_col:
            df['datetime'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
            df = df[df['datetime'].notna()]

        # Get benefit types for filter dropdown
        if 'benefit' in df.columns:
            benefit_types = sorted(df['benefit'].dropna().unique().tolist())

        # --- Apply filters ---
        if start_date and 'datetime' in df.columns:
            df = df[df['datetime'] >= pd.to_datetime(start_date)]
        if end_date and 'datetime' in df.columns:
            df = df[df['datetime'] <= pd.to_datetime(end_date)]
        if 'benefit' in df.columns and benefit_type != 'all':
            df = df[df['benefit'].astype(str).str.lower() == benefit_type.lower()]

        if 'prov_name' in df.columns:
            provider_list = sorted(df['prov_name'].dropna().unique())
            if provider_filter:
                df = df[df['prov_name'] == provider_filter]

        if not df.empty and 'prov_name' in df.columns:
            # --- Define current and previous periods ---
            current_start = pd.to_datetime(start_date) if start_date else df['datetime'].min()
            current_end = pd.to_datetime(end_date) if end_date else df['datetime'].max()
            period_length = current_end - current_start
            previous_end = current_start - timedelta(days=1)
            previous_start = previous_end - period_length

            current_df = df[(df['datetime'] >= current_start) & (df['datetime'] <= current_end)]
            previous_df = df[(df['datetime'] >= previous_start) & (df['datetime'] <= previous_end)]

            # --- Calculate KPIs ---
            kpi_total_providers = current_df['prov_name'].nunique()
            kpi_total_claims = len(current_df)
            kpi_total_amount = current_df['amount'].sum()

            member_col = next((col for col in ['claim_me', 'admit_id', 'member_id'] if col in current_df.columns), None)
            unique_members = current_df[member_col].nunique() if member_col else current_df.index.nunique()

            kpi_avg_amount_claim = kpi_total_amount / unique_members if unique_members else 0
            kpi_avg_amount_provider = kpi_total_amount / kpi_total_providers if kpi_total_providers else 0

            # --- Processing time calculations ---
            admission_col = next((col for col in ['admission_date', 'service_date', 'start_date'] if col in current_df.columns), None)
            if admission_col:
                current_df[admission_col] = pd.to_datetime(current_df[admission_col], errors='coerce')
                current_df = current_df[current_df[admission_col].notna()]
                current_df['processing_days'] = (current_df['datetime'] - current_df[admission_col]).dt.days
                current_df = current_df[current_df['processing_days'] >= 0]

                if not current_df.empty:
                    kpi_avg_processing_days = current_df['processing_days'].mean()
                    kpi_fastest_days = current_df['processing_days'].min()
                    kpi_slowest_days = current_df['processing_days'].max()

            # --- Calculate trend changes ---
            prev_providers = previous_df['prov_name'].nunique() if not previous_df.empty else kpi_total_providers
            prev_claims = len(previous_df) if not previous_df.empty else kpi_total_claims
            prev_amount = previous_df['amount'].sum() if not previous_df.empty else kpi_total_amount
            prev_processing = previous_df['processing_days'].mean() if 'processing_days' in previous_df.columns and not previous_df.empty else kpi_avg_processing_days

            kpi_providers_change = ((kpi_total_providers - prev_providers) / prev_providers * 100) if prev_providers else 0
            kpi_claims_change = ((kpi_total_claims - prev_claims) / prev_claims * 100) if prev_claims else 0
            kpi_amount_change = ((kpi_total_amount - prev_amount) / prev_amount * 100) if prev_amount else 0
            kpi_processing_change = ((kpi_avg_processing_days - prev_processing) / prev_processing * 100) if prev_processing else 0

            # --- Calculate efficiency scores ---
            efficiency_scores, efficiency_metrics = calculate_efficiency_scores(current_df)

            # --- Generate Break-even Analysis Chart ---
            provider_stats = current_df.groupby('prov_name').agg({
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
            overall_avg = current_df['amount'].mean()
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
            fig_break_even = px.scatter(
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
                    fig_break_even.add_trace(
                        go.Scatter(
                            x=np.linspace(1, max_claims, 100),
                            y=break_even_line,
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='Break-Even Line'
                        )
                    )
            
            chart_provider_efficiency = fig_break_even.to_html(full_html=False)
            
            # Prepare top/bottom providers
            provider_stats = provider_stats.sort_values('efficiency_score', ascending=False)
            top_providers = provider_stats.head(5).to_dict('records')
            bottom_providers = provider_stats.tail(5).to_dict('records')

            # --- Generate Other Charts ---
            # 1. Provider Claim Amount Ranking
            prov_sum = current_df.groupby('prov_name')['amount'].sum().reset_index()
            top10 = prov_sum.nlargest(10, 'amount').assign(Category='Top 10')
            bottom10 = prov_sum.nsmallest(10, 'amount').assign(Category='Bottom 10')
            combined = pd.concat([top10, bottom10])
            fig1 = px.bar(combined, x='amount', y='prov_name', orientation='h', color='Category',
                          title='Provider Claim Amount Ranking (Top & Bottom 10)',
                          color_discrete_map={'Top 10': '#10b981', 'Bottom 10': '#ef4444'},
                          labels={'amount': 'Total Amount (KES)', 'prov_name': 'Provider'})
            fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            chart_provider_amount = fig1.to_html(full_html=False)

            # 2. Claims Processing Speed
            if 'processing_days' in current_df.columns:
                speed_rank = current_df.groupby('prov_name')['processing_days'].mean().reset_index()
                fast10 = speed_rank.nsmallest(10, 'processing_days').assign(Category='Fastest')
                slow10 = speed_rank.nlargest(10, 'processing_days').assign(Category='Slowest')
                combined_speed = pd.concat([fast10, slow10])
                fig2 = px.bar(combined_speed, x='processing_days', y='prov_name', orientation='h',
                              color='Category', title='Claims Processing Speed (Fastest & Slowest 10)',
                              color_discrete_map={'Fastest': '#10b981', 'Slowest': '#ef4444'},
                              labels={'processing_days': 'Average Processing Days', 'prov_name': 'Provider'})
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                chart_processing_speed = fig2.to_html(full_html=False)

            # 3. Claims Volume vs Average Amount
            vol_vs_amt = current_df.groupby('prov_name').agg(total_claims=('amount', 'count'),
                                                             avg_amount=('amount', 'mean')).reset_index()
            fig3 = px.scatter(vol_vs_amt, x='total_claims', y='avg_amount', size='avg_amount',
                              hover_name='prov_name', title='Claims Volume vs Average Amount',
                              labels={'total_claims': 'Number of Claims', 'avg_amount': 'Average Amount (KES)'})
            fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            chart_volume_avg = fig3.to_html(full_html=False)

            # 4. Claims Seasonality
            if 'datetime' in current_df.columns:
                seasonality = current_df.groupby(current_df['datetime'].dt.to_period('M'))['amount'].sum().reset_index()
                seasonality['datetime'] = seasonality['datetime'].astype(str)
                fig4 = px.line(seasonality, x='datetime', y='amount', title='Claims Seasonality',
                              labels={'datetime': 'Month', 'amount': 'Total Amount (KES)'})
                fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                chart_seasonality = fig4.to_html(full_html=False)

            # 5. Provider Diversity Index
            service_col = next((col for col in ['service_code', 'benefit', 'procedure_code'] if col in current_df.columns), None)
            if service_col:
                diversity = current_df.groupby('prov_name')[service_col].nunique().reset_index()
                total_services = current_df[service_col].nunique()
                diversity['diversity_index'] = (diversity[service_col] / total_services) * 100
                fig5 = px.bar(diversity.nlargest(15, 'diversity_index'), x='prov_name', y='diversity_index',
                             title='Provider Service Diversity Index (Top 15)',
                             labels={'prov_name': 'Provider', 'diversity_index': 'Diversity Index (%)'})
                fig5.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                chart_diversity_index = fig5.to_html(full_html=False)

            # 6. Claim Amount Variability
            fig6 = px.box(current_df, x='prov_name', y='amount', points='all',
                         title='Claim Amount Variability by Provider',
                         labels={'prov_name': 'Provider', 'amount': 'Claim Amount (KES)'})
            fig6.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            chart_variability = fig6.to_html(full_html=False)

            # 7. Provider Efficiency Matrix
            agg_dict = {
                'total_claims': ('amount', 'count'),
                'total_amount': ('amount', 'sum')
            }
            if 'processing_days' in current_df.columns:
                agg_dict['avg_processing'] = ('processing_days', 'mean')

            efficiency_matrix = current_df.groupby('prov_name').agg(**agg_dict).reset_index()
            efficiency_matrix['efficiency_score'] = (
                (efficiency_matrix['total_claims'] / efficiency_matrix['total_claims'].max() * 40) +
                (1 - (efficiency_matrix['total_amount'] / efficiency_matrix['total_amount'].max()) * 30)
            )
            if 'avg_processing' in efficiency_matrix.columns:
                efficiency_matrix['efficiency_score'] += (1 - (efficiency_matrix['avg_processing'] / efficiency_matrix['avg_processing'].max()) * 30)
            else:
                efficiency_matrix['efficiency_score'] += 30

            fig7 = px.scatter(
                efficiency_matrix, x='total_claims', y='total_amount',
                size='efficiency_score', color='efficiency_score',
                hover_name='prov_name', title='Provider Efficiency Matrix',
                labels={'total_claims': 'Number of Claims', 'total_amount': 'Total Amount (KES)',
                        'efficiency_score': 'Efficiency Score'}
            )
            fig7.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            chart_efficiency_matrix = fig7.to_html(full_html=False)

            # 8. Provider Concentration Map
            location_col = next((col for col in ['claim_pod', 'location', 'region'] if col in current_df.columns), None)
            if location_col:
                map_data = current_df.groupby(location_col)['amount'].sum().reset_index()
                fig8 = px.choropleth(map_data, locations=location_col, locationmode='country names',
                                     color='amount', title='Provider Concentration by Location',
                                     labels={'amount': 'Total Amount (KES)'})
                fig8.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                chart_provider_map = fig8.to_html(full_html=False)

    # --- Format context for template ---
    def format_value(val, is_currency=False, is_days=False, decimal_places=0):
        if pd.isna(val) or val is None:
            return "—"
        if is_currency:
            return f"KES {val:,.{decimal_places}f}" if decimal_places else f"KES {int(val):,}"
        if is_days:
            return f"{val:.{decimal_places}f} days" if decimal_places else f"{int(val):,} days"
        return f"{val:,.{decimal_places}f}" if decimal_places else f"{int(val):,}"

    context = {
        'dataset_ids': dataset_ids,
        'selected_id': selected_id,
        'benefit_type': benefit_type,
        'benefit_types': benefit_types,
        'start_date': start_date,
        'end_date': end_date,
        'provider_filter': provider_filter,
        'provider_list': provider_list,
        'fixed_cost': fixed_cost,
        'variable_rate': variable_rate * 100,  # Convert back to percentage
        'kpi_total_providers': format_value(kpi_total_providers),
        'kpi_total_claims': format_value(kpi_total_claims),
        'kpi_total_amount': format_value(kpi_total_amount, is_currency=True),
        'kpi_avg_amount_claim': format_value(kpi_avg_amount_claim, is_currency=True, decimal_places=2),
        'kpi_avg_amount_provider': format_value(kpi_avg_amount_provider, is_currency=True, decimal_places=2),
        'kpi_avg_processing_days': format_value(kpi_avg_processing_days, is_days=True, decimal_places=1),
        'kpi_fastest_days': format_value(kpi_fastest_days, is_days=True),
        'kpi_slowest_days': format_value(kpi_slowest_days, is_days=True),
        'kpi_providers_change': kpi_providers_change,
        'kpi_claims_change': kpi_claims_change,
        'kpi_amount_change': kpi_amount_change,
        'kpi_processing_change': kpi_processing_change,
        'efficiency_scores': efficiency_scores,
        'efficiency_metrics': efficiency_metrics,
        'top_providers': top_providers,
        'bottom_providers': bottom_providers,
        'chart_provider_efficiency': chart_provider_efficiency,
        'chart_provider_amount': chart_provider_amount,
        'chart_processing_speed': chart_processing_speed,
        'chart_volume_avg': chart_volume_avg,
        'chart_seasonality': chart_seasonality,
        'chart_diversity_index': chart_diversity_index,
        'chart_variability': chart_variability,
        'chart_efficiency_matrix': chart_efficiency_matrix,
        'chart_provider_map': chart_provider_map
    }

    return render(request, 'myapp/provider_efficiency1.html', context)


# --- Utility functions ---
def calculate_efficiency_scores(df):
    if df.empty or 'prov_name' not in df.columns:
        return {'cost_efficiency': 0, 'operational_efficiency': 0, 'service_quality': 0}, \
               {'avg_claim_amount': 0, 'cost_per_member': 0, 'avg_processing_days': 0,
                'claims_per_day': 0, 'service_diversity': 0, 'member_satisfaction': 0}

    metrics = {}
    metrics['avg_claim_amount'] = df['amount'].mean() if 'amount' in df.columns else 0

    member_col = next((col for col in ['claim_me', 'admit_id', 'member_id'] if col in df.columns), None)
    unique_members = df[member_col].nunique() if member_col else df.index.nunique()
    metrics['cost_per_member'] = df['amount'].sum() / unique_members if unique_members else 0

    metrics['avg_processing_days'] = df['processing_days'].mean() if 'processing_days' in df.columns else 0
    metrics['claims_per_day'] = len(df) / max((df['datetime'].max() - df['datetime'].min()).days, 1) if 'datetime' in df.columns else 0

    service_col = next((col for col in ['service_code', 'benefit', 'procedure_code'] if col in df.columns), None)
    metrics['service_diversity'] = df[service_col].nunique() if service_col else 0

    processing_score = 1 - min(metrics['avg_processing_days'] / 30, 1) if metrics['avg_processing_days'] > 0 else 0.5
    amount_score = 1 - min(metrics['avg_claim_amount'] / 100000, 1) if metrics['avg_claim_amount'] > 0 else 0.5
    metrics['member_satisfaction'] = (processing_score * 0.6 + amount_score * 0.4) * 10

    # Efficiency scores (0-100)
    scores = {
        'cost_efficiency': max(0, min(100, 100 * (1 - min(metrics['avg_claim_amount'] / 50000, 1)))),
        'operational_efficiency': max(0, min(100, 100 * (1 - min(metrics['avg_processing_days'] / 30, 1)))) if metrics['avg_processing_days'] else 70,
        'service_quality': (min(metrics['service_diversity'] / 20 * 100, 100) * 0.4 + metrics['member_satisfaction'] * 0.6)
    }

    return scores, metrics


def get_database_tables():
    """
    Return all non-internal tables from the database.
    Supports SQLite and PostgreSQL.
    """
    vendor = connection.vendor  # 'sqlite', 'postgresql', etc.

    with connection.cursor() as cursor:
        if vendor == 'sqlite':
            cursor.execute("""
                SELECT name 
                FROM sqlite_master 
                WHERE type='table'
                  AND name NOT LIKE 'sqlite_%'
                  AND name NOT LIKE 'django_%'
                  AND name NOT LIKE 'auth_%'
                  AND name NOT LIKE 'sessions%'
            """)
        elif vendor == 'postgresql':
            cursor.execute("""
                SELECT tablename 
                FROM pg_catalog.pg_tables
                WHERE schemaname = 'public'
                  AND tablename NOT LIKE 'django_%'
                  AND tablename NOT LIKE 'auth_%'
                  AND tablename NOT LIKE 'sessions%'
            """)
        else:
            raise NotImplementedError(
                f"Database vendor '{vendor}' is not supported yet."
            )

        return [row[0] for row in cursor.fetchall()]
##########

###############

############

import networkx as nx


########
@login_required(login_url='login')
def diagnostic_patterns_view1(request):
    # ✅ Get available datasets
    dataset_ids = get_database_tables()

    # ✅ User-selected filters
    selected_id = request.GET.get('dataset_id')
    benefit_type = request.GET.get('benefit_type', 'all')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    ailment_filter = request.GET.get('ailment')  # 🔄 was diagnosis_filter
    gender_filter = request.GET.get('gender')
    min_amount = request.GET.get('min_amount')
    max_amount = request.GET.get('max_amount')
    age_range = request.GET.get('age_range')
    dependent_type = request.GET.get('dependent_type', '')

    # Chart placeholders - REORDERED to prioritize ailment-treatment patterns
    charts = {
        'chart_ailment_treatment_patterns': None,  # 🔥 PRIORITY CHART - FIRST
        'chart1_top_ailments': None,
        'chart2_age_distribution': None,
        'chart3_ailment_network': None,
        'chart4_gender_ailment_heatmap': None,
        'chart5_avg_amount_age_group': None,
        'chart6_ailment_seasonality': None,
        'chart7_claim_dist_by_dependents': None,
        'chart8_ailment_by_provider': None,
        'chart9_chronic_acute': None,
        'chart10_cost_outliers': None,
        'chart11_ailment_trends': None
    }

    # Dropdown filter lists
    ailment_list = []
    gender_list = []
    benefit_types = []
    dependent_types = []
    summary_stats = None
    top_ailments_table = None

    # ✅ Only proceed if a dataset is selected and exists
    if selected_id and selected_id in dataset_ids:
        # Load dataset from DB
        df = pd.read_sql(f'SELECT * FROM "{selected_id}"', connection)
        print(f"✅ Loaded dataset '{selected_id}' with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")

        # --- Standardize Amount ---
        amt_cols = [c for c in df.columns if c.lower() in ['amount', 'claim_amount']]
        if amt_cols:
            df['amount'] = pd.to_numeric(
                df[amt_cols[0]].astype(str).str.replace(r'[^\d.]', '', regex=True),
                errors='coerce'
            )
        else:
            df['amount'] = 0
        print(f"💰 Amount column stats: {df['amount'].describe()}")

        # --- Parse Dates ---
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        if date_cols:
            df['datetime'] = pd.to_datetime(df[date_cols[0]], errors='coerce', dayfirst=True)
            print(f"📅 Date sample: {df['datetime'].head()}")

        # --- Age Calculation ---
        if 'dob' in df.columns:
            df['dob'] = pd.to_datetime(df['dob'], errors='coerce', dayfirst=True)
            df['age'] = ((pd.Timestamp.now() - df['dob']).dt.days / 365.25).astype(int)
            print(f"👤 Age stats: {df['age'].describe()}")

        # --- Get filter options for dropdowns ---
        if 'benefit' in df.columns:
            benefit_types = sorted(df['benefit'].dropna().unique().tolist())
        
        if 'dependent_type' in df.columns:
            dependent_types = sorted(df['dependent_type'].dropna().unique().tolist())

        # --- Apply Filters ---
        if start_date:
            df = df[df['datetime'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['datetime'] <= pd.to_datetime(end_date)]

        # Benefit Type Filter
        if benefit_type != 'all' and 'benefit' in df.columns:
            df = df[df['benefit'].astype(str).str.lower() == benefit_type.lower()]

        # Amount filters
        if min_amount:
            df = df[df['amount'] >= float(min_amount)]
        if max_amount:
            df = df[df['amount'] <= float(max_amount)]

        # Age range filter
        if age_range and 'age' in df.columns:
            if age_range == '0-18':
                df = df[df['age'].between(0, 18)]
            elif age_range == '19-35':
                df = df[df['age'].between(19, 35)]
            elif age_range == '36-50':
                df = df[df['age'].between(36, 50)]
            elif age_range == '51-65':
                df = df[df['age'].between(51, 65)]
            elif age_range == '65+':
                df = df[df['age'] > 65]

        # Dependent type filter
        if dependent_type and 'dependent_type' in df.columns:
            df = df[df['dependent_type'] == dependent_type]

        # --- Ailment Column Detection ---
        ailment_cols = [c for c in df.columns if any(keyword in c.lower() for keyword in ['ailment'])]
        print(f"🩺 Ailment columns found: {ailment_cols}")

        if ailment_cols:
            ailment_list = sorted(df[ailment_cols[0]].dropna().unique().tolist())
            if ailment_filter:
                df = df[df[ailment_cols[0]] == ailment_filter]

        # --- Gender Filter ---
        if 'gender' in df.columns:
            gender_list = sorted(df['gender'].dropna().unique().tolist())
            if gender_filter:
                df = df[df['gender'] == gender_filter]

        print(f"📊 Data after filtering: {len(df)} rows remain")

        # --- Calculate Summary Statistics ---
        if len(df) > 0:
            summary_stats = {
                'total_claims': len(df),
                'total_amount': df['amount'].sum(),
                'unique_ailments': df[ailment_cols[0]].nunique() if ailment_cols else 0,
                'avg_claim': df['amount'].mean(),
                'top_ailment': df[ailment_cols[0]].mode().iloc[0] if ailment_cols and len(df[ailment_cols[0]].mode()) > 0 else 'N/A',
                'avg_age': df['age'].mean() if 'age' in df.columns else 0
            }

        # ---------------- CHARTS - REORDERED ---------------- #

        # 🔥 PRIORITY: Ailment-Treatment Patterns Heatmap
        if ailment_cols:
            treatment_cols = [c for c in df.columns if any(keyword in c.lower() 
                                for keyword in ['treatment', 'procedure', 'service', 'therapy', 'medication', 'claim'])]

            
            if treatment_cols:
                dt_df = df.groupby([ailment_cols[0], treatment_cols[0]]).size().reset_index(name='frequency')
                top_ailments = df[ailment_cols[0]].value_counts().head(15).index
                top_treatments = df[treatment_cols[0]].value_counts().head(20).index
                dt_filtered = dt_df[(dt_df[ailment_cols[0]].isin(top_ailments)) & (dt_df[treatment_cols[0]].isin(top_treatments))]
                
                if len(dt_filtered) > 0:
                    fig_dt = px.density_heatmap(
                        dt_filtered, x=treatment_cols[0], y=ailment_cols[0], z='frequency',
                        title="Ailment-Treatment Patterns", color_continuous_scale='Blues', height=600
                    )
                    fig_dt.update_layout(xaxis_title="Treatment/Procedure", yaxis_title="Ailment", xaxis={'tickangle': 45})
                    charts['chart_ailment_treatment_patterns'] = fig_dt.to_html(full_html=False)

        # 1️⃣ Top Ailments by Claim Amount
        if ailment_cols:
            top_ailment = df.groupby(ailment_cols[0]).agg({'amount': ['sum', 'count', 'mean']}).reset_index()
            top_ailment.columns = ['ailment', 'total_amount', 'claim_count', 'avg_amount']
            top_ailment['percentage'] = (top_ailment['total_amount'] / top_ailment['total_amount'].sum()) * 100
            top_ailment = top_ailment.nlargest(15, 'total_amount').sort_values('total_amount')
            
            top_ailments_table = top_ailment.nlargest(20, 'total_amount').to_dict('records')
            fig1 = px.bar(top_ailment, x='total_amount', y='ailment', orientation='h',
                          title="Top 15 Ailments by Claim Amount",
                          color='total_amount', color_continuous_scale='Blues')
            fig1.update_layout(height=500)
            charts['chart1_top_ailments'] = fig1.to_html(full_html=False)

        # 2️⃣ Age Distribution
        if 'age' in df.columns:
            fig2 = px.histogram(df, x='age', nbins=20, title="Age Distribution of Claimants",
                               color_discrete_sequence=['#1e3a8a'])
            charts['chart2_age_distribution'] = fig2.to_html(full_html=False)

        # 3️⃣ Diagnosis Co-occurrence Network
        if ailment_cols:
            primary_diag_col = ailment_cols[0]
            patient_id_col = 'admit_id' if 'admit_id' in df.columns else df.index

            diag_pairs = (
                df.dropna(subset=[primary_diag_col])
                .groupby(patient_id_col)[primary_diag_col]
                .apply(lambda x: list(set(x)))
            )

            edges = []
            for diags in diag_pairs:
                if len(diags) > 1:
                    for i, a in enumerate(diags):
                        for b in diags[i+1:]:
                            edges.append((a, b))

            if edges:
                G = nx.Graph()
                G.add_edges_from(edges)

                pos = nx.spring_layout(G, k=0.5)
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                node_x, node_y, node_text = [], [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)

                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                        line=dict(width=0.5, color='#888')))
                fig3.add_trace(go.Scatter(
                    x=node_x, y=node_y, mode='markers+text',
                    text=node_text, textposition='top center',
                    marker=dict(size=10, color='#e30613')
                ))
                fig3.update_layout(title="Diagnosis Co-occurrence Network", showlegend=False)
                charts['chart3_diag_network'] = fig3.to_html(full_html=False)

        # 4️⃣ Gender × Diagnosis Heatmap
        if ailment_cols and 'gender' in df.columns:
            heat_df = df.groupby(['gender', ailment_cols[0]]).size().reset_index(name='count')
            top_diags_gender = df[ailment_cols[0]].value_counts().head(10).index
            heat_df = heat_df[heat_df[ailment_cols[0]].isin(top_diags_gender)]
            
            fig4 = px.density_heatmap(heat_df, x='gender', y=ailment_cols[0], z='count',
                                      color_continuous_scale='Blues',
                                      title="Gender × Diagnosis Distribution")
            charts['chart4_gender_diag_heatmap'] = fig4.to_html(full_html=False)

        # 5️⃣ Average Claim Amount by Age Group
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100], 
                                    labels=['0-18', '19-35', '36-50', '51-65', '65+'])
            age_avg = df.groupby('age_group')['amount'].mean().reset_index()
            
            fig5 = px.bar(age_avg, x='age_group', y='amount', 
                         title="Average Claim Amount by Age Group",
                         color='amount', color_continuous_scale='Blues')
            charts['chart5_avg_amount_age_group'] = fig5.to_html(full_html=False)

        # 6️⃣ Diagnosis Seasonality
        if ailment_cols and 'datetime' in df.columns:
            df['month'] = df['datetime'].dt.month_name()
            monthly_diag = df.groupby(['month', ailment_cols[0]]).size().reset_index(name='count')
            top_diags_season = df[ailment_cols[0]].value_counts().head(8).index
            monthly_diag = monthly_diag[monthly_diag[ailment_cols[0]].isin(top_diags_season)]
            
            fig6 = px.line(monthly_diag, x='month', y='count', color=ailment_cols[0],
                          title="Diagnosis Seasonality Trends",
                          category_orders={"month": ["January", "February", "March", "April", "May", "June",
                                                   "July", "August", "September", "October", "November", "December"]})
            charts['chart6_diag_seasonality'] = fig6.to_html(full_html=False)

        # 7️⃣ Claim Distribution by Dependent Type
        if ailment_cols and 'dependent_type' in df.columns:
            dep_diag = df.groupby(['dependent_type', ailment_cols[0]]).size().reset_index(name='count')
            top_diags_dep = df[ailment_cols[0]].value_counts().head(10).index
            dep_diag = dep_diag[dep_diag[ailment_cols[0]].isin(top_diags_dep)]
            
            fig7 = px.sunburst(dep_diag, path=['dependent_type', ailment_cols[0]], values='count',
                              title="Claim Distribution by Dependent Type & Diagnosis")
            charts['chart7_claim_dist_by_dependents'] = fig7.to_html(full_html=False)

        # 8️⃣ Diagnosis by Provider Type
        if ailment_cols and 'prov_name' in df.columns:
            prov_diag = df.groupby(['prov_name', ailment_cols[0]]).size().reset_index(name='count')
            top_provs = df['prov_name'].value_counts().head(5).index
            top_diags_prov = df[ailment_cols[0]].value_counts().head(10).index
            prov_diag = prov_diag[
                (prov_diag['prov_name'].isin(top_provs)) & 
                (prov_diag[ailment_cols[0]].isin(top_diags_prov))
            ]
            
            fig8 = px.bar(prov_diag, x='prov_name', y='count', color=ailment_cols[0],
                         title="Diagnosis Distribution by Provider")
            charts['chart8_diag_by_provider'] = fig8.to_html(full_html=False)

        # 9️⃣ Chronic vs Acute Conditions (simplified)
        if ailment_cols:
            # Simple classification based on diagnosis name
            chronic_keywords = ['chronic', 'diabetes', 'hypertension', 'asthma', 'arthritis', 'heart disease']
            df['condition_type'] = df[ailment_cols[0]].apply(
                lambda x: 'Chronic' if any(keyword in str(x).lower() for keyword in chronic_keywords) else 'Acute'
            )
            
            cond_type = df['condition_type'].value_counts().reset_index()
            fig9 = px.pie(cond_type, values='count', names='condition_type',
                         title="Chronic vs Acute Conditions Distribution")
            charts['chart9_chronic_acute'] = fig9.to_html(full_html=False)

        # 🔟 Diagnosis Cost Outliers
        if ailment_cols:
            # Calculate outliers using IQR method
            diag_stats = df.groupby(ailment_cols[0])['amount'].agg(['mean', 'std', 'count']).reset_index()
            diag_stats = diag_stats[diag_stats['count'] > 5]  # Only consider diagnoses with enough data
            
            # Identify outliers (mean + 2*std)
            diag_stats['upper_bound'] = diag_stats['mean'] + 2 * diag_stats['std']
            
            # Get top 10 diagnoses with highest outlier potential
            diag_stats = diag_stats.nlargest(10, 'upper_bound')
            
            fig10 = px.bar(diag_stats, x=ailment_cols[0], y='upper_bound',
                          title="Diagnosis Cost Outlier Potential (Mean + 2σ)",
                          color='upper_bound', color_continuous_scale='Reds')
            charts['chart10_cost_outliers'] = fig10.to_html(full_html=False)

        # 1️⃣1️⃣ Diagnosis Trend Over Time
        if ailment_cols and 'datetime' in df.columns:
            df['month_year'] = df['datetime'].dt.to_period('M').astype(str)
            top_diags_trend = df[ailment_cols[0]].value_counts().head(5).index
            trend_data = df[df[ailment_cols[0]].isin(top_diags_trend)]
            trend_data = trend_data.groupby(['month_year', ailment_cols[0]]).size().reset_index(name='count')
            
            fig11 = px.line(trend_data, x='month_year', y='count', color=ailment_cols[0],
                           title="Top 5 Diagnosis Trends Over Time")
            charts['chart11_diag_trends'] = fig11.to_html(full_html=False)

    # ✅ Render with reordered context
    return render(request, 'myapp/diagnostic-patterns1.html', {
        'dataset_ids': dataset_ids,
        'selected_id': selected_id,
        'benefit_type': benefit_type,
        'start_date': start_date,
        'end_date': end_date,
        'ailment_list': ailment_list,
        'gender_list': gender_list,
        'benefit_types': benefit_types,
        'dependent_types': dependent_types,
        'ailment_filter': ailment_filter,
        'gender_filter': gender_filter,
        'min_amount': min_amount,
        'max_amount': max_amount,
        'age_range': age_range,
        'dependent_type': dependent_type,
        'summary_stats': summary_stats,
        'top_ailments_table': top_ailments_table,
        **charts
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
def minet_claim_prediction(request):
    """View for Minet claims prediction page"""
    return render(request, 'myapp/minet_claim_prediction.html')

@login_required
def minet_forecast_volume(request):
    """View for Minet forecast volume page"""
    return render(request, 'myapp/minet_forecast_volume.html')

@login_required
def minet_confidence_interval(request):
    """View for Minet confidence interval page with dataset-driven confidence interval calculation"""
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from django.utils import timezone
    from datetime import timedelta

    context = {
        'selected_confidence': 95,
        'avg_ci_range': None,
        'outliers': [],
        'outlier_count': 0,
        'visualizations': {}
    }

    try:
        # Get confidence level from query param
        selected_confidence = int(request.GET.get('confidence', 95))
        z_values = {90: 1.645, 95: 1.96, 99: 2.576}
        z_value = z_values.get(selected_confidence, 1.96)
        context['selected_confidence'] = selected_confidence

        # Load claims data from database
        qs = ClaimRecord.objects.values('claim_prov_date', 'amount')
        df = pd.DataFrame.from_records(qs)

        if df.empty:
            context['error'] = "No claims data found."
            return render(request, 'myapp/minet_confidence_interval.html', context)

        # Ensure correct data types
        df['date'] = pd.to_datetime(df['claim_prov_date'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['date', 'amount'])

        if df.empty:
            context['error'] = "No valid claims data with dates and amounts."
            return render(request, 'myapp/minet_confidence_interval.html', context)

        # Aggregate by month
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

        # Calculate Confidence Intervals
        monthly_data['ci_lower'] = monthly_data['avg_amount'] - z_value * monthly_data['std_amount'] / np.sqrt(monthly_data['claim_count'])
        monthly_data['ci_upper'] = monthly_data['avg_amount'] + z_value * monthly_data['std_amount'] / np.sqrt(monthly_data['claim_count'])

        # Average CI Range (for stat card)
        monthly_data['ci_range'] = monthly_data['ci_upper'] - monthly_data['ci_lower']
        avg_ci_range = monthly_data['ci_range'].mean()
        context['avg_ci_range'] = f"{avg_ci_range:,.0f}"  # formatted KES

        # Outlier Detection (last 12 months)
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

        # Create Plotly Chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=monthly_data['date'],
            y=monthly_data['total_amount'],
            mode='lines+markers',
            name='Total Claims',
            line=dict(color='var(--minet-primary)')
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
            fillcolor='rgba(0, 150, 200, 0.2)',
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
        import traceback
        traceback.print_exc()

    return render(request, 'myapp/minet_confidence_interval.html', context)

@login_required
def minet_impact_simulation(request):
    """View for Minet impact simulation page"""
    return render(request, 'myapp/minet_impact_simulation.html')

@login_required
def minet_explainability(request):
    """View for Minet explainability page"""
    return render(request, 'myapp/minet_explainability.html')

@login_required
def machine_learning(request):
    """View for Machine Learning page"""
    return render(request, 'myapp/machine_learning.html')

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
        # Include claim_ce in values() so it appears in df if you want to inspect it
        claims = ClaimRecord.objects.order_by('-claim_prov_date').values(
            'amount', 'claim_prov_date', 'benefit', 'benefit_desc', 'claim_me', 'prov_name', 'claim_ce'
        )

        # raw total rows
        raw_count = ClaimRecord.objects.count()
        context['visualizations']['debug']['raw_count'] = raw_count

        if raw_count == 0:
            context['visualizations']['debug']['error'] = "No records found in database"
            return render(request, 'myapp/safaricom_home.html', context)

        # ---- KEY CHANGE: use distinct count of claim_ce for total_claims ----
        unique_claims = ClaimRecord.objects.values('claim_ce').distinct().count()
        context['visualizations']['summary_stats']['total_claims'] = unique_claims
        # keep it visible for debugging too
        context['visualizations']['debug']['unique_claims'] = unique_claims

        # Convert to DataFrame
        df = pd.DataFrame.from_records(claims)
        # store some claim_ce samples for debug
        if 'claim_ce' in df.columns:
            context['visualizations']['debug']['claim_ce_sample'] = df['claim_ce'].head(3).tolist()

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

        # Summary stats (unchanged)
        total_amount = ClaimRecord.objects.aggregate(total=Sum('amount'))['total'] or 0
        unique_members = ClaimRecord.objects.values('claim_me').distinct().count()
        avg_claim = total_amount / unique_members if unique_members > 0 else 0
        context['visualizations']['summary_stats'].update({
            'total_amount': float(total_amount),
            'avg_claim': float(avg_claim),
            'unique_members': unique_members
        })

        # ===== SINGLE INTERACTIVE CHART WITH TREND LINES =====
        if not df.empty:
            daily_df = df.resample('D').size().reset_index(name='count')
            weekly_df = df.resample('W-MON').size().reset_index(name='count')
            monthly_df = df.resample('M').size().reset_index(name='count')

            # Add trend lines for each aggregation
            def add_trend_line(data_df, period_name):
                # Convert datetime to numeric for linear regression
                x = np.arange(len(data_df))
                y = data_df['count'].values
                
                # Linear regression
                slope, intercept = np.polyfit(x, y, 1)
                trend_line = slope * x + intercept
                
                return go.Scatter(
                    x=data_df['datetime'], 
                    y=trend_line,
                    mode='lines',
                    name=f'{period_name} Trend',
                    line=dict(dash='dash', width=2),
                    visible=False
                )

            fig = go.Figure()

            # Daily trace
            fig.add_trace(go.Scatter(
                x=daily_df['datetime'], y=daily_df['count'],
                mode='lines+markers', name='Daily Claims', visible=True,
                line=dict(color='blue')
            ))
            fig.add_trace(add_trend_line(daily_df, 'Daily'))

            # Weekly trace
            fig.add_trace(go.Scatter(
                x=weekly_df['datetime'], y=weekly_df['count'],
                mode='lines+markers', name='Weekly Claims', visible=False,
                line=dict(color='orange')
            ))
            fig.add_trace(add_trend_line(weekly_df, 'Weekly'))

            # Monthly trace
            fig.add_trace(go.Scatter(
                x=monthly_df['datetime'], y=monthly_df['count'],
                mode='lines+markers', name='Monthly Claims', visible=False,
                line=dict(color='green')
            ))
            fig.add_trace(add_trend_line(monthly_df, 'Monthly'))

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
                             args=[{"visible": [True, True, False, False, False, False]},
                                   {"title": "Daily Claims Submitted"}]),
                        dict(label="Weekly",
                             method="update",
                             args=[{"visible": [False, False, True, True, False, False]},
                                   {"title": "Weekly Claims Submitted"}]),
                        dict(label="Monthly",
                             method="update",
                             args=[{"visible": [False, False, False, False, True, True]},
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

        # ===== LORENZ CHART (Wealth Distribution) =====
        if 'claim_me' in df.columns and 'amount' in df.columns:
            # Calculate cumulative distribution of claims by member
            member_totals = df.groupby('claim_me')['amount'].sum().sort_values()
            total_amount_all = member_totals.sum()
            
            # Calculate Lorenz curve data
            cumulative_members = np.arange(1, len(member_totals) + 1) / len(member_totals)
            cumulative_amount = member_totals.cumsum() / total_amount_all
            
            # Perfect equality line
            perfect_equality = np.linspace(0, 1, len(cumulative_members))
            
            # Calculate Gini coefficient
            gini = 1 - 2 * np.trapz(cumulative_amount, cumulative_members)
            
            lorenz_fig = go.Figure()
            
            # Lorenz curve
            lorenz_fig.add_trace(go.Scatter(
                x=cumulative_members, y=cumulative_amount,
                mode='lines', name='Lorenz Curve',
                line=dict(color='#1BB64F', width=3),
                fill='tonexty'
            ))
            
            # Perfect equality line
            lorenz_fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines', name='Perfect Equality',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            lorenz_fig.update_layout(
                title=f'Lorenz Curve (Gini Coefficient: {gini:.3f})',
                xaxis_title='Cumulative Proportion of Members',
                yaxis_title='Cumulative Proportion of Claims Amount',
                showlegend=True,
                margin=dict(l=20, r=20, t=60, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                annotations=[
                    dict(
                        x=0.5, y=0.5,
                        xref="paper", yref="paper",
                        text=f"Gini: {gini:.3f}",
                        showarrow=False,
                        font=dict(size=16, color="black"),
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1
                    )
                ]
            )
            
            context['visualizations']['lorenz_chart'] = lorenz_fig.to_html(full_html=False)

        # ===== BENEFIT CATEGORY CHART =====
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

        # ===== TOP PROVIDERS BY AMOUNT =====
        if 'prov_name' in df.columns:
            top_providers = df.groupby('prov_name').agg(
                total_amount=('amount', 'sum'),
                claim_count=('amount', 'count')
            ).nlargest(10, 'total_amount').reset_index()

            if not top_providers.empty:
                providers_fig = px.bar(
                    top_providers, x='prov_name', y='total_amount',
                    title='Top Providers by Amount',
                    labels={'prov_name': 'Provider', 'total_amount': 'Total Amount (KES)'},
                    color_discrete_sequence=['#1BB64F']
                )
                providers_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20),
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)')
                context['visualizations']['top_providers'] = providers_fig.to_html(full_html=False)

        # ===== TOP CLAIMANTS (TABLE ONLY, ANONYMIZED) =====
        if 'claim_me' in df.columns:
            top_claimants = df.groupby('claim_me').agg(
                total_amount=('amount', 'sum'),
                claim_count=('amount', 'count')
            ).nlargest(10, 'total_amount').reset_index()

            if not top_claimants.empty:
                # Anonymize Member IDs
                top_claimants['claim_me'] = top_claimants['claim_me'].apply(
                    lambda x: f"Member-{hash(x) % 10000}"
                )
                context['visualizations']['top_claimants_table'] = top_claimants.to_dict('records')

                freq = df['claim_me'].value_counts().reset_index()
                freq.columns = ['claim_me', 'count']
                freq['claim_me'] = freq['claim_me'].apply(lambda x: f"Member-{hash(x) % 10000}")
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

    # Clean claim_ce in case of spaces/nulls
    clean_claims = claims.exclude(claim_ce__isnull=True).annotate(
        claim_ce_trimmed=Trim('claim_ce')
    )

    total_amount = clean_claims.aggregate(total=Sum('amount'))['total'] or 0
    unique_claims = clean_claims.values('claim_ce_trimmed').distinct().count()  # ✅ truly unique
    unique_members = clean_claims.values('claim_me').distinct().count()
    unique_providers = clean_claims.values('prov_name').distinct().count()

    summary_stats = {
        'total_claims': unique_claims,
        'total_amount': total_amount,
        'avg_claim': total_amount / unique_claims if unique_claims > 0 else 0,
        'unique_members': unique_members,
        'unique_providers': unique_providers,
        'claims_per_member': unique_claims / unique_members if unique_members > 0 else 0,
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
            count=Count('claim_ce', distinct=True),
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
        claim_count=Count('claim_ce', distinct=True),
        total_amount=Sum('amount'),
        avg_amount=Avg('amount')
    ).order_by('-total_amount')

    # === 5. Provider Network Analysis ===
    provider_stats = claims.values('prov_name').annotate(
        claim_count=Count('claim_ce', distinct=True),
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
        claim_count=Count('claim_ce', distinct=True),
        total_amount=Sum('amount')
    ).order_by('day_of_week')

    # === 8. Additional Analyses ===
    gender_stats = claims.values('gender').annotate(
        claim_count=Count('claim_ce', distinct=True),
        total_amount=Sum('amount'),
        avg_amount=Avg('amount')
    ).order_by('-total_amount')

    monthly_trend = claims.annotate(
        month=ExtractMonth('claim_prov_date')
    ).values('month').annotate(
        claim_count=Count('claim_ce', distinct=True),
        total_amount=Sum('amount')
    ).order_by('month')

    top_ailments = claims.values('ailment').annotate(
        claim_count=Count('claim_ce', distinct=True),
        total_amount=Sum('amount')
    ).order_by('-claim_count')[:10]

    cost_center_analysis = claims.values('cost_center').annotate(
        claim_count=Count('claim_ce', distinct=True),
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

@login_required(login_url='login')
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
    
    # ---- Summary stats ----
    total_amount = data['amount'].sum() if 'amount' in data.columns else 0
    unique_members = data['claim_me'].nunique() if 'claim_me' in data.columns else 0
    total_claims = data['claim_ce'].nunique() if 'claim_ce' in data.columns else 0

    visualizations = {
        'summary_stats': {
            # ✅ total claims now based on unique claim_ce values
            'total_claims': total_claims,
            'total_amount': total_amount,
            # ✅ avg_claim = total amount ÷ unique members
            'avg_claim': (total_amount / unique_members) if unique_members > 0 else 0,
            'unique_members': unique_members,
            'unique_providers': data['prov_name'].nunique() if 'prov_name' in data.columns else 0,
            'claims_per_member': (total_claims / unique_members) if unique_members > 0 else 0,
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
    
    # ---- Visualization code (unchanged) ----
    if not data.empty and cat_cols:
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
    
    # Other visualizations remain unchanged...
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
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.utils import timezone
from datetime import timedelta, datetime
from django.db.models import Q, Sum, Count, Max, Min
from .models import ClaimRecord
import json
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

@login_required
def temporal_analysis(request):
    username = request.user.username
    context = {
        "username": username,
        "time_unit": request.GET.get("time_unit", "month"),
        "metric": request.GET.get("metric", "Total Amount"),
        "benefit_types": ClaimRecord.objects.values_list("benefit", flat=True).distinct().order_by("benefit"),
        "providers": ClaimRecord.objects.values_list("prov_name", flat=True).distinct().order_by("prov_name"),
        "cost_centers": ClaimRecord.objects.values_list("cost_center", flat=True).distinct().order_by("cost_center"),
        "error_message": None,
        "summary_stats": None,
        "applied_filters": {
            "time_period": request.GET.get("time_period", "all"),
            "benefit_type": request.GET.get("benefit_type", "all"),
            "provider": request.GET.get("provider", "all"),
            "cost_center": request.GET.get("cost_center", "all"),
        },
    }

    # Chart placeholders
    charts = [
        "temporal_chart", "cumulative_chart", "dow_chart", "month_chart",
        "hour_chart", "boxplot_chart", "categorical_chart", "heatmap_chart",
        "rolling_avg_chart", "providers_chart", "benefits_chart", "anomaly_chart",
        "decomposition_chart", "seasonality_chart", "correlation_chart",
        "weekly_pattern_chart", "trend_analysis_chart", "comparison_chart",
    ]
    for chart in charts:
        context[chart] = None

    try:
        # -----------------------------
        # Apply filters
        # -----------------------------
        time_period = request.GET.get("time_period", "all")
        benefit_type = request.GET.get("benefit_type", "all")
        provider = request.GET.get("provider", "all")
        cost_center = request.GET.get("cost_center", "all")

        claims = ClaimRecord.objects.all()

        # Time filter
        if time_period != "all":
            today = timezone.now().date()
            if time_period == "3m":
                start_date = today - timedelta(days=90)
                claims = claims.filter(claim_prov_date__gte=start_date)
            elif time_period == "6m":
                start_date = today - timedelta(days=180)
                claims = claims.filter(claim_prov_date__gte=start_date)
            elif time_period == "12m":
                start_date = today - timedelta(days=365)
                claims = claims.filter(claim_prov_date__gte=start_date)
            else:  # custom year
                try:
                    start_date = datetime(int(time_period), 1, 1).date()
                    end_date = datetime(int(time_period), 12, 31).date()
                    claims = claims.filter(claim_prov_date__range=[start_date, end_date])
                except Exception:
                    pass

        if benefit_type != "all":
            claims = claims.filter(benefit=benefit_type)
        if provider != "all":
            claims = claims.filter(prov_name=provider)
        if cost_center != "all":
            claims = claims.filter(cost_center=cost_center)

        # -----------------------------
        # Summary statistics
        # -----------------------------
        summary_data = claims.aggregate(
            total_claims=Count("claim_ce", distinct=True),  # ✅ unique claim_ce
            total_amount=Sum("amount"),
            max_amount=Max("amount"),
            min_amount=Min("amount"),
        )

        # Correct average claim
        if summary_data["total_amount"] and summary_data["total_claims"]:
            avg_amount = summary_data["total_amount"] / summary_data["total_claims"]
        else:
            avg_amount = 0

        context["summary_stats"] = {
            "total_claims": f"{summary_data['total_claims']:,}" if summary_data["total_claims"] else "0",
            "total_amount": f"KES {summary_data['total_amount']:,.2f}" if summary_data["total_amount"] else "KES 0.00",
            "avg_amount": f"KES {avg_amount:,.2f}" if avg_amount else "KES 0.00",
            "max_amount": f"KES {summary_data['max_amount']:,.2f}" if summary_data["max_amount"] else "KES 0.00",
            "min_amount": f"KES {summary_data['min_amount']:,.2f}" if summary_data["min_amount"] else "KES 0.00",
        }

        # -----------------------------
        # Build dataframe for charts
        # -----------------------------
        claims_data = claims.values(
            "id", "claim_prov_date", "amount", "prov_name", "benefit",
            "cost_center", "gender", "dependent_type", "ailment",
        )
        df = pd.DataFrame.from_records(claims_data)

        if df.empty:
            context["error_message"] = "No claims found matching your filters"
            return render(request, "myapp/temporal_analysis.html", context)

        # Ensure numeric amounts
        df["amount"] = pd.to_numeric(
            df["amount"].astype(str).str.replace(r"[^\d.]", "", regex=True).replace("", "0"),
            errors="coerce",
        ).fillna(0)

        df["datetime"] = pd.to_datetime(df["claim_prov_date"], errors="coerce")
        df = df.dropna(subset=["datetime"]).sort_values("datetime")

        if df.empty:
            context["error_message"] = "No valid claim dates available"
            return render(request, "myapp/temporal_analysis.html", context)

        df.set_index("datetime", inplace=True)

        # Create filter description
        filter_description = []
        if time_period != 'all':
            filter_description.append(f"Period: {time_period}")
        if benefit_type != 'all':
            filter_description.append(f"Benefit: {benefit_type}")
        if provider != 'all':
            filter_description.append(f"Provider: {provider}")
        if cost_center != 'all':
            filter_description.append(f"Cost Center: {cost_center}")
        
        filter_text = " | ".join(filter_description) if filter_description else "All Data"

        # -----------------------------
        # 1. MAIN TEMPORAL ANALYSIS
        # -----------------------------
        period_map = {
            'day': 'D',
            'week': 'W-MON',
            'month': 'M',
            'quarter': 'Q',
            'year': 'Y'
        }
        time_unit = request.GET.get('time_unit', 'month')
        resample_rule = period_map.get(time_unit, 'M')

        temporal = df.resample(resample_rule).agg({
            'amount': ['sum', 'count', 'mean', 'max', 'min', 'std'],
            'id': 'count'
        }).round(2)
        
        temporal.columns = ['total_amount', 'claim_count', 'avg_amount', 'max_amount', 'min_amount', 'std_amount', 'id_count']
        temporal = temporal.reset_index()
        temporal = temporal[temporal['claim_count'] > 0]  # Remove periods with no claims

        # Main temporal chart with enhanced insights
        metric = request.GET.get('metric', 'Total Amount')
        if metric == 'Claim Count':
            y_col, title_suffix, label = 'claim_count', "Claim Count", "Number of Claims"
        elif metric == 'Average Amount':
            y_col, title_suffix, label = 'avg_amount', "Average Claim Amount", "Average Amount (KES)"
        elif metric == 'Max Amount':
            y_col, title_suffix, label = 'max_amount', "Maximum Claim Amount", "Max Amount (KES)"
        elif metric == 'Min Amount':
            y_col, title_suffix, label = 'min_amount', "Minimum Claim Amount", "Min Amount (KES)"
        else:
            y_col, title_suffix, label = 'total_amount', "Total Claim Amount", "Total Amount (KES)"

        title = f"{title_suffix} by {time_unit.capitalize()} - {filter_text}"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=temporal['datetime'], y=temporal[y_col], 
                               mode='lines+markers', name=title_suffix,
                               line=dict(color='#1BB64F', width=3),
                               marker=dict(size=6)))
        
        # Add trend line
        if len(temporal) > 1:
            z = np.polyfit(range(len(temporal)), temporal[y_col], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(x=temporal['datetime'], y=p(range(len(temporal))),
                                   mode='lines', name='Trend Line',
                                   line=dict(color='#FF6B6B', width=2, dash='dash')))
        
        fig.update_layout(title=dict(text=title, x=0.5, xanchor='center'),
                         xaxis_title=f"Time ({time_unit.capitalize()})",
                         yaxis_title=label,
                         template='plotly_white',
                         hovermode='x unified')
        context['temporal_chart'] = fig.to_html(full_html=False)

        # -----------------------------
        # 2. CUMULATIVE ANALYSIS
        # -----------------------------
        temporal['cumulative_amount'] = temporal['total_amount'].cumsum()
        temporal['cumulative_claims'] = temporal['claim_count'].cumsum()
        
        cum_fig = make_subplots(specs=[[{"secondary_y": True}]])
        cum_fig.add_trace(go.Scatter(x=temporal['datetime'], y=temporal['cumulative_amount'],
                                   name="Cumulative Amount", line=dict(color='#3498DB', width=3)),
                         secondary_y=False)
        cum_fig.add_trace(go.Scatter(x=temporal['datetime'], y=temporal['cumulative_claims'],
                                   name="Cumulative Claims", line=dict(color='#9B59B6', width=3)),
                         secondary_y=True)
        
        cum_fig.update_layout(title=dict(text=f"Cumulative Growth Analysis - {filter_text}", x=0.5),
                            xaxis_title="Time",
                            template='plotly_white')
        cum_fig.update_yaxes(title_text="Cumulative Amount (KES)", secondary_y=False)
        cum_fig.update_yaxes(title_text="Cumulative Claims", secondary_y=True)
        context['cumulative_chart'] = cum_fig.to_html(full_html=False)

        # -----------------------------
        # 3. DEEP DAY-OF-WEEK ANALYSIS
        # -----------------------------
        df['day_of_week'] = df.index.day_name()
        df['day_of_week_num'] = df.index.dayofweek
        
        # Multiple perspectives on day-of-week analysis
        dow_analysis = df.groupby('day_of_week').agg({
            'amount': ['sum', 'mean', 'count', 'std'],
            'id': 'count'
        }).round(2)
        dow_analysis.columns = ['total_amount', 'avg_amount', 'transaction_count', 'std_amount', 'claim_count']
        dow_analysis = dow_analysis.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        # Create comprehensive day-of-week visualization
        dow_fig = make_subplots(rows=2, cols=2, 
                               subplot_titles=('Total Amount by Day', 'Average Claim by Day', 
                                             'Number of Claims by Day', 'Variability by Day'),
                               specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                     [{"secondary_y": False}, {"secondary_y": False}]])
        
        # Total amount
        dow_fig.add_trace(go.Bar(x=dow_analysis.index, y=dow_analysis['total_amount'],
                               name="Total Amount", marker_color='#1BB64F'),
                         row=1, col=1)
        
        # Average amount
        dow_fig.add_trace(go.Bar(x=dow_analysis.index, y=dow_analysis['avg_amount'],
                               name="Avg Amount", marker_color='#3498DB'),
                         row=1, col=2)
        
        # Claim count
        dow_fig.add_trace(go.Bar(x=dow_analysis.index, y=dow_analysis['claim_count'],
                               name="Claim Count", marker_color='#9B59B6'),
                         row=2, col=1)
        
        # Variability (std)
        dow_fig.add_trace(go.Bar(x=dow_analysis.index, y=dow_analysis['std_amount'],
                               name="Std Deviation", marker_color='#E74C3C'),
                         row=2, col=2)
        
        dow_fig.update_layout(height=600, title=dict(text=f"Deep Day-of-Week Analysis - {filter_text}", x=0.5),
                            showlegend=False, template='plotly_white')
        context['dow_chart'] = dow_fig.to_html(full_html=False)

        # -----------------------------
        # 4. MONTHLY DEEP DIVE
        # -----------------------------
        df['month'] = df.index.month_name()
        df['month_num'] = df.index.month
        
        monthly_analysis = df.groupby('month').agg({
            'amount': ['sum', 'mean', 'count', 'std', 'max'],
            'id': 'count'
        }).round(2)
        monthly_analysis.columns = ['total_amount', 'avg_amount', 'transaction_count', 'std_amount', 'max_amount', 'claim_count']
        monthly_analysis = monthly_analysis.reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ])
        
        month_fig = go.Figure()
        month_fig.add_trace(go.Bar(x=monthly_analysis.index, y=monthly_analysis['total_amount'],
                                 name="Total Amount", marker_color='#1BB64F'))
        month_fig.add_trace(go.Scatter(x=monthly_analysis.index, y=monthly_analysis['avg_amount'],
                                     name="Average Amount", line=dict(color='#E74C3C', width=3),
                                     yaxis='y2'))
        
        month_fig.update_layout(title=dict(text=f"Monthly Analysis with Trends - {filter_text}", x=0.5),
                              xaxis_title="Month",
                              yaxis_title="Total Amount (KES)",
                              yaxis2=dict(title="Average Amount (KES)", overlaying='y', side='right'),
                              template='plotly_white')
        context['month_chart'] = month_fig.to_html(full_html=False)

        # -----------------------------
        # 5. HOURLY ANALYSIS (if time data available)
        # -----------------------------
        if 'claim_prov_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['claim_prov_date']):
            df['hour'] = df.index.hour
            hourly_analysis = df.groupby('hour').agg({
                'amount': ['sum', 'mean', 'count', 'std'],
                'id': 'count'
            }).round(2)
            hourly_analysis.columns = ['total_amount', 'avg_amount', 'transaction_count', 'std_amount', 'claim_count']
            
            hour_fig = make_subplots(specs=[[{"secondary_y": True}]])
            hour_fig.add_trace(go.Bar(x=hourly_analysis.index, y=hourly_analysis['total_amount'],
                                    name="Total Amount", marker_color='#3498DB'),
                             secondary_y=False)
            hour_fig.add_trace(go.Scatter(x=hourly_analysis.index, y=hourly_analysis['avg_amount'],
                                        name="Average Amount", line=dict(color='#E74C3C', width=3)),
                             secondary_y=True)
            
            hour_fig.update_layout(title=dict(text=f"Hourly Claim Patterns - {filter_text}", x=0.5),
                                 xaxis_title="Hour of Day",
                                 template='plotly_white')
            hour_fig.update_yaxes(title_text="Total Amount (KES)", secondary_y=False)
            hour_fig.update_yaxes(title_text="Average Amount (KES)", secondary_y=True)
            context['hour_chart'] = hour_fig.to_html(full_html=False)

        # -----------------------------
        # 6. VARIABILITY AND DISTRIBUTION ANALYSIS
        # -----------------------------
        box_fig = px.box(df.reset_index(), x='month', y='amount',
                        title=f"Claim Amount Distribution by Month - {filter_text}",
                        labels={'amount': 'Claim Amount (KES)', 'month': 'Month'},
                        template='plotly_white')
        box_fig.update_traces(marker_color='#9B59B6')
        context['boxplot_chart'] = box_fig.to_html(full_html=False)

        # -----------------------------
        # 7. CATEGORICAL TEMPORAL TRENDS
        # -----------------------------
        # Top 5 benefits for clearer visualization
        top_benefits = df.groupby('benefit')['amount'].sum().nlargest(5).index
        df_top_benefits = df[df['benefit'].isin(top_benefits)]
        
        categorical_data = df_top_benefits.groupby([pd.Grouper(freq='M'), 'benefit'])['amount'].sum().reset_index()
        
        cat_fig = px.area(categorical_data, x='datetime', y='amount', color='benefit',
                         title=f"Temporal Trends by Top 5 Benefit Types - {filter_text}",
                         template='plotly_white')
        context['categorical_chart'] = cat_fig.to_html(full_html=False)

        # -----------------------------
        # 8. HEATMAP ANALYSIS
        # -----------------------------
        heatmap_data = df.groupby([df['day_of_week_num'], df['month_num']])['amount'].sum().unstack(fill_value=0)
        
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        heatmap_fig.update_layout(title=dict(text=f"Claims Heatmap: Day of Week vs Month - {filter_text}", x=0.5),
                                xaxis_title="Month",
                                yaxis_title="Day of Week",
                                template='plotly_white')
        context['heatmap_chart'] = heatmap_fig.to_html(full_html=False)

        # -----------------------------
        # 9. ADVANCED TIME SERIES ANALYSIS
        # -----------------------------
        
        # Rolling averages with different windows
        daily_data = df.resample('D')['amount'].sum().reset_index()
        for window in [7, 30]:
            daily_data[f'rolling_{window}'] = daily_data['amount'].rolling(window=window).mean()
        
        rolling_fig = go.Figure()
        rolling_fig.add_trace(go.Scatter(x=daily_data['datetime'], y=daily_data['amount'],
                                       name='Daily Amount', line=dict(color='lightgray')))
        rolling_fig.add_trace(go.Scatter(x=daily_data['datetime'], y=daily_data['rolling_7'],
                                       name='7-Day Rolling Avg', line=dict(color='#3498DB', width=3)))
        rolling_fig.add_trace(go.Scatter(x=daily_data['datetime'], y=daily_data['rolling_30'],
                                       name='30-Day Rolling Avg', line=dict(color='#E74C3C', width=3)))
        
        rolling_fig.update_layout(title=dict(text=f"Rolling Average Analysis - {filter_text}", x=0.5),
                                xaxis_title="Date",
                                yaxis_title="Amount (KES)",
                                template='plotly_white')
        context['rolling_avg_chart'] = rolling_fig.to_html(full_html=False)
        
        # Anomaly detection
        daily_data['z_score'] = np.abs(stats.zscore(daily_data['amount'].fillna(0)))
        anomalies = daily_data[daily_data['z_score'] > 3]
        
        anomaly_fig = go.Figure()
        anomaly_fig.add_trace(go.Scatter(x=daily_data['datetime'], y=daily_data['amount'],
                                       name='Daily Claims', mode='lines',
                                       line=dict(color='#3498DB')))
        if not anomalies.empty:
            anomaly_fig.add_trace(go.Scatter(x=anomalies['datetime'], y=anomalies['amount'],
                                           name='Anomalies', mode='markers',
                                           marker=dict(color='red', size=8, symbol='x')))
        
        anomaly_fig.update_layout(title=dict(text=f"Anomaly Detection (Z-score > 3) - {filter_text}", x=0.5),
                                template='plotly_white')
        context['anomaly_chart'] = anomaly_fig.to_html(full_html=False)

        # -----------------------------
        # 10. TOP PROVIDERS AND BENEFITS
        # -----------------------------
        top_providers = df.groupby('prov_name')['amount'].sum().nlargest(10)
        providers_fig = px.bar(x=top_providers.values, y=top_providers.index, orientation='h',
                              title=f"Top 10 Providers by Total Claims - {filter_text}",
                              labels={'x': 'Total Amount (KES)', 'y': 'Provider'},
                              template='plotly_white')
        providers_fig.update_traces(marker_color='#1BB64F')
        context['providers_chart'] = providers_fig.to_html(full_html=False)
        
        top_benefits = df.groupby('benefit')['amount'].sum().nlargest(10)
        benefits_fig = px.pie(values=top_benefits.values, names=top_benefits.index,
                             title=f"Top 10 Benefits Distribution - {filter_text}",
                             template='plotly_white')
        context['benefits_chart'] = benefits_fig.to_html(full_html=False)

        # -----------------------------
        # 11. SEASONAL DECOMPOSITION
        # -----------------------------
        if len(daily_data) > 30:
            try:
                # Ensure we have a regular time series
                daily_data = daily_data.set_index('datetime').asfreq('D').fillna(0)
                decomposition = seasonal_decompose(daily_data['amount'], model='additive', period=30)
                
                decomp_fig = make_subplots(rows=4, cols=1, subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))
                
                decomp_fig.add_trace(go.Scatter(x=daily_data.index, y=decomposition.observed, name='Original'), row=1, col=1)
                decomp_fig.add_trace(go.Scatter(x=daily_data.index, y=decomposition.trend, name='Trend'), row=2, col=1)
                decomp_fig.add_trace(go.Scatter(x=daily_data.index, y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
                decomp_fig.add_trace(go.Scatter(x=daily_data.index, y=decomposition.resid, name='Residual'), row=4, col=1)
                
                decomp_fig.update_layout(height=800, title=dict(text=f"Time Series Decomposition - {filter_text}", x=0.5),
                                      showlegend=False, template='plotly_white')
                context['decomposition_chart'] = decomp_fig.to_html(full_html=False)
            except Exception as e:
                print(f"Decomposition error: {e}")

        # -----------------------------
        # 12. WEEKLY PATTERN ANALYSIS
        # -----------------------------
        df['week'] = df.index.isocalendar().week
        df['year'] = df.index.year
        weekly_pattern = df.groupby(['year', 'week'])['amount'].sum().reset_index()
        weekly_pattern['date'] = weekly_pattern.apply(lambda x: datetime.strptime(f"{x['year']}-{x['week']}-1", "%Y-%W-%w"), axis=1)
        
        weekly_fig = px.line(weekly_pattern, x='date', y='amount',
                           title=f"Weekly Claim Patterns - {filter_text}",
                           template='plotly_white')
        weekly_fig.update_traces(line=dict(color='#9B59B6', width=2))
        context['weekly_pattern_chart'] = weekly_fig.to_html(full_html=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        context['error_message'] = f"An error occurred during analysis: {str(e)}"

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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from myapp.models import ClaimRecord


@login_required
def claims_prediction_home(request):
    # === Collect filter values ===
    selected_time_period = request.GET.get('time_period', 'all')
    selected_benefit_type = request.GET.get('benefit_type', 'all')
    selected_provider = request.GET.get('provider', 'all')

    # Default forecast horizon (in months)
    selected_forecast_months = int(request.GET.get('forecast_months', 6))

    # === Predefined forecast horizon options ===
    forecast_horizon_options = [3, 6, 8, 12, 18, 24, 36]  # months

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
        'forecast_horizon_options': forecast_horizon_options,
        'selected_time_period': selected_time_period,
        'selected_benefit_type': selected_benefit_type,
        'selected_provider': selected_provider,
        'selected_forecast_months': selected_forecast_months,
        'forecast_next_month': None,
        'forecast_growth': None,
        'forecast_accuracy': None,
        'forecast_r2': None,
        'forecast_mae': None,
        'forecast_rmse': None,
        'forecast_mape': None,
        'forecast_smape': None,
        'forecast_trend': None
    }

    try:
        # === FILTER DATA ===
        queryset = ClaimRecord.objects.all()

        if selected_time_period != 'all':
            today = timezone.now().date()
            days_map = {'3m': 90, '6m': 180, '12m': 365, '24m': 730}
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

        # === Debug chart ===
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
            # Primary model: ARIMA
            model = ARIMA(monthly_data['amount'], order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_months)
            forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                           periods=forecast_months, freq='M')
            predictions = model_fit.predict(start=1, end=len(monthly_data))
        except Exception as e:
            try:
                # Backup model: Exponential Smoothing
                model = ExponentialSmoothing(monthly_data['amount'], trend='add', seasonal=None)
                model_fit = model.fit()
                forecast = model_fit.forecast(forecast_months)
                forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                               periods=forecast_months, freq='M')
                predictions = model_fit.fittedvalues
            except:
                # Last fallback: Linear Regression
                x = np.arange(len(monthly_data))
                y = monthly_data['amount'].values
                coeffs = np.polyfit(x, y, 1)
                forecast = np.polyval(coeffs, np.arange(len(monthly_data), len(monthly_data) + forecast_months))
                predictions = np.polyval(coeffs, x)
                forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                               periods=forecast_months, freq='M')

        # === METRICS ===
        actuals = np.array(monthly_data['amount'], dtype=float)
        predictions = np.array(predictions, dtype=float)

        if np.all(actuals == actuals[0]):
            r2 = np.nan
        else:
            r2 = r2_score(actuals, predictions)

        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)

        non_zero_mask = actuals != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / actuals[non_zero_mask])) * 100
            smape = np.mean(2.0 * np.abs(predictions - actuals) / (np.abs(actuals) + np.abs(predictions))) * 100
            confidence = max(0, min(100, 100 - mape))
        else:
            mape = smape = confidence = np.nan

        # Trend detection
        if len(actuals) > 1:
            if actuals[-1] > np.mean(actuals):
                trend = "Upward"
            else:
                trend = "Downward"
        else:
            trend = "Stable"

        # Save metrics
        context.update({
            'forecast_accuracy': None if np.isnan(confidence) else round(confidence, 1),
            'forecast_r2': None if np.isnan(r2) else round(r2, 3),
            'forecast_mae': None if np.isnan(mae) else round(mae, 2),
            'forecast_rmse': None if np.isnan(rmse) else round(rmse, 2),
            'forecast_mape': None if np.isnan(mape) else round(mape, 2),
            'forecast_smape': None if np.isnan(smape) else round(smape, 2),
            'forecast_trend': trend
        })

        # === COMBINE HISTORICAL & FORECAST ===
        forecast_df = pd.DataFrame({'date': forecast_dates, 'amount': forecast, 'type': 'Forecast'})
        historical_df = monthly_data.copy()
        historical_df['type'] = 'Historical'
        combined_df = pd.concat([historical_df, forecast_df])

        # === FORECAST CHART ===
        fig = px.line(combined_df, x='date', y='amount', color='type',
                      title=f'Monthly Claims Volume with {forecast_months}-Month Forecast',
                      labels={'amount': 'Total Amount (KES)', 'date': 'Month'},
                      markers=True)

        fig.add_vrect(x0=last_date, x1=forecast_dates[-1], fillcolor="lightgray",
                      opacity=0.2, line_width=0, annotation_text="Forecast Period",
                      annotation_position="top left")

        # Add confidence intervals if available
        if 'model_fit' in locals() and hasattr(model_fit, "get_forecast"):
            conf_int = model_fit.get_forecast(steps=forecast_months).conf_int()
            fig.add_trace(go.Scatter(x=forecast_dates, y=conf_int.iloc[:, 0], fill=None,
                                     mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast_dates, y=conf_int.iloc[:, 1], fill='tonexty',
                                     mode='lines', line=dict(width=0),
                                     fillcolor='rgba(27, 182, 79, 0.2)',
                                     name='Confidence Interval'))

        context['visualizations']['forecast_volume'] = fig.to_html(full_html=False)

        # Save growth calculation
        context['forecast_next_month'] = float(forecast[0]) if len(forecast) > 0 else None
        if len(forecast) > 0 and monthly_data['amount'].iloc[-1] != 0:
            context['forecast_growth'] = round(((forecast[0] - monthly_data['amount'].iloc[-1]) /
                                               monthly_data['amount'].iloc[-1] * 100), 2)

    except Exception as e:
        context['error'] = f"Error processing forecast: {str(e)}"

    return render(request, 'myapp/forecasted_volume.html', context)





@login_required
def confidence_intervals(request):
    """
    View for interactive confidence interval analysis with filters:
    - Confidence level
    - Metric (avg / total)
    - Frequency (M, Q, Y)
    - Diagnosis (all / specific)
    """

    # --- Default context ---
    context = {
        "username": request.user.username,
        "confidence_level": 95,
        "metric": "avg",
        "frequency": "M",
        "diagnosis": "all",
        "plot": None,
        "summary": {},
        "outliers": [],
    }

    try:
        # --- 1. Get filters from GET params ---
        selected_confidence = int(request.GET.get("confidence", 95))
        metric = request.GET.get("metric", "avg")   # avg / total
        frequency = request.GET.get("frequency", "M")  # M / Q / Y
        diagnosis = request.GET.get("diagnosis", "all")

        context["confidence_level"] = selected_confidence
        context["metric"] = metric
        context["frequency"] = frequency
        context["diagnosis"] = diagnosis

        # Z-values
        z_values = {90: 1.645, 95: 1.96, 99: 2.576}
        z_value = z_values.get(selected_confidence, 1.96)

        # --- 2. Load claims data ---
        qs = ClaimRecord.objects.all().values("claim_prov_date", "amount", "diagnosis")
        df = pd.DataFrame.from_records(qs)

        if df.empty:
            context["error"] = "No claims data found."
            return render(request, "confidence_interval.html", context)

        df["date"] = pd.to_datetime(df["claim_prov_date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df.dropna(subset=["date", "amount"], inplace=True)

        # --- Diagnosis filtering ---
        if diagnosis != "all":
            df = df[df["diagnosis"] == diagnosis]

        if df.empty:
            context["error"] = "No valid claims data after filtering."
            return render(request, "confidence_interval.html", context)

        # --- 3. Aggregate by chosen frequency ---
        agg = df.set_index("date").resample(frequency).agg({
            "amount": ["sum", "count", "mean", "std"]
        }).reset_index()
        agg.columns = ["date", "total_amount", "claim_count", "avg_amount", "std_amount"]
        agg["std_amount"] = agg["std_amount"].fillna(0)

        # --- 4. Calculate CI ---
        agg["ci_lower"] = agg["avg_amount"] - z_value * agg["std_amount"] / np.sqrt(agg["claim_count"])
        agg["ci_upper"] = agg["avg_amount"] + z_value * agg["std_amount"] / np.sqrt(agg["claim_count"])
        agg["ci_range"] = agg["ci_upper"] - agg["ci_lower"]

        # --- 5. Summary metrics ---
        context["summary"] = {
            "avg_width": round(agg["ci_range"].mean(), 2),
            "widest_period": agg.loc[agg["ci_range"].idxmax(), "date"].strftime("%b %Y") if not agg.empty else "—",
            "narrowest_period": agg.loc[agg["ci_range"].idxmin(), "date"].strftime("%b %Y") if not agg.empty else "—",
            "stability": round(100 * (1 - (agg["ci_range"].std() / (agg["ci_range"].mean() or 1))), 1),
            "outlier_count": 0
        }

        # --- 6. Outlier detection (last 12 periods) ---
        last_periods = agg[agg["date"] >= (agg["date"].max() - pd.DateOffset(months=12))]
        outliers = []
        for _, row in last_periods.iterrows():
            if row["avg_amount"] > row["ci_upper"]:
                outliers.append({"date": row["date"].strftime("%b %Y"),
                                 "msg": f"+KES {row['avg_amount'] - row['ci_upper']:,.0f} above upper limit"})
            elif row["avg_amount"] < row["ci_lower"]:
                outliers.append({"date": row["date"].strftime("%b %Y"),
                                 "msg": f"-KES {row['ci_lower'] - row['avg_amount']:,.0f} below lower limit"})
        context["outliers"] = outliers
        context["summary"]["outlier_count"] = len(outliers)

        # --- 7. Choose metric for plotting ---
        if metric == "total":
            y_series = agg["total_amount"]
            y_label = "Total Claim Amount (KES)"
            trace_name = "Total Claims"
        else:
            y_series = agg["avg_amount"]
            y_label = "Average Claim Amount (KES)"
            trace_name = "Average Claim"

        # --- 8. Build Plotly chart ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg["date"], y=y_series,
                                 mode="lines+markers", name=trace_name,
                                 line=dict(color="#E30613")))
        fig.add_trace(go.Scatter(x=agg["date"], y=agg["ci_upper"],
                                 mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=agg["date"], y=agg["ci_lower"],
                                 mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(227,6,19,0.15)",
                                 name=f"{selected_confidence}% Confidence Interval"))

        fig.update_layout(
            title=f"{trace_name} with {selected_confidence}% Confidence Intervals",
            xaxis_title="Period",
            yaxis_title=y_label,
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40)
        )

        context["plot"] = fig.to_html(full_html=False)

        # --- 9. AJAX support ---
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            html = render(request, "confidence_interval.html", context).content.decode("utf-8")
            return JsonResponse({"html": html})

    except Exception as e:
        context["error"] = f"Error generating confidence intervals: {str(e)}"

    return render(request, "confidence_interval.html", context)




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



import traceback
import numpy as np
import pandas as pd
import shap
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from myapp.models import ClaimRecord

@login_required
def explainability(request):
    """
    Advanced Explainability view for claims time-series.
    Provides model forecasts, SHAP feature importance, 
    partial dependence, and fallback correlation insights.
    """
    context = {
        'active_tab': 'claims-prediction',
        'visualizations': {},
        'error': None
    }

    try:
        # --- 1. Load & clean data ---
        df = pd.DataFrame.from_records(
            ClaimRecord.objects.values('claim_prov_date', 'amount')
        )
        if df.empty:
            context['error'] = "No claims data found."
            return render(request, 'explainability.html', context)

        df['date'] = pd.to_datetime(df['claim_prov_date'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df = df.dropna(subset=['date']).sort_values('date')

        # --- 2. Monthly aggregation ---
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

        if len(monthly_data) < 6:
            context['error'] = "Need at least 6 months of data for explainability."
            return render(request, 'explainability.html', context)

        # --- 3. Feature engineering ---
        lagged_df = pd.DataFrame({'y': monthly_data['amount']})
        for lag in range(1, 4):
            lagged_df[f'lag_{lag}'] = monthly_data['amount'].shift(lag)

        # Add rolling statistics
        lagged_df['rolling_mean_3'] = monthly_data['amount'].rolling(3).mean()
        lagged_df['rolling_std_3'] = monthly_data['amount'].rolling(3).std()
        lagged_df['trend'] = np.arange(len(monthly_data))

        lagged_df = lagged_df.dropna()

        X = lagged_df.drop(columns=['y'])
        y = lagged_df['y']

        # --- 4. Fit ARIMA baseline model ---
        try:
            model = ARIMA(y, order=(1, 1, 1))
            model_fit = model.fit()
            forecast_next = model_fit.forecast(steps=1).iloc[0]
            context['forecasted_next_month'] = f"KES {forecast_next:,.2f}"
        except Exception as e:
            context['forecasted_next_month'] = "Forecast unavailable"
            model_fit = None

        # --- 5. Prediction wrapper for SHAP ---
        def arima_predict(data_as_array):
            preds = []
            y_list = y.tolist()
            for row in data_as_array:
                try:
                    temp_series = y_list.copy()
                    # inject last values as lags
                    for i, val in enumerate(row):
                        temp_series[-(i+1)] = val
                    pred_model = ARIMA(temp_series, order=(1, 1, 1))
                    pred_fit = pred_model.fit()
                    preds.append(pred_fit.forecast(steps=1).iloc[0])
                except Exception:
                    preds.append(np.nan)
            return np.array(preds)

        # --- 6. SHAP explainability ---
        try:
            explainer = shap.KernelExplainer(arima_predict, X)
            shap_values = explainer.shap_values(X, nsamples=30)

            if shap_values is not None and not np.all(np.isnan(shap_values)):
                shap_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Mean SHAP Value': np.nan_to_num(np.abs(shap_values)).mean(axis=0)
                }).sort_values('Mean SHAP Value', ascending=True)

                shap_fig = px.bar(
                    shap_df,
                    x='Mean SHAP Value',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance (SHAP, ARIMA)',
                    color='Mean SHAP Value',
                    color_continuous_scale='Blues'
                )
                context['visualizations']['shap_values'] = shap_fig.to_html(full_html=False)

                # Partial dependence for top 2 features
                for top_feature in shap_df.tail(2)['Feature']:
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
                    context['visualizations'][f'pd_{top_feature}'] = pd_fig.to_html(full_html=False)

            else:
                raise ValueError("Empty SHAP values")

        except Exception:
            # --- 7. Fallback: correlation-based importance ---
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

        # --- 8. Add raw claims trend chart ---
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=monthly_data['date'], y=monthly_data['amount'],
            mode='lines+markers', name='Claims Amount'
        ))
        if model_fit:
            trend_fig.add_trace(go.Scatter(
                x=[monthly_data['date'].max() + timedelta(days=30)],
                y=[forecast_next],
                mode='markers+text',
                name='Forecast Next Month',
                text=["Forecast"],
                textposition="top center",
                marker=dict(color="red", size=10)
            ))
        trend_fig.update_layout(title="Monthly Claims Trend with Forecast")
        context['visualizations']['trend'] = trend_fig.to_html(full_html=False)

        return render(request, 'explainability.html', context)

    except Exception:
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
    import io
    import csv
    from django.http import HttpResponse
    from openpyxl import Workbook
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
    from reportlab.lib.styles import getSampleStyleSheet

    export_type = request.GET.get('export', None)

    context = {
        'metrics': {},
        'risky_claims': [],
        'risk_distribution_data': json.dumps({'bins': [], 'counts': []}),
        'suspicious_providers': [],
        'diagnosis_patterns': [],
        'monthly_trends': []
    }

    try:
        queryset = ClaimRecord.objects.values(
            'claim_prov_date', 'amount', 'prov_name', 'icd10_code'
        )
        df = pd.DataFrame(list(queryset))

        if df.empty:
            context['error'] = "No claim data found."
            return render(request, 'risk_scores.html', context)

        # --- Data cleaning ---
        df['date'] = pd.to_datetime(df['claim_prov_date'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df['provider_name'] = df['prov_name'].fillna("Unknown")
        df['icd10_code'] = df['icd10_code'].fillna("Unknown")

        # Run fraud detection
        df = detect_fraud_anomalies(df)

        # --- Metrics ---
        fraud_count = int(df['fraud_flag'].sum())
        fraud_rate = fraud_count / len(df)
        fraud_amount = df.loc[df['fraud_flag'] == 1, 'amount'].sum()

        context['metrics'] = {
            'fraud_count': fraud_count,
            'fraud_rate': f"{fraud_rate:.1%}",
            'fraud_amount': f"KES {fraud_amount:,.2f}"
        }

        # Risky claims (top 50 for analysis table)
        risky_claims = df[df['fraud_flag'] == 1] \
            .sort_values('fraud_score', ascending=False) \
            .head(50)
        context['risky_claims'] = risky_claims.to_dict('records')

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

        # --- EXPORT OPTIONS ---
        if export_type == "csv":
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="fraud_report.csv"'
            df.to_csv(path_or_buf=response, index=False)
            return response

        elif export_type == "excel":
            response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            response['Content-Disposition'] = 'attachment; filename="fraud_report.xlsx"'
            wb = Workbook()
            ws = wb.active
            ws.append(list(df.columns))
            for row in df.itertuples(index=False):
                ws.append(row)
            wb.save(response)
            return response

        elif export_type == "pdf":
            response = HttpResponse(content_type='application/pdf')
            response['Content-Disposition'] = 'attachment; filename="fraud_report.pdf"'
            doc = SimpleDocTemplate(response)
            styles = getSampleStyleSheet()
            elements = [Paragraph("Fraud Detection Report", styles['Title']), Spacer(1, 12)]
            data = [list(df.columns)] + df.head(30).values.tolist()
            elements.append(Table(data))
            doc.build(elements)
            return response

    except Exception as e:
        context['error'] = f"Error: {str(e)}"

    return render(request, 'risk_scores.html', context)


def flag_claim(request, claim_id):
    if request.method == "POST":
        claim = get_object_or_404(Claim, id=claim_id)
        claim.flagged = True   # assuming you have a `flagged` BooleanField
        claim.save()
        return JsonResponse({"status": "success", "claim_id": claim.id})
    return HttpResponseNotAllowed(["POST"])







@login_required
def suspicious_providers(request):
    """Suspicious providers dashboard (DB-driven)"""
    import plotly.express as px

    sort_by = request.GET.get('sort', 'count')  # Default sort by fraud count
    context = {'active_tab': 'fraud-detection', 'visualizations': {}, 'sort_by': sort_by}

    try:
        queryset = ClaimRecord.objects.values(
            'id', 'claim_prov_date', 'amount', 'prov_name', 'icd10_code'
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



#######################
#####################
##################
@login_required
def minet_suspicious_providers(request):
    """Suspicious providers analysis with dataset selection"""
    context = {
        'available_datasets': get_database_tables(),
        'active_tab': 'suspicious-providers'
    }
    
    sort_by = request.GET.get('sort', 'count')  # Default sort by fraud count
    context['sort_by'] = sort_by
    
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'run_analysis':
            try:
                dataset_name = request.POST.get('dataset')
                
                if not dataset_name:
                    context['error'] = "Please select a dataset to analyze"
                    return render(request, 'minet_suspicious_providers.html', context)
                
                logger.info(f"Starting suspicious providers analysis on {dataset_name}")
                
                # Get and process data
                df = get_table_data(dataset_name)
                
                if df.empty:
                    context['error'] = f"Dataset '{dataset_name}' is empty or could not be loaded"
                    return render(request, 'minet_suspicious_providers.html', context)
                
                # Standardize column names to match our model
                column_mapping = {}
                if 'prov_name' in df.columns:
                    column_mapping['prov_name'] = 'provider_name'
                if 'claim_prov_date' in df.columns:
                    column_mapping['claim_prov_date'] = 'date'
                # Map other columns that might be used by detect_fraud_anomalies
                if 'icd10_code' in df.columns:
                    column_mapping['icd10_code'] = 'icd10_code'
                if 'amount' in df.columns:
                    column_mapping['amount'] = 'amount'
                
                df = df.rename(columns=column_mapping)
                
                # Ensure required columns exist for fraud detection
                required_cols = ['icd10_code', 'amount', 'provider_name']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    context['error'] = f"Dataset missing required columns: {', '.join(missing_cols)}"
                    return render(request, 'minet_suspicious_providers.html', context)
                
                # Run fraud detection
                df = detect_fraud_anomalies(df)
                
                # Prepare data
                df['date'] = pd.to_datetime(df.get('date'), errors='coerce')
                
                # Find amount column
                amount_column = None
                for col in ['amount', 'claim_amount', 'total_amount']:
                    if col in df.columns:
                        amount_column = col
                        break
                
                if amount_column:
                    df[amount_column] = pd.to_numeric(df[amount_column], errors='coerce').fillna(0)
                
                # Find provider column
                provider_column = None
                for col in ['provider_name', 'prov_name', 'provider']:
                    if col in df.columns:
                        provider_column = col
                        break
                
                if not provider_column:
                    context['error'] = "No provider information found in the dataset"
                    return render(request, 'minet_suspicious_providers.html', context)
                
                df['provider_name'] = df[provider_column].fillna("Unknown")
                
                # Aggregate provider stats
                provider_fraud = df.groupby('provider_name').agg(
                    Total_Amount=(amount_column, 'sum') if amount_column else ('fraud_flag', 'count'),
                    Fraud_Count=('fraud_flag', 'sum'),
                    Total_Claims=('fraud_flag', 'count')
                ).reset_index()
                
                provider_fraud['Fraud_Rate'] = (provider_fraud['Fraud_Count'] / provider_fraud['Total_Claims'] * 100).round(2)
                
                # Sort based on filter
                if sort_by == 'count':
                    top_fraud = provider_fraud.sort_values('Fraud_Count', ascending=False).head(20)
                elif sort_by == 'rate':
                    top_fraud = provider_fraud[provider_fraud['Total_Claims'] > 5].sort_values('Fraud_Rate', ascending=False).head(20)
                elif sort_by == 'amount':
                    top_fraud = provider_fraud.sort_values('Total_Amount', ascending=False).head(20)
                else:
                    top_fraud = provider_fraud.sort_values('Fraud_Count', ascending=False).head(20)
                
                # Create visualization
                import plotly.express as px
                
                if sort_by == 'count':
                    fig = px.bar(
                        top_fraud,
                        x='provider_name',
                        y='Fraud_Count',
                        color='Fraud_Rate',
                        title=f"Top Providers by Fraud Count - {dataset_name}",
                        labels={'provider_name': 'Provider', 'Fraud_Count': 'Fraud Cases', 'Fraud_Rate': 'Fraud Rate (%)'},
                        hover_data=['Total_Amount', 'Total_Claims']
                    )
                elif sort_by == 'rate':
                    fig = px.bar(
                        top_fraud,
                        x='provider_name',
                        y='Fraud_Rate',
                        color='Fraud_Count',
                        title=f"Top Providers by Fraud Rate - {dataset_name}",
                        labels={'provider_name': 'Provider', 'Fraud_Rate': 'Fraud Rate (%)', 'Fraud_Count': 'Fraud Cases'},
                        hover_data=['Total_Amount', 'Total_Claims']
                    )
                else:  # amount
                    fig = px.bar(
                        top_fraud,
                        x='provider_name',
                        y='Total_Amount',
                        color='Fraud_Rate',
                        title=f"Top Providers by Total Amount - {dataset_name}",
                        labels={'provider_name': 'Provider', 'Total_Amount': 'Total Amount', 'Fraud_Rate': 'Fraud Rate (%)'},
                        hover_data=['Fraud_Count', 'Total_Claims']
                    )
                
                context['visualization'] = fig.to_html(full_html=False)
                
                # Provider data for template
                context['provider_data'] = top_fraud.rename(columns={'provider_name': 'Provider'}).to_dict('records')
                
                # Data for comparison chart
                context['provider_names'] = list(top_fraud['provider_name'])
                context['provider_rates'] = list(top_fraud['Fraud_Rate'])
                context['avg_rate'] = round(provider_fraud['Fraud_Rate'].mean(), 1) if not provider_fraud.empty else 0
                
                # Summary metrics
                context['metrics'] = {
                    'total_providers': len(provider_fraud),
                    'high_risk_providers': len(top_fraud),
                    'avg_fraud_rate': round(provider_fraud['Fraud_Rate'].mean(), 1) if not provider_fraud.empty else 0,
                    'total_fraud_cases': int(provider_fraud['Fraud_Count'].sum())
                }
                
                context['analysis_complete'] = True
                context['dataset_name'] = dataset_name
                
                # Store results in session
                request.session['provider_results'] = provider_fraud.to_json()
                
            except Exception as e:
                logger.error(f"Error during provider analysis: {str(e)}", exc_info=True)
                context['error'] = f"Error processing dataset: {str(e)}"
    
    return render(request, 'minet_suspicious_providers.html', context)



from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import numpy as np
import pandas as pd
import plotly.express as px

from myapp.models import ClaimRecord


@login_required
def diagnosis_patterns(request):
    """
    Deep, descriptive diagnosis patterns analysis (basic -> deep).
    NOTE: the original fraud section is left intact (not modified) as requested.
    """
    context = {
        'active_tab': 'fraud-detection',
        'visualizations': {},
        'diagnosis_data': None,
        'deep_tables': {},   # additional prepared tables for the template
        'error': None
    }

    try:
        # -------------------------
        # Original fraud block (LEFT UNCHANGED per request)
        # -------------------------
        min_claims = int(request.GET.get('min_claims', 5))

        queryset = ClaimRecord.objects.values(
            'ailment', 'amount', 'claim_prov_date', 'prov_name'
        )
        df_fraud = pd.DataFrame(list(queryset))

        if df_fraud.empty:
            context['error'] = "No claims data available."
            return render(request, 'diagnosis_patterns1.html', context)

        # simulated fraud flag (original)
        df_fraud['fraud_flag'] = np.random.choice([0, 1], size=len(df_fraud), p=[0.9, 0.1])

        df_fraud.rename(columns={'ailment': 'Diagnosis'}, inplace=True)

        diagnosis_fraud = df_fraud.groupby('Diagnosis').agg({
            'amount': 'sum',
            'fraud_flag': ['sum', 'count']
        }).reset_index()

        diagnosis_fraud.columns = ['Diagnosis', 'Total Amount', 'Fraud Count', 'Total Claims']
        diagnosis_fraud['Fraud Rate'] = diagnosis_fraud['Fraud Count'] / diagnosis_fraud['Total Claims']

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

        # -------------------------
        # Deep descriptive analysis (additional, non-fraud)
        # -------------------------
        cols = [
            'ailment', 'amount', 'claim_prov_date', 'prov_name', 'gender', 'dob',
            'claim_me', 'claim_ce', 'benefit', 'benefit_desc', 'cost_center', 'admit_id'
        ]
        queryset_all = ClaimRecord.objects.values(*cols)
        df = pd.DataFrame(list(queryset_all))

        if df.empty:
            context['deep_error'] = "No additional claims data available for deeper diagnosis analysis."
            return render(request, 'diagnosis_patterns1.html', context)

        df.rename(columns={'ailment': 'Diagnosis', 'claim_prov_date': 'claim_prov_date_raw'}, inplace=True)

        # Clean amounts robustly
        df['amount'] = pd.to_numeric(
            df['amount'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True).replace('', '0'),
            errors='coerce'
        ).fillna(0.0)

        # Parse dates
        df['claim_prov_date'] = pd.to_datetime(df['claim_prov_date_raw'], errors='coerce')
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

        # Drop invalids
        df = df.dropna(subset=['Diagnosis', 'claim_prov_date'])
        if df.empty:
            context['deep_error'] = "No valid dated claims to analyze."
            return render(request, 'diagnosis_patterns1.html', context)

        # Derived fields
        df['year'] = df['claim_prov_date'].dt.year
        df['month'] = df['claim_prov_date'].dt.month
        df['month_name'] = df['claim_prov_date'].dt.strftime('%b')
        df['month_period'] = df['claim_prov_date'].dt.to_period('M').astype(str)
        df['week'] = df['claim_prov_date'].dt.isocalendar().week
        df['day_of_week'] = df['claim_prov_date'].dt.day_name()
        df['Age'] = (df['claim_prov_date'].dt.year - df['dob'].dt.year).where(df['dob'].notna())

        # SUMMARY: overall totals using unique claim_ce
        total_claims = df['claim_ce'].nunique() if 'claim_ce' in df.columns else 0
        total_amount = float(df['amount'].sum())
        unique_diagnoses = df['Diagnosis'].nunique()
        avg_claim = (total_amount / total_claims) if total_claims > 0 else 0.0

        context['deep_summary'] = {
            'total_claims': int(total_claims),
            'total_amount': total_amount,
            'unique_diagnoses': int(unique_diagnoses),
            'avg_claim': round(avg_claim, 2)
        }

        # -------------------------
        # Diagnosis-level summary stats
        # -------------------------
        # Ensure claim_ce exists
        if 'claim_ce' in df.columns:
            diag_stats = df.groupby('Diagnosis').agg(
                Total_Claims=('claim_ce', 'nunique'),   # ✅ unique claim_ce per diagnosis
                Total_Amount=('amount', 'sum'),
                Mean_Amount=('amount', 'mean'),
                Median_Amount=('amount', 'median'),
                Std_Amount=('amount', 'std')
            ).reset_index()
        else:
            diag_stats = df.groupby('Diagnosis').agg(
                Total_Claims=('Diagnosis', 'count'),
                Total_Amount=('amount', 'sum'),
                Mean_Amount=('amount', 'mean'),
                Median_Amount=('amount', 'median'),
                Std_Amount=('amount', 'std')
            ).reset_index()

        # Quantiles
        q = df.groupby('Diagnosis')['amount'].quantile([0.25, 0.5, 0.75, 0.9]).reset_index()
        q.columns = ['Diagnosis', 'quantile', 'amt_q']
        q_pivot = q.pivot(index='Diagnosis', columns='quantile', values='amt_q').reset_index().rename(
            columns={0.25: 'q25', 0.5: 'q50', 0.75: 'q75', 0.9: 'q90'}
        )
        diag_stats = diag_stats.merge(q_pivot, on='Diagnosis', how='left')

        # Contributions
        diag_stats['Pct_of_Claims'] = (diag_stats['Total_Claims'] / diag_stats['Total_Claims'].sum()) * 100
        diag_stats['Pct_of_Amount'] = (diag_stats['Total_Amount'] / diag_stats['Total_Amount'].sum()) * 100
        diag_stats['IQR'] = diag_stats['q75'] - diag_stats['q25']

        # Sort
        diag_stats = diag_stats.sort_values('Total_Claims', ascending=False).reset_index(drop=True)

        # Save table
        context['deep_tables']['diagnosis_stats'] = diag_stats.head(200).to_dict('records')

        # -------------------------
        # Visual: Top diagnoses by count & amount
        # -------------------------
        top_by_count = diag_stats.head(20)
        if not top_by_count.empty:
            fig_count = px.bar(
                top_by_count,
                x='Diagnosis', y='Total_Claims',
                color='Total_Amount',
                hover_data=['Mean_Amount', 'Median_Amount', 'Pct_of_Amount'],
                title='Top Diagnoses by Claim Count (with Total Amount shading)'
            )
            context['visualizations']['top_by_count'] = fig_count.to_html(full_html=False)

        top_by_amount = diag_stats.sort_values('Total_Amount', ascending=False).head(20)
        if not top_by_amount.empty:
            fig_amount = px.bar(
                top_by_amount,
                x='Diagnosis', y='Total_Amount',
                color='Total_Claims',
                hover_data=['Mean_Amount', 'Median_Amount', 'Pct_of_Claims'],
                title='Top Diagnoses by Total Amount'
            )
            context['visualizations']['top_by_amount'] = fig_amount.to_html(full_html=False)

        # -------------------------
        # Visual: distribution (boxplot) for top diagnoses
        # -------------------------
        top_diags_for_box = diag_stats['Diagnosis'].head(12).tolist()
        if top_diags_for_box:
            box_df = df[df['Diagnosis'].isin(top_diags_for_box)].copy()
            # sort diagnosis categories by overall count for consistent ordering
            order = diag_stats[diag_stats['Diagnosis'].isin(top_diags_for_box)]['Diagnosis'].tolist()
            box_fig = px.box(
                box_df,
                x='Diagnosis', y='amount',
                category_orders={'Diagnosis': order},
                title='Claim Amount Distribution for Top Diagnoses (boxplot)',
                labels={'amount': 'Claim Amount (KES)'}
            )
            context['visualizations']['box_by_diagnosis'] = box_fig.to_html(full_html=False)

        # -------------------------
        # Temporal analysis: monthly counts & amounts per diagnosis (top N)
        # -------------------------
        monthly = df.groupby(['month_period', 'Diagnosis']).agg(
            Claims=('Diagnosis', 'count'),
            Amount=('amount', 'sum')
        ).reset_index()

        # choose top N diagnoses by total claims for clearer multi-line charts
        topN = 8
        top_diags = diag_stats['Diagnosis'].head(topN).tolist()
        monthly_top = monthly[monthly['Diagnosis'].isin(top_diags)].copy()

        if not monthly_top.empty:
            monthly_top['month_period_dt'] = pd.to_datetime(monthly_top['month_period'])
            monthly_fig = px.line(
                monthly_top.sort_values('month_period_dt'),
                x='month_period_dt', y='Claims',
                color='Diagnosis', markers=True,
                title=f'Monthly Claims Trend for Top {topN} Diagnoses'
            )
            monthly_fig.update_xaxes(tickformat='%Y-%m')
            context['visualizations']['monthly_trends_top'] = monthly_fig.to_html(full_html=False)

        # -------------------------
        # Seasonality heatmap: month x year amounts (for all diagnoses or top ones)
        # -------------------------
        heat = df.groupby([df['claim_prov_date'].dt.month.rename('month'), df['claim_prov_date'].dt.year.rename('year')])['amount'].sum().unstack(fill_value=0)
        heat = heat.reindex(index=range(1, 13)).fillna(0)
        month_names = [pd.to_datetime(str(m), format='%m').strftime('%b') for m in heat.index]
        heatmap_fig = px.imshow(
            heat.values,
            labels=dict(x='Year', y='Month', color='Total Amount (KES)'),
            x=heat.columns.astype(str).tolist(),
            y=month_names,
            title='Monthly Seasonality: Total Amount (Month vs Year)'
        )
        heatmap_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        context['visualizations']['seasonality_heatmap'] = heatmap_fig.to_html(full_html=False)

        # -------------------------
        # Provider x Diagnosis: stacked bar for top providers
        # -------------------------
        prov_diag = df.groupby(['prov_name', 'Diagnosis']).agg(
            Claims=('Diagnosis', 'count'), Amount=('amount', 'sum')
        ).reset_index()

        # find top providers by claims
        prov_totals = prov_diag.groupby('prov_name')['Claims'].sum().reset_index().sort_values('Claims', ascending=False)
        top_providers = prov_totals['prov_name'].head(12).tolist()
        prov_plot_df = prov_diag[prov_diag['prov_name'].isin(top_providers)].copy()

        if not prov_plot_df.empty:
            prov_fig = px.bar(
                prov_plot_df,
                x='prov_name', y='Claims', color='Diagnosis',
                title='Diagnosis Breakdown for Top Providers (stacked)',
                labels={'prov_name': 'Provider', 'Claims': 'Number of Claims'}
            )
            prov_fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
            context['visualizations']['provider_diagnosis'] = prov_fig.to_html(full_html=False)

        # -------------------------
        # Demographics: gender split and age distribution by diagnosis (top ones)
        # -------------------------
        gender_df = df.groupby(['Diagnosis', 'gender']).size().reset_index(name='Claims')
        if not gender_df.empty:
            # restrict to top diagnoses for readability
            gender_top = gender_df[gender_df['Diagnosis'].isin(top_diags_for_box)]
            gender_fig = px.bar(
                gender_top,
                x='Diagnosis', y='Claims', color='gender',
                barmode='group',
                title='Gender Split for Top Diagnoses'
            )
            context['visualizations']['gender_split_top'] = gender_fig.to_html(full_html=False)

        # Age distribution boxplot for top diagnoses
        if 'Age' in df.columns and not df['Age'].dropna().empty:
            age_df = df[df['Diagnosis'].isin(top_diags_for_box) & df['Age'].notna()]
            if not age_df.empty:
                age_fig = px.box(
                    age_df,
                    x='Diagnosis', y='Age',
                    title='Age Distribution for Top Diagnoses',
                    labels={'Age': 'Age (years)'}
                )
                context['visualizations']['age_distribution_top'] = age_fig.to_html(full_html=False)

        # -------------------------
        # Severity buckets (low/medium/high) based on global percentiles
        # -------------------------
        q25, q50, q75 = df['amount'].quantile([0.25, 0.5, 0.75]).fillna(0).tolist()
        bins = [-np.inf, q25, q50, q75, np.inf]
        labels = ['Very Low (<=Q1)', 'Low (Q1-Q2)', 'Medium (Q2-Q3)', 'High (>Q3)']
        df['severity_bucket'] = pd.cut(df['amount'], bins=bins, labels=labels)
        severity_counts = df.groupby(['severity_bucket', 'Diagnosis']).size().reset_index(name='count')
        # For readability, show severity distribution for top diagnoses
        sev_top = severity_counts[severity_counts['Diagnosis'].isin(top_diags_for_box)]
        if not sev_top.empty:
            sev_fig = px.bar(
                sev_top,
                x='Diagnosis', y='count', color='severity_bucket',
                barmode='stack',
                title='Severity Bucket Distribution for Top Diagnoses'
            )
            context['visualizations']['severity_buckets_top'] = sev_fig.to_html(full_html=False)

        # -------------------------
        # Repeat claimant analysis: identify diagnoses with many repeat claimants
        # -------------------------
        if 'claim_me' in df.columns:
            repeat = df.groupby(['Diagnosis', 'claim_me']).size().reset_index(name='claims_per_member')
            repeat_summary = repeat.groupby('Diagnosis').agg(
                members_with_multiple_claims=('claims_per_member', lambda s: (s > 1).sum()),
                avg_claims_per_member=('claims_per_member', 'mean'),
                max_claims_by_member=('claims_per_member', 'max')
            ).reset_index().sort_values('members_with_multiple_claims', ascending=False)
            context['deep_tables']['repeat_claimants_summary'] = repeat_summary.head(50).to_dict('records')

            # Visual: diagnoses with highest number of members having multiple claims
            if not repeat_summary.empty:
                repeat_fig = px.bar(
                    repeat_summary.head(20),
                    x='Diagnosis', y='members_with_multiple_claims',
                    title='Number of Members with Multiple Claims by Diagnosis'
                )
                context['visualizations']['repeat_claimants'] = repeat_fig.to_html(full_html=False)

        # -------------------------
        # Volatility / variability metrics: CV and IQR highlight
        # -------------------------
        diag_stats['CV'] = diag_stats['Std_Amount'] / diag_stats['Mean_Amount'].replace(0, np.nan)
        # highlight diagnoses with large CV or large IQR
        volatility = diag_stats.sort_values(['CV', 'IQR'], ascending=False).head(20)
        if not volatility.empty:
            vol_fig = px.bar(
                volatility,
                x='Diagnosis', y='CV',
                title='Coefficient of Variation (Std / Mean) by Diagnosis (top volatile)'
            )
            context['visualizations']['volatility_top'] = vol_fig.to_html(full_html=False)

        # -------------------------
        # Prepare final tables for template
        # -------------------------
        context['deep_tables']['diag_stats_full'] = diag_stats.to_dict('records')
        context['deep_tables']['top_by_amount'] = top_by_amount.to_dict('records')
        context['deep_tables']['top_by_count'] = top_by_count.to_dict('records')

    except Exception as e:
        import traceback
        traceback.print_exc()
        context['error'] = f"An error occurred while analyzing diagnosis patterns: {str(e)}"

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


####

##### Minet mothly fraud trend
from .views import detect_fraud_anomalies
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import pandas as pd
import plotly.express as px
from .models import ClaimRecord

@login_required
def minet_monthly_trends(request):
    """Monthly fraud trends analysis with dataset selection"""
    context = {
        'available_datasets': get_database_tables(),
        'active_tab': 'monthly-trends',
        'visualizations': {},
        'trend_data': None
    }
    
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'run_analysis':
            try:
                dataset_name = request.POST.get('dataset')
                
                if not dataset_name:
                    context['error'] = "Please select a dataset to analyze"
                    return render(request, 'minet_monthly_trends.html', context)
                
                logger.info(f"Starting monthly trends analysis on {dataset_name}")
                
                # Get data based on selected dataset
                if dataset_name == 'claim_records':  # Default dataset from model
                    # Pull claims from DB
                    queryset = ClaimRecord.objects.values(
                        'claim_prov_date', 'amount', 'prov_name', 'icd10_code'
                    )
                    df = pd.DataFrame(list(queryset))
                else:
                    # Get data from other datasets
                    df = get_table_data(dataset_name)
                
                if df.empty:
                    context['error'] = f"Dataset '{dataset_name}' is empty or could not be loaded"
                    return render(request, 'minet_monthly_trends.html', context)
                
                # Standardize column names
                column_mapping = {}
                if 'claim_prov_date' in df.columns:
                    column_mapping['claim_prov_date'] = 'date'
                if 'prov_name' in df.columns:
                    column_mapping['prov_name'] = 'provider_name'
                if 'icd10_code' in df.columns:
                    column_mapping['icd10_code'] = 'icd10_code'
                if 'amount' in df.columns:
                    column_mapping['amount'] = 'amount'
                
                df = df.rename(columns=column_mapping)
                
                # Ensure required columns exist
                required_cols = ['date', 'amount']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    context['error'] = f"Dataset missing required columns: {', '.join(missing_cols)}"
                    return render(request, 'minet_monthly_trends.html', context)
                
                # Data cleaning
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
                
                # Add provider name if available
                if 'provider_name' in df.columns:
                    df['provider_name'] = df['provider_name'].fillna("Unknown")
                else:
                    df['provider_name'] = "Unknown"
                
                # Add ICD10 code if available
                if 'icd10_code' in df.columns:
                    df['icd10_code'] = df['icd10_code'].fillna("Unknown")
                else:
                    df['icd10_code'] = "Unknown"
                
                # Run fraud detection
                df = detect_fraud_anomalies(df)
                
                # Monthly aggregation
                df = df.dropna(subset=['date'])  # Remove rows with invalid dates
                
                # Create monthly fraud trends
                fraud_over_time = (
                    df.set_index('date')
                      .resample('M')['fraud_flag']
                      .sum()
                      .reset_index()
                )
                
                # Calculate additional metrics
                total_fraud = df['fraud_flag'].sum()
                total_claims = len(df)
                fraud_rate = (total_fraud / total_claims * 100) if total_claims > 0 else 0
                
                # Create visualization
                fig = px.line(
                    fraud_over_time,
                    x='date',
                    y='fraud_flag',
                    title=f"Monthly Fraud Cases - {dataset_name}",
                    markers=True,
                    labels={"date": "Month", "fraud_flag": "Fraud Cases"}
                )
                fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis=dict(showgrid=True, gridcolor="lightgrey"),
                    yaxis=dict(showgrid=True, gridcolor="lightgrey"),
                    height=500
                )
                
                # Create bar chart for comparison
                fig_bar = px.bar(
                    fraud_over_time,
                    x='date',
                    y='fraud_flag',
                    title=f"Monthly Fraud Cases (Bar Chart) - {dataset_name}",
                    labels={"date": "Month", "fraud_flag": "Fraud Cases"},
                    color='fraud_flag',
                    color_continuous_scale='Reds'
                )
                fig_bar.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis=dict(showgrid=True, gridcolor="lightgrey"),
                    yaxis=dict(showgrid=True, gridcolor="lightgrey"),
                    height=500
                )
                
                context['visualizations']['trends'] = fig.to_html(full_html=False)
                context['visualizations']['bar_chart'] = fig_bar.to_html(full_html=False)
                context['trend_data'] = fraud_over_time.to_dict('records')
                context['metrics'] = {
                    'total_fraud': total_fraud,
                    'total_claims': total_claims,
                    'fraud_rate': round(fraud_rate, 2),
                    'analysis_period': f"{fraud_over_time['date'].min().strftime('%Y-%m')} to {fraud_over_time['date'].max().strftime('%Y-%m')}" if not fraud_over_time.empty else "N/A"
                }
                context['analysis_complete'] = True
                context['dataset_name'] = dataset_name
                
            except Exception as e:
                logger.error(f"Error during monthly trends analysis: {str(e)}", exc_info=True)
                context['error'] = f"Error processing dataset: {str(e)}"
    
    return render(request, 'minet_monthly_trends.html', context)


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

    


# views.py
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from datetime import datetime, timedelta
import pandas as pd
from .models import ClaimRecord  # Adjust import based on your model location

@login_required
def minet_potential_cases(request):
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
        return render(request, 'minet_potential_cases.html', {
            'fraud_cases': [],
            'date_ranges': ["All Time", "Last 7 Days", "Last 30 Days", "This Month", "Custom Range"],
            'risk_levels': ["High", "Medium", "Low"],
            'providers': providers
        })

    # Convert Decimal to float and handle missing values
    df['amount'] = df['amount'].astype(float).fillna(0)
    
    # Convert date field to datetime
    df['claim_prov_date'] = pd.to_datetime(df['claim_prov_date'], errors='coerce')
    
    # Filter out records with invalid dates if needed
    df = df.dropna(subset=['claim_prov_date'])

    # Fraud score calculation using Z-score
    amount_mean = df['amount'].mean()
    amount_std = df['amount'].std()
    
    if amount_std > 0:  # Avoid division by zero
        df['fraud_score'] = (df['amount'] - amount_mean) / amount_std
        df['fraud_score'] = df['fraud_score'].abs() / df['fraud_score'].abs().max()
    else:
        df['fraud_score'] = 0
        
    df['fraud_score'] = df['fraud_score'].fillna(0)

    def classify_risk(score):
        if score >= 0.7: 
            return "High"
        elif score >= 0.4: 
            return "Medium"
        return "Low"

    df['risk_level'] = df['fraud_score'].apply(classify_risk)

    # Get filter values from request
    date_range = request.GET.get('date_range', 'All Time')
    risk_level = request.GET.get('risk_level', '')
    provider = request.GET.get('provider', '')
    start_date = request.GET.get('start_date', '')
    end_date = request.GET.get('end_date', '')

    # Date range filter
    if date_range != 'All Time':
        today = pd.Timestamp.today()
        
        if date_range == 'Last 7 Days':
            df = df[df['claim_prov_date'] >= (today - pd.Timedelta(days=7))]
        elif date_range == 'Last 30 Days':
            df = df[df['claim_prov_date'] >= (today - pd.Timedelta(days=30))]
        elif date_range == 'This Month':
            df = df[df['claim_prov_date'].dt.month == today.month]
        elif date_range == 'Custom Range' and start_date and end_date:
            try:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df['claim_prov_date'] >= start_dt) & (df['claim_prov_date'] <= end_dt)]
            except:
                pass  # If date parsing fails, don't filter

    # Risk filter
    if risk_level and risk_level != "All Risk Levels":
        df = df[df['risk_level'] == risk_level]

    # Provider filter (skip if 'All')
    if provider and provider != "All Providers":
        df = df[df['prov_name'] == provider]

    # Prepare data for template
    fraud_df = df[df['risk_level'].isin(['High', 'Medium'])]
    
    # Format data for template
    fraud_cases = []
    for _, row in fraud_df.iterrows():
        fraud_cases.append({
            'claim_id': row.get('admit_id', 'N/A'),
            'date': row.get('claim_prov_date'),
            'claim_me': row.get('claim_me', 'N/A'),
            'provider_name': row.get('prov_name', 'N/A'),
            'amount': row.get('amount', 0),
            'fraud_score': row.get('fraud_score', 0),
            'risk_level': row.get('risk_level', 'Low')
        })

    return render(request, 'minet_potential_cases.html', {
        'fraud_cases': fraud_cases,
        'date_ranges': ["All Time", "Last 7 Days", "Last 30 Days", "This Month", "Custom Range"],
        'risk_levels': ["All Risk Levels", "High", "Medium", "Low"],
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



#################
#################

##########Claim prediction 

def claim_prediction1_view(request):
    # Example predictions
    predictions = [
        {"claim_id": 101, "amount": 5000.75, "confidence": 92.3},
        {"claim_id": 102, "amount": 3500.00, "confidence": 88.1},
        {"claim_id": 103, "amount": 4250.50, "confidence": 95.0},
    ]
    return render(request, "myapp/claim_prediction1.html", {"predictions": predictions})

def claim_prediction2_view(request):
    return render(request, "myapp/claim_prediction2.html")

def claim_prediction3_view(request):
    return render(request, "myapp/claim_prediction3.html")




#######
######


#######

#######


##### Minet 

@login_required
def minet_forecast_volume(request):
    selected_time_period = request.GET.get('time_period', 'all')
    selected_benefit_type = request.GET.get('benefit_type', 'all')
    selected_provider = request.GET.get('provider', 'all')
    selected_forecast_months = int(request.GET.get('forecast_months', 3))

    context = {
        'username': request.user.username,
        'active_tab': 'minet-forecast-volume',
        'visualizations': {},
        'benefit_types': sorted(ClaimRecord.objects.values_list('benefit', flat=True)
                                 .exclude(benefit__isnull=True).exclude(benefit='').distinct()),
        'providers': sorted(ClaimRecord.objects.values_list('prov_name', flat=True)
                             .exclude(prov_name__isnull=True).exclude(prov_name='').distinct()),
        'selected_time_period': selected_time_period,
        'selected_benefit_type': selected_benefit_type,
        'selected_provider': selected_provider,
        'selected_forecast_months': selected_forecast_months,
        'forecast_accuracy': None,
        'forecast_r2': None,
        'forecast_mae': None
    }

    try:
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

        df = pd.DataFrame.from_records(queryset.values('claim_prov_date', 'amount'))
        if df.empty:
            context['error'] = "No claims data found."
            return render(request, 'myapp/minet_forecast_volume.html', context)

        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df['datetime'] = pd.to_datetime(df['claim_prov_date'], errors='coerce')
        df.dropna(subset=['datetime'], inplace=True)

        monthly_data = df.groupby(pd.Grouper(key='datetime', freq='M'))['amount'].sum().reset_index()
        monthly_data.rename(columns={'datetime': 'date'}, inplace=True)

        debug_fig = px.bar(monthly_data, x='date', y='amount', title="Raw Monthly Aggregated Data",
                           labels={'amount': 'Total Amount (KES)', 'date': 'Month'}, text='amount')
        debug_fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        context['visualizations']['raw_monthly_data'] = debug_fig.to_html(full_html=False)

        if len(monthly_data) < 3:
            context['error'] = "Not enough historical data for forecasting."
            return render(request, 'myapp/minet_forecast_volume.html', context)

        last_date = monthly_data['date'].max()
        forecast_months = selected_forecast_months

        try:
            model = ARIMA(monthly_data['amount'], order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_months)
            forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                           periods=forecast_months, freq='M')
            predictions = model_fit.predict(start=1, end=len(monthly_data))
        except Exception:
            x = np.arange(len(monthly_data))
            y = monthly_data['amount'].values
            coeffs = np.polyfit(x, y, 1)
            forecast = np.polyval(coeffs, np.arange(len(monthly_data), len(monthly_data) + forecast_months))
            predictions = np.polyval(coeffs, x)
            forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                           periods=forecast_months, freq='M')

        actuals = monthly_data['amount'].astype(float).values
        predictions = np.array(predictions, dtype=float)
        r2 = r2_score(actuals, predictions) if len(set(actuals)) > 1 else np.nan
        mae = mean_absolute_error(actuals, predictions)
        mape = np.mean(np.abs((actuals - predictions) / np.where(actuals != 0, actuals, 1))) * 100
        confidence = max(0, min(100, 100 - mape))

        context['forecast_accuracy'] = round(confidence, 1)
        context['forecast_r2'] = None if np.isnan(r2) else round(r2, 3)
        context['forecast_mae'] = round(mae, 2)

        forecast_df = pd.DataFrame({'date': forecast_dates, 'amount': forecast, 'type': 'Forecast'})
        historical_df = monthly_data.copy()
        historical_df['type'] = 'Historical'
        combined_df = pd.concat([historical_df, forecast_df])

        fig = px.line(combined_df, x='date', y='amount', color='type',
                      title='Monthly Claims Volume with Forecast',
                      labels={'amount': 'Total Amount (KES)', 'date': 'Month'},
                      markers=True)
        fig.add_vrect(x0=last_date, x1=forecast_dates[-1], fillcolor="lightgray",
                      opacity=0.2, line_width=0, annotation_text="Forecast Period",
                      annotation_position="top left")
        context['visualizations']['forecast_volume'] = fig.to_html(full_html=False)

    except Exception as e:
        context['error'] = f"Error processing forecast: {e}"

    return render(request, 'myapp/minet_forecast_volume.html', context)



@login_required
def minet_impact_simulation(request):
    """Impact simulation view for Minet Claim Prediction tab"""
    qs = ClaimRecord.objects.values('amount')
    df = pd.DataFrame.from_records(qs)

    if df.empty:
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'error': 'No claims data found'}, status=400)
        return render(request, 'minet_impact_simulation.html', {'error': 'No claims data found'})

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
        'active_tab': 'minet-claim-prediction',
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
    return render(request, 'minet_impact_simulation.html', context)



##### modelling

# views.py
# views.py
# views.py
import logging
import traceback
import base64
import json
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from django.db import connection
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_GET, require_POST
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings
from django.core.serializers.json import DjangoJSONEncoder

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score, max_error, mean_absolute_percentage_error
)

# Setup logging for this module
logger = logging.getLogger(__name__)


class NumpyJSONEncoder(DjangoJSONEncoder):
    """Custom JSON encoder that handles NumPy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class MLModelManager:
    """
    Central class for managing ML operations including:
    - loading and cleaning data,
    - training models,
    - calculating feature importance,
    - generating performance visualizations.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.data = None
        self.cleaned_df = None
        self.feature_importance = None
        self.metrics = None
        self.required_columns = None
        self.selected_model_type = None
        self.model_params = {}
        self.training_date = None
        self._available_tables = None
        self.cleaned_df_json = None

    @property
    def available_tables(self):
        """Return list of non-internal tables from the database."""
        if self._available_tables is None:
            self._available_tables = self.get_database_tables()
        return self._available_tables

    def get_database_tables():
        """
        Return all non-internal tables from the database.
        Works with SQLite and PostgreSQL.
        """
        vendor = connection.vendor  # 'sqlite', 'postgresql', etc.

        with connection.cursor() as cursor:
            if vendor == 'sqlite':
                cursor.execute("""
                    SELECT name 
                    FROM sqlite_master
                    WHERE type='table'
                    AND name NOT LIKE 'sqlite_%'
                    AND name NOT LIKE 'django_%'
                    AND name NOT LIKE 'auth_%'
                    AND name NOT LIKE 'sessions%'
                """)
            elif vendor == 'postgresql':
                cursor.execute("""
                    SELECT tablename
                    FROM pg_catalog.pg_tables
                    WHERE schemaname = 'public'
                    AND tablename NOT LIKE 'django_%'
                    AND tablename NOT LIKE 'auth_%'
                    AND tablename NOT LIKE 'sessions%'
                """)
            else:
                raise NotImplementedError(
                    f"Database vendor '{vendor}' not supported yet."
                )

            tables = [row[0] for row in cursor.fetchall()]
        return tables

    def load_data(self, table_name):
        """
        Load data from the specified table into a DataFrame.
        Returns True on success, False otherwise.
        """
        try:
            # Generic loading of any table
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 50000")  # Limit rows for memory
                columns = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
            
            df = pd.DataFrame(rows, columns=columns)
            
            # Store original data
            self.data = df
            return True
            
        except Exception as e:
            logger.error(f"Error loading data from table '{table_name}': {e}", exc_info=True)
            return False

    def get_columns(self):
        """Return sorted list of columns for currently loaded dataset."""
        if self.data is not None:
            return sorted(self.data.columns.tolist())
        return []

    def clean_data(self, target_column=None):
        """
        Clean self.data by handling missing values, outliers, and creating derived features.
        Optimized for memory usage.
        """
        try:
            if self.data is None:
                logger.warning("clean_data called but no data loaded")
                return False

            df = self.data.copy()
            
            # Sample data if too large (for memory optimization)
            if len(df) > 20000:
                df = df.sample(n=20000, random_state=42)
                logger.info(f"Sampled data to 20,000 rows for memory optimization")
            
            # Handle missing values
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('Unknown')
                elif pd.api.types.is_numeric_dtype(df[col]):
                    # Use median for numeric columns
                    df[col] = df[col].fillna(df[col].median())
            
            # Convert date columns if present
            date_columns = ['claim_prov_date', 'dob']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Create derived features
            if 'dob' in df.columns and 'claim_prov_date' in df.columns:
                df['age_at_claim'] = (df['claim_prov_date'] - df['dob']).dt.days / 365.25
                df['age_at_claim'] = df['age_at_claim'].fillna(df['age_at_claim'].median())
            
            if 'claim_prov_date' in df.columns:
                df['claim_month'] = df['claim_prov_date'].dt.month
                df['claim_quarter'] = df['claim_prov_date'].dt.quarter
                df['claim_year'] = df['claim_prov_date'].dt.year
            
            # Handle outliers in numeric columns using IQR method
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column and target_column in numeric_cols:
                numeric_cols.remove(target_column)  # Don't cap the target variable
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            
            # For high cardinality categorical columns, use label encoding instead of one-hot
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                if df[col].nunique() > 50:  # Very high cardinality
                    # Use label encoding for very high cardinality columns
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                elif df[col].nunique() > 20:  # High cardinality
                    # Keep only top categories and group others as 'Other'
                    top_categories = df[col].value_counts().nlargest(10).index.tolist()
                    df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
            
            self.cleaned_df = df
            # Convert to JSON string for session storage
            self.cleaned_df_json = df.to_json(date_format='iso', orient='split')
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}", exc_info=True)
            return False

    def load_cleaned_df_from_json(self, json_str):
        """Load cleaned_df DataFrame from a JSON string."""
        try:
            self.cleaned_df = pd.read_json(json_str, orient='split')
            return True
        except Exception as e:
            logger.error(f"Failed to load cleaned_df from JSON: {e}", exc_info=True)
            return False

    def train_model(self, model_types, target, features, params, cleaned_df_json=None):
        """
        Train one or more ML models with given parameters on the cleaned dataset.
        Optimized for memory usage.
        """
        try:
            # Load dataset - prioritize JSON if provided
            if cleaned_df_json:
                if not self.load_cleaned_df_from_json(cleaned_df_json):
                    logger.error("Failed to load cleaned_df from JSON in train_model")
                    return {"success": False, "error": "Failed to load cleaned data"}
            elif self.cleaned_df is None:
                return {"success": False, "error": "No dataset loaded"}

            # Ensure features are strings
            features = [str(f) for f in features]

            # Validate columns
            if target not in self.cleaned_df.columns:
                return {"success": False, "error": f"Target '{target}' not found"}
            for f in features:
                if f not in self.cleaned_df.columns:
                    return {"success": False, "error": f"Feature '{f}' not found"}

            X = self.cleaned_df[features].copy()
            y = self.cleaned_df[target]

            # Convert datetime features to numeric
            for f in features:
                if pd.api.types.is_datetime64_any_dtype(X[f]):
                    X[f] = X[f].map(pd.Timestamp.toordinal).fillna(X[f].map(pd.Timestamp.toordinal).median())

            # Handle missing values in target
            if y.isnull().any():
                if y.dtype == 'object':
                    y = y.fillna('Unknown')
                else:
                    y = y.fillna(y.median())

            # Identify categorical and numeric features
            categorical_features = []
            numerical_features = []
            
            for f in features:
                if X[f].dtype == 'object' or (X[f].nunique() < 20 and X[f].dtype != 'float64' and X[f].dtype != 'int64'):
                    categorical_features.append(f)
                else:
                    numerical_features.append(f)

            # Remove features with no variance
            low_variance_features = []
            for col in numerical_features:
                if X[col].nunique() <= 1:
                    low_variance_features.append(col)
            
            for col in categorical_features:
                if X[col].nunique() <= 1:
                    low_variance_features.append(col)
            
            if low_variance_features:
                logger.warning(f"Removing low variance features: {low_variance_features}")
                X = X.drop(columns=low_variance_features)
                numerical_features = [f for f in numerical_features if f not in low_variance_features]
                categorical_features = [f for f in categorical_features if f not in low_variance_features]

            # Further reduce categorical features if too many
            if len(categorical_features) > 5:
                # Keep only top 5 categorical features by correlation with target
                categorical_correlations = []
                for col in categorical_features:
                    if X[col].dtype == 'object':
                        # For categorical vs numeric correlation
                        temp_df = X[[col]].copy()
                        temp_df[col] = LabelEncoder().fit_transform(temp_df[col])
                        correlation = np.corrcoef(temp_df[col], y.fillna(y.median()))[0, 1]
                        categorical_correlations.append((col, abs(correlation)))
                
                # Sort by absolute correlation and keep top 5
                categorical_correlations.sort(key=lambda x: x[1], reverse=True)
                keep_features = [x[0] for x in categorical_correlations[:5]]
                drop_features = [f for f in categorical_features if f not in keep_features]
                
                if drop_features:
                    logger.warning(f"Dropping low-correlation categorical features: {drop_features}")
                    X = X.drop(columns=drop_features)
                    categorical_features = keep_features

            # Build preprocessor with memory optimization
            numeric_transformer = Pipeline(steps=[
                ('scaler', RobustScaler())
            ])

            # Use sparse output for one-hot encoding to save memory
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True, max_categories=20))
            ])

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            # Ensure model_types is a list
            if isinstance(model_types, str):
                model_types = [model_types]

            results = {}
            
            # Use smaller test size for large datasets
            test_size = 0.2 if len(X) < 10000 else 0.1
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Loop through each selected model
            for model_type in model_types:
                try:
                    # Select model with memory-efficient parameters
                    if model_type == 'rf':
                        model = RandomForestRegressor(
                            n_estimators=min(50, int(params.get('rf_trees', 50))),  # Reduced trees
                            max_depth=int(params.get('rf_depth', 10)),
                            random_state=42,
                            n_jobs=1,  # Single job to reduce memory
                            verbose=0
                        )
                    elif model_type == 'lr':
                        model = Ridge(alpha=float(params.get('lr_alpha', 1.0)), random_state=42)
                    elif model_type == 'gb':
                        model = GradientBoostingRegressor(
                            n_estimators=min(50, int(params.get('gb_est', 50))),  # Reduced estimators
                            learning_rate=float(params.get('gb_rate', 0.1)),
                            random_state=42,
                            verbose=0
                        )
                    else:
                        model = RandomForestRegressor(
                            n_estimators=50,
                            max_depth=10,
                            random_state=42,
                            n_jobs=1,
                            verbose=0
                        )

                    # Build pipeline
                    pipeline = Pipeline([
                        ('preprocessor', self.preprocessor),
                        ('estimator', model)
                    ])

                    # Train model with memory monitoring
                    logger.info(f"Training {model_type} with {len(X_train)} samples")
                    pipeline.fit(X_train, y_train)
                    
                    # Transform test data in chunks to save memory
                    y_pred = pipeline.predict(X_test)

                    # Calculate metrics - convert all to native Python types
                    metrics = {
                        'mae': float(mean_absolute_error(y_test, y_pred)),
                        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                        'r2': float(r2_score(y_test, y_pred)),
                        'explained_variance': float(explained_variance_score(y_test, y_pred)),
                    }

                    # Only calculate expensive metrics for small datasets
                    if len(y_test) < 10000:
                        metrics.update({
                            'max_error': float(max_error(y_test, y_pred)),
                            'mape': float(mean_absolute_percentage_error(y_test, y_pred))
                        })

                    # Use smaller cross-validation for large datasets
                    cv_folds = 3 if len(X) > 10000 else 5
                    cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring='r2')
                    metrics['cv_r2_mean'] = float(cv_scores.mean())
                    metrics['cv_r2_std'] = float(cv_scores.std())

                    # Feature importance (only for tree-based models)
                    feature_importances = {}
                    if hasattr(model, "feature_importances_"):
                        try:
                            # Get feature names after preprocessing
                            feature_names = numerical_features.copy()
                            
                            # Get categorical feature names
                            if categorical_features:
                                ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
                                cat_feature_names = ohe.get_feature_names_out(categorical_features)
                                feature_names.extend(cat_feature_names)
                            
                            if len(feature_names) == len(model.feature_importances_):
                                # Convert numpy types to native Python types
                                feature_importances = {
                                    str(feature): float(importance) 
                                    for feature, importance in zip(feature_names, model.feature_importances_.tolist())
                                }
                        except Exception as e:
                            logger.warning(f"Failed to process feature importances: {e}")

                    # Generate charts (only for successful models)
                    performance_chart = self._generate_performance_chart(metrics)
                    prediction_chart = self._generate_prediction_chart(y_test, y_pred, model_type)
                    feature_chart = self._generate_feature_importance_chart(feature_importances)

                    # Store actual predictions with IDs if available
                    predictions = []
                    if 'admit_id' in self.cleaned_df.columns:
                        # Get the indices of the test set
                        test_indices = X_test.index
                        sample_size = min(20, len(test_indices))
                        for i in range(sample_size):
                            idx = test_indices[i]
                            predictions.append({
                                'id': str(self.cleaned_df.loc[idx, 'admit_id']),
                                'actual': float(y_test.iloc[i]) if hasattr(y_test, 'iloc') else float(y_test[i]),
                                'predicted': float(y_pred[i])
                            })
                    else:
                        # Create synthetic IDs if no ID column exists
                        sample_size = min(20, len(y_test))
                        for i in range(sample_size):
                            predictions.append({
                                'id': f"test_{i}",
                                'actual': float(y_test.iloc[i]) if hasattr(y_test, 'iloc') else float(y_test[i]),
                                'predicted': float(y_pred[i])
                            })

                    # Convert model parameters to JSON-serializable format
                    serializable_params = {}
                    for param_name, param_value in model.get_params().items():
                        if isinstance(param_value, (np.integer, np.int32, np.int64)):
                            serializable_params[param_name] = int(param_value)
                        elif isinstance(param_value, (np.floating, np.float32, np.float64)):
                            serializable_params[param_name] = float(param_value)
                        elif isinstance(param_value, np.bool_):
                            serializable_params[param_name] = bool(param_value)
                        else:
                            serializable_params[param_name] = param_value

                    # Store model result
                    results[model_type] = {
                        "metrics": metrics,
                        "params": serializable_params,
                        "feature_importances": feature_importances,
                        "performance_chart": performance_chart,
                        "prediction_chart": prediction_chart,
                        "feature_importance_chart": feature_chart,
                        "predictions": predictions
                    }

                except Exception as e:
                    logger.error(f"Error training {model_type}: {e}")
                    results[model_type] = {
                        "error": str(e),
                        "metrics": {},
                        "predictions": []
                    }

            # Save last trained model info
            self.model = pipeline
            self.selected_model_type = model_types
            self.model_params = params
            self.required_columns = features
            self.training_date = datetime.now()

            return {
                "success": True,
                "target": target,
                "features": features,
                "model_type": model_types,
                "results": results
            }

        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Error in train_model: {e}\nTraceback:\n{tb_str}")
            return {"success": False, "error": str(e)}

    def _generate_feature_importance_chart(self, feature_importances):
        """Generate base64 encoded feature importance chart."""
        try:
            if not feature_importances or len(feature_importances) == 0:
                return None

            # Sort features by importance and take top 10
            sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:10]
            
            features = [f[0] for f in top_features]
            importances = [f[1] for f in top_features]

            plt.figure(figsize=(10, 6))
            plt.barh(features, importances, color='#e30613')
            plt.title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
            plt.xlabel('Importance Score', fontsize=10)
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            plt.close()
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error generating feature importance chart: {e}")
            return None

    def _generate_performance_chart(self, metrics):
        """Generate base64 encoded performance comparison chart."""
        try:
            if not metrics:
                return None

            # Prepare data for visualization
            metric_names = ['R²', 'Explained Variance', 'MAE', 'RMSE']
            metric_values = [
                metrics.get('r2', 0),
                metrics.get('explained_variance', 0),
                1 / (1 + metrics.get('mae', 1)),  # Inverse for better visualization
                1 / (1 + metrics.get('rmse', 1))   # Inverse for better visualization
            ]

            plt.figure(figsize=(8, 5))
            bars = plt.bar(metric_names, metric_values, color=['#0033A0', '#1a4ab3', '#e30613', '#b8000b'])
            
            # Add value labels on bars
            for i, v in enumerate(metric_values):
                if i < 2:  # R² and Explained Variance
                    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
                else:  # MAE and RMSE (inverse)
                    plt.text(i, v + 0.01, f'{1/v - 1:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            plt.title('Model Performance Metrics', fontsize=12, fontweight='bold')
            plt.ylabel('Score', fontsize=10)
            plt.ylim(0, 1.1)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            plt.close()
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error generating performance chart: {e}")
            return None

    def _generate_prediction_chart(self, y_true, y_pred, model_name):
        """Generate actual vs predicted scatter plot."""
        try:
            plt.figure(figsize=(8, 5))
            
            # Sample points if too many for plotting
            if len(y_true) > 1000:
                indices = np.random.choice(len(y_true), 1000, replace=False)
                y_true_sampled = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
                y_pred_sampled = y_pred[indices]
            else:
                y_true_sampled = y_true
                y_pred_sampled = y_pred
            
            # Create scatter plot
            plt.scatter(y_true_sampled, y_pred_sampled, alpha=0.6, color='#0033A0', s=10)
            
            # Add perfect prediction line
            max_val = max(max(y_true_sampled), max(y_pred_sampled))
            min_val = min(min(y_true_sampled), min(y_pred_sampled))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
            
            plt.xlabel('Actual Values', fontsize=10)
            plt.ylabel('Predicted Values', fontsize=10)
            plt.title(f'Actual vs Predicted - {model_name.upper()}', fontsize=12, fontweight='bold')
            plt.grid(alpha=0.3)
            
            # Add R² text
            r2 = r2_score(y_true, y_pred)
            plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            plt.close()
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error generating prediction chart: {e}")
            return None


@login_required
def machine_learning(request):
    """Render the main Machine Learning dashboard page."""
    manager = MLModelManager()
    context = {
        'page_title': 'Machine Learning Analysis',
        'database_tables': manager.available_tables,
    }
    return render(request, 'machine_learning.html', context)


@login_required
@require_GET
def get_table_columns(request):
    """AJAX endpoint returning columns of the selected database table."""
    table_name = request.GET.get('table', '')
    if not table_name:
        return JsonResponse({'columns': [], 'error': 'No table specified'}, status=400)

    try:
        manager = MLModelManager()
        if not manager.load_data(table_name):
            return JsonResponse({'columns': [], 'error': 'Failed to load data'}, status=500)
        
        # Get target column suggestion (prefer amount if available)
        target_suggestion = 'amount' if 'amount' in manager.data.columns else None
        
        columns = manager.get_columns()
        
        # Clean the data and store in session
        if not manager.clean_data(target_column=target_suggestion):
            return JsonResponse({'columns': [], 'error': 'Failed to clean data'}, status=500)
        
        # Store cleaned data in session
        request.session['cleaned_df_json'] = manager.cleaned_df_json
        request.session['table_name'] = table_name
        
        return JsonResponse({
            'columns': columns,
            'target_suggestion': target_suggestion
        })

    except Exception as e:
        logger.error(f"Error in get_table_columns: {e}", exc_info=True)
        return JsonResponse({'columns': [], 'error': str(e)}, status=500)


@login_required
@require_POST
def train_model(request):
    """Train machine learning models and return results."""
    try:
        # Get cleaned data from session
        cleaned_df_json = request.session.get('cleaned_df_json')
        table_name = request.session.get('table_name')
        
        if not cleaned_df_json or not table_name:
            return JsonResponse({'success': False, 'error': 'No cleaned data found. Please select a table first.'}, status=400)

        # Parse request data
        data = request.POST.dict()
        target = data.get('target')
        features = request.POST.getlist('features[]')
        model_types = request.POST.getlist('model_types[]')
        
        if not model_types:
            return JsonResponse({'success': False, 'error': 'No models selected'}, status=400)
        if not target:
            return JsonResponse({'success': False, 'error': 'No target specified'}, status=400)
        if not features:
            return JsonResponse({'success': False, 'error': 'No features selected'}, status=400)

        # Prepare parameters with memory-safe defaults
        params = {
            'rf_trees': min(50, int(data.get('rf_trees', 50))),  # Reduced for memory
            'rf_depth': int(data.get('rf_depth', 10)),
            'gb_est': min(50, int(data.get('gb_est', 50))),      # Reduced for memory
            'gb_rate': float(data.get('gb_rate', 0.1)),
            'lr_alpha': float(data.get('lr_alpha', 1.0))
        }

        # Train models
        manager = MLModelManager()
        result = manager.train_model(
            model_types=model_types,
            target=target,
            features=features,
            params=params,
            cleaned_df_json=cleaned_df_json
        )

        # Use custom JSON encoder to handle NumPy types
        return JsonResponse(result, encoder=NumpyJSONEncoder)

    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Exception in train_model: {e}\n{tb_str}")
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
    

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.db import connection
from django.contrib.auth.decorators import login_required
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import csv
from io import BytesIO
import logging
from .models import ClaimRecord

# Set up logging
logger = logging.getLogger(__name__)

# =======================
# Fraud Detection Engine
# =======================

class FraudDetector:
    """Enhanced rule-based fraud detection engine"""
    
    def __init__(self):
        self.clinical_rules = self._load_clinical_rules()
        # Adjusted rule weights for better sensitivity
        self.rule_weights = {
            'member_frequency_score': 0.25,
            'provider_frequency_score': 0.20,
            'claim_amount_score': 0.20,
            'upcoding_score': 0.25,
            'treatment_mismatch_score': 0.25,
            'age_mismatch_score': 0.10,
            'collusion_risk_score': 0.20,
            'temporal_anomaly_score': 0.10,
            'provider_flag_rate': 0.15
        }
    
    def _load_clinical_rules(self):
        """Load clinical validation rules with expanded mappings"""
        return {
            'ICD10_CPT_mappings': {
                'J18.9': ['99203', '99213', 'J0698', '99214', '99215'],
                'E11.65': ['25000', '83036', 'A4215', 'G0202', '82947'],
                'I10': ['99212', '99213', '99214', '99215'],
                'M54.5': ['97140', '97110', '97530']
            },
            'age_treatment_guidelines': {
                '99203': (0, 120),
                'J0698': (2, 120),
                '97110': (5, 120),
                '97530': (3, 120)
            }
        }

    def preprocess_data(self, df):
        """Enhanced data cleaning and preparation"""
        # Standardize column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Ensure required columns exist
        required_cols = ['claim_id', 'provider_id', 'member_id', 'treatment_code', 
                        'diagnosis_code', 'claim_amount', 'submission_date']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        # Create claim_id if missing
        if df['claim_id'].isnull().all():
            df['claim_id'] = [f"CLM_{i}" for i in range(1, len(df)+1)]
        
        # Handle dates robustly
        date_cols = [col for col in df.columns if 'date' in col]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        if df['submission_date'].isnull().all():
            df['submission_date'] = pd.to_datetime('today')
        
        # Calculate member age with better handling
        df['member_age'] = 30  # Default
        if 'dob' in df.columns:
            df['member_age'] = (df['submission_date'].dt.year - 
                               pd.to_datetime(df['dob']).dt.year)
            df['member_age'] = df['member_age'].clip(0, 120)  # Reasonable bounds
        
        # Enhanced missing value handling
        fill_values = {
            'provider_id': 'UNKNOWN',
            'provider_type': 'UNKNOWN',
            'treatment_code': 'UNSPECIFIED',
            'diagnosis_code': 'UNSPECIFIED',
            'claim_amount': 0
        }
        
        # Handle claim amount specially
        if 'claim_amount' in df.columns:
            df['claim_amount'] = pd.to_numeric(df['claim_amount'], errors='coerce')
            valid_claims = df['claim_amount'].notna() & (df['claim_amount'] >= 0)
            fill_values['claim_amount'] = df.loc[valid_claims, 'claim_amount'].median()
            df['claim_amount_log'] = np.log1p(df['claim_amount'].fillna(fill_values['claim_amount']))
        
        df.fillna(fill_values, inplace=True)
        
        # Create enhanced features
        df = self._create_temporal_features(df)
        df = self._create_network_features(df)
        df = self._create_clinical_features(df)
        
        return df
    
    def _create_temporal_features(self, df):
        """Enhanced time-based features"""
        try:
            if 'submission_date' in df.columns:
                # Member claim patterns
                if 'member_id' in df.columns:
                    df['member_claim_count_7d'] = df.groupby('member_id')['claim_id'].transform(
                        lambda x: x.rolling('7D', on='submission_date').count())
                    df['member_claim_count_30d'] = df.groupby('member_id')['claim_id'].transform(
                        lambda x: x.rolling('30D', on='submission_date').count())
                
                # Provider claim patterns
                if 'provider_id' in df.columns:
                    df['provider_claim_count_7d'] = df.groupby('provider_id')['claim_id'].transform(
                        lambda x: x.rolling('7D', on='submission_date').count())
                    df['provider_claim_count_30d'] = df.groupby('provider_id')['claim_id'].transform(
                        lambda x: x.rolling('30D', on='submission_date').count())
                
                # Time-based features
                df['submission_hour'] = df['submission_date'].dt.hour
                df['submission_day'] = df['submission_date'].dt.dayofweek
                df['is_after_hours'] = ((df['submission_hour'] < 8) | (df['submission_hour'] > 18))
                df['is_weekend'] = df['submission_day'].isin([5, 6])

        except Exception as e:
            logger.error(f"Error creating temporal features: {str(e)}")
        
        return df
    
    def _create_network_features(self, df):
        """Enhanced network analysis features"""
        try:
            if 'provider_id' in df.columns and 'member_id' in df.columns:
                # Provider network metrics
                df['provider_degree'] = df.groupby('provider_id')['member_id'].transform('nunique')
                df['provider_claim_freq'] = df.groupby('provider_id')['claim_id'].transform('count')
                
                # Member network metrics
                df['member_degree'] = df.groupby('member_id')['provider_id'].transform('nunique')
                df['member_claim_freq'] = df.groupby('member_id')['claim_id'].transform('count')
                
                # Provider-member relationship strength
                df['provider_member_freq'] = df.groupby(['provider_id', 'member_id'])['claim_id'].transform('count')
                
        except Exception as e:
            logger.error(f"Error creating network features: {str(e)}")
        
        return df
    
    def _create_clinical_features(self, df):
        """Enhanced clinical validation features"""
        try:
            if 'treatment_code' in df.columns and 'diagnosis_code' in df.columns:
                # Treatment-diagnosis validity
                df['treatment_valid'] = df.apply(
                    lambda x: x['treatment_code'] in self.clinical_rules['ICD10_CPT_mappings'].get(
                        x['diagnosis_code'], []),
                    axis=1
                )
                
                # Age appropriateness
                df['age_appropriate'] = df.apply(
                    lambda x: self._check_age_appropriate(x['treatment_code'], x.get('member_age', 30)),
                    axis=1
                )
                
                # Treatment frequency analysis
                df['treatment_freq'] = df.groupby('treatment_code')['treatment_code'].transform('count')
                
        except Exception as e:
            logger.error(f"Error creating clinical features: {str(e)}")
        
        return df
    
    def _check_age_appropriate(self, treatment_code, age):
        """Enhanced age-appropriateness check"""
        age_range = self.clinical_rules['age_treatment_guidelines'].get(treatment_code, (0, 120))
        return age_range[0] <= age <= age_range[1]
    
    def calculate_rule_scores(self, df):
        """Enhanced fraud scoring with better sensitivity"""
        try:
            # Frequency-based scores
            df['member_frequency_score'] = self._calculate_member_frequency_score(df)
            df['provider_frequency_score'] = self._calculate_provider_frequency_score(df)
            
            # Amount-based scores
            df['claim_amount_score'] = self._calculate_amount_score(df)
            df['upcoding_score'] = self._calculate_upcoding_score(df)
            
            # Clinical validity
            df['treatment_mismatch_score'] = (~df['treatment_valid']).astype(int)
            df['age_mismatch_score'] = (~df['age_appropriate']).astype(int)
            
            # Network analysis
            df['collusion_risk_score'] = self._calculate_collusion_risk(df)
            
            # Temporal patterns
            df['temporal_anomaly_score'] = self._calculate_temporal_anomalies(df)
            
            # Provider flag rate
            df['provider_flag_rate'] = self._calculate_provider_flag_rate(df)
            
            # Composite score with enhanced sensitivity
            df['rule_based_score'] = self._calculate_composite_score(df)
            
            # Triggered rules
            df['triggered_rules'] = df.apply(self._get_triggered_rules, axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating rule scores: {str(e)}")
            return df
    
    def _calculate_member_frequency_score(self, df):
        """Enhanced member frequency analysis"""
        if 'member_claim_count_7d' not in df.columns:
            return 0
        
        # Use robust statistics
        median = df['member_claim_count_7d'].median()
        mad = (df['member_claim_count_7d'] - median).abs().median()
        
        # Modified z-score
        scores = 0.6745 * (df['member_claim_count_7d'] - median) / (mad + 1e-6)
        
        # Normalize to 0-1 range
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    
    def _calculate_provider_frequency_score(self, df):
        """Enhanced provider frequency analysis"""
        if 'provider_claim_count_7d' not in df.columns:
            return 0
            
        # Use robust statistics
        median = df['provider_claim_count_7d'].median()
        mad = (df['provider_claim_count_7d'] - median).abs().median()
        
        # Modified z-score
        scores = 0.6745 * (df['provider_claim_count_7d'] - median) / (mad + 1e-6)
        
        # Normalize to 0-1 range
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    
    def _calculate_amount_score(self, df):
        """Enhanced claim amount analysis"""
        if 'claim_amount_log' not in df.columns:
            return 0
            
        # Use robust statistics
        median = df['claim_amount_log'].median()
        mad = (df['claim_amount_log'] - median).abs().median()
        
        # Modified z-score
        scores = 0.6745 * (df['claim_amount_log'] - median) / (mad + 1e-6)
        
        # Normalize to 0-1 range
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    
    def _calculate_upcoding_score(self, df):
        """Enhanced upcoding detection"""
        if 'provider_type' not in df.columns or 'claim_amount' not in df.columns:
            return 0
            
        # Calculate specialty averages
        specialty_avg = df.groupby('provider_type')['claim_amount'].transform('median')
        
        # Calculate deviation from specialty average
        deviation = (df['claim_amount'] - specialty_avg) / (specialty_avg + 1e-6)
        
        # Normalize to 0-1 range
        return (deviation - deviation.min()) / (deviation.max() - deviation.min() + 1e-6)
    
    def _calculate_collusion_risk(self, df):
        """Enhanced collusion risk detection"""
        if 'provider_member_freq' not in df.columns:
            return 0
            
        # Calculate provider-member relationship strength
        strength = df['provider_member_freq'] / df.groupby('member_id')['claim_id'].transform('count')
        
        # Normalize to 0-1 range
        return (strength - strength.min()) / (strength.max() - strength.min() + 1e-6)
    
    def _calculate_temporal_anomalies(self, df):
        """Enhanced temporal anomaly detection"""
        if 'is_after_hours' not in df.columns or 'is_weekend' not in df.columns:
            return 0
            
        # Combine multiple temporal indicators
        score = (df['is_after_hours'].astype(int) * 0.5 + 
                df['is_weekend'].astype(int) * 0.5)
        
        return score
    
    def _calculate_provider_flag_rate(self, df):
        """Calculate provider historical flag rate"""
        # Placeholder - would need historical data
        return 0
    
    def _calculate_composite_score(self, df):
        """Enhanced composite score calculation"""
        scores = []
        for rule, weight in self.rule_weights.items():
            if rule in df.columns:
                # Apply weighting and scale for better sensitivity
                weighted_score = df[rule] * weight * 10
                scores.append(weighted_score)
        
        # Normalize final score between 0-1
        total_score = sum(scores)
        max_possible = sum(self.rule_weights.values()) * 10
        return total_score / (max_possible + 1e-6)
    
    def _get_triggered_rules(self, row):
        """Get list of triggered rules with enhanced thresholds"""
        triggered = []
        threshold = 0.5  # Lowered threshold for more sensitivity
        
        for rule, weight in self.rule_weights.items():
            if rule in row.index and row[rule] * weight >= threshold:
                triggered.append({
                    'name': rule.replace('_score', '').replace('_', ' ').title(),
                    'description': self._get_rule_description(rule),
                    'score': row[rule],
                    'weight': weight,
                    'weighted_score': row[rule] * weight
                })
        
        # Sort by weighted score descending
        triggered.sort(key=lambda x: x['weighted_score'], reverse=True)
        return triggered
    
    def _get_rule_description(self, rule_name):
        """Enhanced rule descriptions"""
        descriptions = {
            'member_frequency_score': 'Unusually high claim frequency for member',
            'provider_frequency_score': 'Abnormal billing pattern for provider',
            'claim_amount_score': 'Unusually high claim amount',
            'upcoding_score': 'Possible upcoding to more expensive services',
            'treatment_mismatch_score': 'Treatment not matching diagnosis',
            'age_mismatch_score': 'Treatment not appropriate for patient age',
            'collusion_risk_score': 'Potential provider-member collusion pattern',
            'temporal_anomaly_score': 'Unusual submission time patterns',
            'provider_flag_rate': 'Provider historical fraud flag rate'
        }
        return descriptions.get(rule_name, 'Potential fraud indicator')

# =======================
# Fraud Detection Views
# =======================

def get_database_tables():
    """
    Return all user-defined tables from the database.
    Supports SQLite and PostgreSQL.
    Excludes internal system tables and Django's default tables.
    """
    vendor = connection.vendor  # 'sqlite', 'postgresql', 'mysql', etc.

    with connection.cursor() as cursor:
        if vendor == 'sqlite':
            cursor.execute("""
                SELECT name 
                FROM sqlite_master 
                WHERE type='table' 
                  AND name NOT LIKE 'sqlite_%'
                  AND name NOT LIKE 'django_%'
                  AND name NOT LIKE 'auth_%'
                  AND name NOT LIKE 'sessions%'
                ORDER BY name
            """)
        elif vendor == 'postgresql':
            cursor.execute("""
                SELECT tablename 
                FROM pg_catalog.pg_tables 
                WHERE schemaname='public'
                  AND tablename NOT LIKE 'django_%'
                  AND tablename NOT LIKE 'auth_%'
                  AND tablename NOT LIKE 'sessions%'
                ORDER BY tablename
            """)
        else:
            raise NotImplementedError(
                f"Database vendor '{vendor}' is not supported yet."
            )

        return [row[0] for row in cursor.fetchall()]

def get_table_data(table_name):
    """Return all data from a specific table as a DataFrame with error handling"""
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {table_name}")
            columns = [col[0] for col in cursor.description]
            data = cursor.fetchall()
            
            if not data:
                return pd.DataFrame(columns=columns)
                
            return pd.DataFrame(data, columns=columns)
            
    except Exception as e:
        logger.error(f"Error fetching data from {table_name}: {str(e)}")
        return pd.DataFrame()

from django.views.decorators.csrf import csrf_exempt

@login_required
def fraud_rule_based(request):
    """Enhanced fraud detection view with dataset selection and comprehensive reporting"""
    context = {
        'available_datasets': get_database_tables(),
        'default_threshold': 0.5
    }
    
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'run_analysis':
            try:
                threshold = float(request.POST.get('threshold', 0.5))
                dataset_name = request.POST.get('dataset')
                
                if not dataset_name:
                    context['error'] = "Please select a dataset to analyze"
                    return render(request, 'fraud_rule_based.html', context)
                
                logger.info(f"Starting fraud analysis on {dataset_name} with threshold {threshold}")
                
                # Get and process data
                df = get_table_data(dataset_name)
                
                if df.empty:
                    context['error'] = f"Dataset '{dataset_name}' is empty or could not be loaded"
                    return render(request, 'fraud_rule_based.html', context)
                
                # Standardize column names to match our model
                column_mapping = {}
                if 'prov_name' in df.columns:
                    column_mapping['prov_name'] = 'provider_name'
                if 'claim_prov_date' in df.columns:
                    column_mapping['claim_prov_date'] = 'date'
                
                df = df.rename(columns=column_mapping)
                
                # Run enhanced fraud detection
                df = detect_fraud_anomalies(df)
                
                # Calculate metrics
                fraud_count = int(df['fraud_flag'].sum())
                fraud_rate = fraud_count / len(df) if len(df) > 0 else 0
                
                # Calculate fraud amount based on available amount column
                amount_column = None
                for col in ['amount', 'claim_amount', 'total_amount']:
                    if col in df.columns:
                        amount_column = col
                        break
                
                fraud_amount = 0
                if amount_column:
                    fraud_amount = df.loc[df['fraud_flag'] == 1, amount_column].sum()
                
                # Get high risk claims
                high_risk = df[df['fraud_flag'] == 1].sort_values('fraud_score', ascending=False)
                high_risk_records = high_risk.to_dict('records')
                
                # Calculate suspicious providers if data available
                suspicious_providers = []
                provider_column = None
                for col in ['provider_name', 'prov_name', 'provider']:
                    if col in df.columns:
                        provider_column = col
                        break
                
                if provider_column and amount_column:
                    provider_fraud = df.groupby(provider_column).agg(
                        total_amount=(amount_column, 'sum'),
                        fraud_count=('fraud_flag', 'sum'),
                        total_claims=('fraud_flag', 'count')
                    ).reset_index()
                    provider_fraud['fraud_rate'] = provider_fraud['fraud_count'] / provider_fraud['total_claims']
                    suspicious_providers = provider_fraud.sort_values('fraud_count', ascending=False).head(10).to_dict('records')
                
                # Calculate diagnosis patterns if data available
                diagnosis_patterns = []
                diagnosis_column = None
                for col in ['icd10_code', 'diagnosis_code', 'diagnosis']:
                    if col in df.columns:
                        diagnosis_column = col
                        break
                
                if diagnosis_column and amount_column:
                    diagnosis_fraud = df.groupby(diagnosis_column).agg(
                        total_amount=(amount_column, 'sum'),
                        fraud_count=('fraud_flag', 'sum'),
                        total_claims=('fraud_flag', 'count')
                    ).reset_index()
                    diagnosis_fraud['fraud_rate'] = diagnosis_fraud['fraud_count'] / diagnosis_fraud['total_claims']
                    diagnosis_patterns = diagnosis_fraud[diagnosis_fraud['total_claims'] > 10].sort_values(
                        'fraud_rate', ascending=False
                    ).head(10).to_dict('records')
                
                # Risk score distribution
                bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
                df['score_bin'] = pd.cut(df['fraud_score'], bins=bins, labels=bin_labels, include_lowest=True)
                bin_counts = df['score_bin'].value_counts().reindex(bin_labels, fill_value=0).tolist()
                
                # Prepare context for template
                context.update({
                    'analysis_complete': True,
                    'dataset_name': dataset_name,
                    'high_risk': high_risk_records,
                    'threshold': threshold,
                    'metrics': {
                        'fraud_count': fraud_count,
                        'fraud_rate': f"{fraud_rate:.1%}",
                        'fraud_amount': f"KES {fraud_amount:,.2f}" if fraud_amount else "N/A",
                        'total_claims': len(df)
                    },
                    'suspicious_providers': suspicious_providers,
                    'diagnosis_patterns': diagnosis_patterns,
                    'risk_distribution_data': json.dumps({'bins': bin_labels, 'counts': bin_counts})
                })
                
                # Store results in session for download
                request.session['fraud_results'] = df.to_json()
                request.session['fraud_threshold'] = threshold
                request.session['fraud_dataset'] = dataset_name
                
            except Exception as e:
                logger.error(f"Error during fraud analysis: {str(e)}", exc_info=True)
                context['error'] = f"Error processing dataset: {str(e)}"
                return render(request, 'fraud_rule_based.html', context)
        
        elif action == 'download_results':
            if 'fraud_results' in request.session:
                try:
                    results = pd.read_json(request.session['fraud_results'])
                    threshold = request.session.get('fraud_threshold', 0.5)
                    dataset_name = request.session.get('fraud_dataset', 'dataset')
                    
                    response = HttpResponse(content_type='text/csv')
                    response['Content-Disposition'] = (
                        f'attachment; filename="fraud_results_{dataset_name}_threshold_{threshold}.csv"'
                    )
                    
                    results.to_csv(response, index=False)
                    return response
                    
                except Exception as e:
                    logger.error(f"Error generating download: {str(e)}")
                    return HttpResponse("Error generating download", status=500)
    
    return render(request, 'fraud_rule_based.html', context)


#########################################
############################################
##############################################
######################################################


import base64
import json
import logging
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.db import connection
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# Helper functions
def get_database_tables():
    """Get list of available database tables across different database backends."""
    vendor = connection.vendor  # 'sqlite', 'postgresql', 'mysql', etc.

    with connection.cursor() as cursor:
        if vendor == "sqlite":
            cursor.execute("""
                SELECT name 
                FROM sqlite_master 
                WHERE type='table' 
                AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
        elif vendor == "postgresql":
            cursor.execute("""
                SELECT tablename 
                FROM pg_catalog.pg_tables 
                WHERE schemaname='public'
                ORDER BY tablename
            """)
        elif vendor == "mysql":
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = DATABASE()
                ORDER BY table_name
            """)
        else:
            raise NotImplementedError(f"Database vendor '{vendor}' is not supported yet.")

        return [row[0] for row in cursor.fetchall()]

def get_table_data(table_name):
    """Get data from specified table"""
    return pd.read_sql(f'SELECT * FROM "{table_name}"', connection)

def generate_flagging_reason(row):
    """Generate human-readable flagging reasons for claims"""
    reasons = []
    
    # Risk score based flagging
    if 'Final_Risk_Score' in row and row['Final_Risk_Score'] < 0.4:
        reasons.append(f"High risk score ({row['Final_Risk_Score']:.2f})")
    
    # Provider risk flagging
    if 'PRS' in row and row['PRS'] > 0.7:
        reasons.append(f"High provider risk ({row['PRS']:.2f})")
    
    # Anomaly detection flagging
    if 'Anomaly_Score' in row and row['Anomaly_Score'] > 0.9:
        reasons.append("Anomaly detected")
    
    # Reconstruction error flagging
    if 'Reconstruction_Error' in row and row['Reconstruction_Error'] > 0.95:
        reasons.append("Unusual pattern detected")
    
    # Cluster-based flagging
    if 'Risk_Label' in row and row['Risk_Label'] == 'High Risk':
        reasons.append("High risk cluster")
    
    # Claim amount flagging
    if 'Claim_Amount' in row and row['Claim_Amount'] > 50000:  # Example threshold
        reasons.append("High claim amount")
    
    # Return the combined reasons or a default message
    return " | ".join(reasons) if reasons else "Multiple risk factors"

def calculate_fraud_scores(df):
    """Calculate comprehensive fraud risk scores"""
    try:
        df = df.copy()
        
        # Treatment Consistency Score (TCS)
        if 'Treatment_Type' in df.columns and 'Customer_ID' in df.columns:
            valid_customers = df['Customer_ID'].notna()
            treatment_counts = df[valid_customers].groupby('Customer_ID')['Treatment_Type'].nunique()
            
            df['TCS'] = 0.5
            valid_mask = valid_customers & df['Treatment_Type'].notna()
            if treatment_counts.max() > 0:
                df.loc[valid_mask, 'TCS'] = (
                    df[valid_mask]['Customer_ID'].map(treatment_counts) / 
                    treatment_counts.max()
                )
        else:
            df['TCS'] = 0.5
        
        # Claim Amount Deviation Score (CADS)
        if 'Claim_Amount' in df.columns and 'Provider_Type' in df.columns:
            provider_avg = df.groupby('Provider_Type')['Claim_Amount'].transform('mean')
            df['CADS'] = abs(df['Claim_Amount'] - provider_avg) / provider_avg.replace(0, 1)
            df['CADS'] = df['CADS'] / df['CADS'].max() if df['CADS'].max() > 0 else 0.5
        else:
            df['CADS'] = 0.5
        
        # Visit Frequency Score (VFS)
        if 'Customer_ID' in df.columns:
            claim_counts = df[df['Customer_ID'].notna()].groupby('Customer_ID')['Claim_ID'].transform('count')
            df['VFS'] = claim_counts / claim_counts.max() if claim_counts.max() > 0 else 0.5
        else:
            df['VFS'] = 0.5
        
        # Provider Risk Score (PRS)
        provider_risk = {
            'Hospital': 0.8,
            'Clinic': 0.5,
            'Pharmacy': 0.3,
            'Laboratory': 0.6,
            'Unknown': 0.5
        }
        df['PRS'] = df['Provider_Type'].map(provider_risk).fillna(0.5) if 'Provider_Type' in df.columns else 0.5
        
        # Geographic Distance Score (GDS)
        if all(col in df.columns for col in ['Customer_Lat', 'Customer_Lon', 'Provider_Lat', 'Provider_Lon']):
            valid_coords = (
                df['Customer_Lat'].notna() & 
                df['Customer_Lon'].notna() & 
                df['Provider_Lat'].notna() & 
                df['Provider_Lon'].notna()
            )
            if valid_coords.any():
                coords = df[valid_coords][['Customer_Lat', 'Customer_Lon', 'Provider_Lat', 'Provider_Lon']].values
                distances = np.array([cdist([x[:2]], [x[2:]], 'euclidean')[0][0] for x in coords])
                df.loc[valid_coords, 'GDS'] = distances / distances.max()
            df['GDS'] = df['GDS'].fillna(0.5)
        else:
            df['GDS'] = 0.5
        
        # Calculate final composite score
        weights = {
            'TCS': 0.15,
            'CADS': 0.20,
            'VFS': 0.10,
            'PRS': 0.15,
            'GDS': 0.10,
        }
        
        valid_weights = {k: v for k, v in weights.items() if k in df.columns}
        weight_sum = sum(valid_weights.values())
        normalized_weights = {k: v/weight_sum for k, v in valid_weights.items()}
        
        df['Final_Risk_Score'] = sum(df[col] * weight for col, weight in normalized_weights.items())
        
        min_score = df['Final_Risk_Score'].min()
        max_score = df['Final_Risk_Score'].max()
        if max_score > min_score:
            df['Final_Risk_Score'] = (df['Final_Risk_Score'] - min_score) / (max_score - min_score)
        else:
            df['Final_Risk_Score'] = 0.5
        
        # Generate flagging reasons for all claims
        df['Flagging_Reason'] = df.apply(generate_flagging_reason, axis=1)
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating fraud scores: {str(e)}")
        if 'Final_Risk_Score' not in df.columns:
            df['Final_Risk_Score'] = 0.5
        if 'Flagging_Reason' not in df.columns:
            df['Flagging_Reason'] = "Risk calculation error"
        return df

def detect_anomalies(df):
    """Detect anomalies using Isolation Forest"""
    try:
        features = ['Claim_Amount', 'TCS', 'CADS', 'VFS', 'PRS']
        features = [f for f in features if f in df.columns]
        
        if len(features) < 3:
            logger.warning("Insufficient features for anomaly detection")
            df['Anomaly_Score'] = 0.5
            return df
        
        scaler = StandardScaler()
        X = scaler.fit_transform(df[features])
        
        clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        clf.fit(X)
        
        df['Anomaly_Score'] = clf.decision_function(X)
        df['Anomaly_Score'] = (df['Anomaly_Score'] - df['Anomaly_Score'].min()) / \
                             (df['Anomaly_Score'].max() - df['Anomaly_Score'].min())
        
        return df
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        df['Anomaly_Score'] = 0.5
        return df

def enhanced_cluster_claims(df):
    """Enhanced claim clustering using K-Means"""
    try:
        features = ['Final_Risk_Score', 'Claim_Amount', 'Anomaly_Score']
        features = [f for f in features if f in df.columns]
        
        if len(features) < 2:
            logger.warning("Insufficient features for clustering")
            df['Risk_Label'] = 'Medium Risk'
            return df
        
        scaler = StandardScaler()
        X = scaler.fit_transform(df[features])
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        df['PC1'] = X_pca[:, 0]
        df['PC2'] = X_pca[:, 1]
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)
        
        cluster_risk = {
            0: 'Low Risk',
            1: 'Medium Risk',
            2: 'High Risk'
        }
        df['Risk_Label'] = df['Cluster'].map(cluster_risk)
        
        return df
        
    except Exception as e:
        logger.error(f"Error in claim clustering: {str(e)}")
        df['Risk_Label'] = 'Medium Risk'
        return df

def generate_executive_summary(data):
    """Generate comprehensive executive summary with dynamic insights"""
    summary = {}
    
    summary["Total_Claims_Analyzed"] = len(data)
    summary["Flagged_Claims"] = data['Flagged'].sum() if 'Flagged' in data.columns else 0
    summary["Flagging_Rate"] = f"{data['Flagged'].mean() * 100:.1f}%" if 'Flagged' in data.columns else "0%"
    summary["Average_Risk_Score_All_Claims"] = f"{data['Final_Risk_Score'].mean():.2f}" if 'Final_Risk_Score' in data.columns else "N/A"
    
    if 'Flagged' in data.columns and data['Flagged'].sum() > 0:
        summary["Average_Risk_Score_Flagged_Claims"] = f"{data[data['Flagged']]['Final_Risk_Score'].mean():.2f}"
    else:
        summary["Average_Risk_Score_Flagged_Claims"] = "N/A"
    
    # Claims amount distribution
    if 'Claim_Amount' in data.columns:
        median_claim = data['Claim_Amount'].median()
        if median_claim < 5000:
            summary["Claims_Amount_Distribution"] = "mostly small claims (median < KES 5,000)"
        elif median_claim < 20000:
            summary["Claims_Amount_Distribution"] = "moderate value claims (median KES 5,000-20,000)"
        else:
            summary["Claims_Amount_Distribution"] = "high value claims (median > KES 20,000)"
    else:
        summary["Claims_Amount_Distribution"] = "N/A"
    
    # Claims trend analysis
    if 'Submission_Date' in data.columns:
        try:
            data['Submission_Date'] = pd.to_datetime(data['Submission_Date'])
            daily_counts = data.groupby(data['Submission_Date'].dt.date).size()
            
            if len(daily_counts) > 1:
                trend = "stable submission patterns"
                pct_change = daily_counts.pct_change().mean()
                
                if pct_change > 0.05:
                    trend = f"an increasing trend ({pct_change:.0%} daily growth)"
                elif pct_change < -0.05:
                    trend = f"a decreasing trend ({abs(pct_change):.0%} daily decline)"
                
                if daily_counts.std() / daily_counts.mean() > 0.3:
                    trend += " with significant volatility"
            else:
                trend = "insufficient data for trend analysis"
        except:
            trend = "could not analyze trends due to date format issues"
    else:
        trend = "no timestamp data available for trend analysis"
    summary["Claims_Trend_Analysis"] = trend
    
    # Risk score distribution
    if 'Final_Risk_Score' in data.columns:
        risk_skew = data['Final_Risk_Score'].skew()
        if risk_skew > 0.5:
            summary["Risk_Score_Distribution"] = "positively skewed with more low-risk claims"
        elif risk_skew < -0.5:
            summary["Risk_Score_Distribution"] = "negatively skewed with more high-risk claims"
        else:
            summary["Risk_Score_Distribution"] = "relatively balanced distribution"
    else:
        summary["Risk_Score_Distribution"] = "N/A"
    
    # Risk cluster findings
    if 'Risk_Label' in data.columns:
        cluster_summary = data['Risk_Label'].value_counts().to_dict()
        cluster_text = ", ".join([f"{k}: {v}" for k, v in cluster_summary.items()])
        summary["Risk_Cluster_Findings"] = f"{cluster_text} clusters identified"
    else:
        summary["Risk_Cluster_Findings"] = "no clustering performed"
    
    # Provider patterns
    if 'Provider_Type' in data.columns and 'Flagged' in data.columns:
        flagged_providers = data[data['Flagged']]['Provider_Type'].value_counts().nlargest(3)
        if not flagged_providers.empty:
            summary["Common_Providers_in_Flagged_Claims"] = f"({', '.join(flagged_providers.index)})"
        else:
            summary["Common_Providers_in_Flagged_Claims"] = "no common patterns identified"
    else:
        summary["Common_Providers_in_Flagged_Claims"] = "no provider data available"
    
    # Treatment patterns
    if 'Treatment_Type' in data.columns and 'Flagged' in data.columns:
        flagged_treatments = data[data['Flagged']]['Treatment_Type'].value_counts().nlargest(3)
        if not flagged_treatments.empty:
            summary["Common_Treatments_in_Flagged_Claims"] = f"({', '.join(flagged_treatments.index)})"
        else:
            summary["Common_Treatments_in_Flagged_Claims"] = "no common patterns identified"
    else:
        summary["Common_Treatments_in_Flagged_Claims"] = "no treatment data available"
    
    # Key findings
    findings = []
    flagged_count = data['Flagged'].sum() if 'Flagged' in data.columns else 0
    
    if flagged_count > 0:
        findings.append(f"Identified {flagged_count} potentially suspicious claims requiring review.")
        
        if 'Provider_Type' in data.columns:
            top_provider = data[data['Flagged']]['Provider_Type'].value_counts().index[0] if data['Flagged'].sum() > 0 else "N/A"
            findings.append(f"Provider type '{top_provider}' appears most frequently in flagged claims.")
        
        if 'Treatment_Type' in data.columns and data['Flagged'].sum() > 0:
            top_treatment = data[data['Flagged']]['Treatment_Type'].value_counts().index[0]
            findings.append(f"Treatment type '{top_treatment}' is most common in flagged claims.")
        
        if 'Final_Risk_Score' in data.columns:
            low_score = data['Final_Risk_Score'].quantile(0.1)
            findings.append(f"10% of claims have risk scores below {low_score:.2f} indicating high suspicion.")
    else:
        findings.append("No claims were flagged as suspicious based on current thresholds.")
    
    summary["Key_Findings"] = " ".join(findings)
    
    # Recommendations
    recommendations = []
    
    if flagged_count > 10:
        recommendations.append("Prioritize investigation of top 10 highest-risk claims.")
    
    if 'Provider_Type' in data.columns and flagged_count > 0:
        recommendations.append("Review claims from providers with highest flag rates.")
    
    if 'Treatment_Type' in data.columns and flagged_count > 0:
        recommendations.append("Audit claims for most common suspicious treatments.")
    
    if 'Final_Risk_Score' in data.columns and data['Final_Risk_Score'].min() < 0.2:
        recommendations.append("Implement enhanced due diligence for claims with risk scores < 0.2.")
    
    if not recommendations:
        recommendations.append("Continue monitoring with current thresholds as no suspicious patterns were detected.")
    
    summary["Recommendations"] = recommendations
    
    return summary

def create_risk_score_chart(df, flagged_column='Flagged'):
    """Create risk score distribution chart as base64 encoded image"""
    try:
        plt.figure(figsize=(10, 6))
        
        if flagged_column in df.columns:
            flagged_data = df[df[flagged_column]]['Final_Risk_Score']
            non_flagged_data = df[~df[flagged_column]]['Final_Risk_Score']
            
            plt.hist([non_flagged_data, flagged_data], bins=20, alpha=0.7, 
                    label=['Non-Flagged', 'Flagged'], color=['blue', 'red'])
        else:
            plt.hist(df['Final_Risk_Score'], bins=20, alpha=0.7, color='blue')
        
        plt.xlabel('Risk Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Fraud Risk Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    except Exception as e:
        logger.error(f"Error creating risk score chart: {str(e)}")
        return None

def create_provider_risk_chart(df, flagged_column='Flagged'):
    """Create provider risk chart as base64 encoded image"""
    try:
        if 'Provider_Type' not in df.columns or flagged_column not in df.columns:
            return None
            
        provider_risk = df[df[flagged_column]].groupby('Provider_Type').size().nlargest(10)
        
        plt.figure(figsize=(10, 6))
        provider_risk.plot(kind='bar', color='red')
        plt.xlabel('Provider Type')
        plt.ylabel('Number of Flagged Claims')
        plt.title('Top Risky Providers by Flagged Claims')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Save to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    except Exception as e:
        logger.error(f"Error creating provider risk chart: {str(e)}")
        return None

def create_treatment_risk_chart(df, flagged_column='Flagged'):
    """Create treatment risk chart as base64 encoded image"""
    try:
        if 'Treatment_Type' not in df.columns or flagged_column not in df.columns:
            return None
            
        treatment_risk = df[df[flagged_column]].groupby('Treatment_Type').size().nlargest(10)
        
        plt.figure(figsize=(10, 6))
        treatment_risk.plot(kind='bar', color='orange')
        plt.xlabel('Treatment Type')
        plt.ylabel('Number of Flagged Claims')
        plt.title('Most Common Treatments in Flagged Claims')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Save to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    except Exception as e:
        logger.error(f"Error creating treatment risk chart: {str(e)}")
        return None

def create_claims_over_time_chart(df, date_column='Submission_Date', flagged_column='Flagged'):
    """Create claims over time chart as base64 encoded image"""
    try:
        if date_column not in df.columns or flagged_column not in df.columns:
            return None
            
        # Convert to datetime and extract date
        df['Date'] = pd.to_datetime(df[date_column]).dt.date
        
        # Group by date
        daily_counts = df.groupby('Date').size()
        daily_flagged = df[df[flagged_column]].groupby('Date').size()
        
        # Create figure
        plt.figure(figsize=(12, 6))
        plt.plot(daily_counts.index, daily_counts.values, label='Total Claims', marker='o')
        plt.plot(daily_flagged.index, daily_flagged.values, label='Flagged Claims', marker='s', color='red')
        
        plt.xlabel('Date')
        plt.ylabel('Number of Claims')
        plt.title('Claims Submission Trend Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Save to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    except Exception as e:
        logger.error(f"Error creating claims over time chart: {str(e)}")
        return None




@csrf_exempt
@login_required
def create_fraud_model(request):
    """AJAX endpoint to train fraud detection models"""
    if request.method == "POST":
        try:
            target = request.POST.get("target")
            model_types = request.POST.getlist("model_types[]")
            features = request.POST.getlist("features[]")

            if not target or not model_types or not features:
                return JsonResponse({"success": False, "error": "Missing required parameters"})

            # TODO: integrate with your FraudDetector class here
            # For now, return dummy success response
            return JsonResponse({
                "success": True,
                "message": f"Model trained successfully with target={target}, "
                           f"models={model_types}, features={features}"
            })
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})

    return JsonResponse({"success": False, "error": "Invalid request"})

@login_required
def fraud_ml_based(request):
    """Machine learning-based fraud detection view"""
    return render(request, 'myapp/fraud_ml_based.html')

@login_required
def download_fraud_results(request):
    """Enhanced download handler with better error handling"""
    if 'fraud_results' not in request.session:
        return HttpResponse("No results available for download", status=404)
    
    try:
        results = pd.read_json(request.session['fraud_results'])
        threshold = request.session.get('fraud_threshold', 0.5)
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = (
            f'attachment; filename="fraud_results_threshold_{threshold}.csv"'
        )
        
        results.to_csv(response, index=False)
        return response
        
    except Exception as e:
        logger.error(f"Error in download_fraud_results: {str(e)}")
        return HttpResponse("Error generating download", status=500)

@login_required
def get_claim_details(request, claim_id):
    """Enhanced AJAX endpoint for claim details"""
    if 'fraud_results' not in request.session:
        return JsonResponse({'error': 'No results available'}, status=404)
    
    try:
        results = pd.read_json(request.session['fraud_results'])
        claim = results[results['claim_id'] == claim_id].iloc[0].to_dict()
        
        # Format triggered rules for better display
        if 'triggered_rules' in claim and claim['triggered_rules']:
            claim['triggered_rules'] = sorted(
                claim['triggered_rules'],
                key=lambda x: x['weighted_score'],
                reverse=True
            )
        
        return JsonResponse(claim)
        
    except Exception as e:
        logger.error(f"Error fetching claim {claim_id}: {str(e)}")
        return JsonResponse({'error': 'Claim not found'}, status=404)

@login_required
def get_dataset_columns(request):
    """Enhanced dataset column info with error handling"""
    dataset_name = request.GET.get('dataset')
    if not dataset_name:
        return JsonResponse({'error': 'Dataset name required'}, status=400)
    
    try:
        df = get_table_data(dataset_name)
        if df.empty:
            return JsonResponse({'error': 'Dataset is empty'}, status=404)
            
        return JsonResponse({
            'columns': list(df.columns),
            'sample_size': len(df)
        })
    except Exception as e:
        logger.error(f"Error getting columns for {dataset_name}: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

def detect_fraud(df):
    """Enhanced fraud detection pipeline with logging"""
    try:
        detector = FraudDetector()
        processed_data = detector.preprocess_data(df.copy())
        
        # Log preprocessing results
        logger.info(f"Preprocessed data shape: {processed_data.shape}")
        if not processed_data.empty:
            logger.debug(f"Sample preprocessed data:\n{processed_data.head(2)}")
        
        results = detector.calculate_rule_scores(processed_data)
        
        # Add risk categories with adjusted bins
        results['risk_category'] = pd.cut(
            results['rule_based_score'],
            bins=[0, 0.4, 0.7, 1],  # Adjusted bins for better distribution
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Add ensemble score (could combine with other models later)
        results['ensemble_score'] = results['rule_based_score']
        
        # Log final results
        logger.info(f"Fraud detection complete. Risk distribution:\n{results['risk_category'].value_counts()}")
        if not results.empty:
            logger.debug(f"Top scores:\n{results.nlargest(3, 'ensemble_score')[['claim_id', 'ensemble_score']]}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in detect_fraud: {str(e)}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['claim_id', 'risk_category', 'ensemble_score'])



#####
#####
#####

###Fraud ml models 
# views.py
# views.py
import logging
import traceback
import time
from datetime import datetime
import csv
import json
from io import BytesIO, StringIO
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from django.http import HttpResponse, JsonResponse
from django.template.loader import render_to_string
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.decorators.http import require_GET, require_POST
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.db import connection

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline

from xhtml2pdf import pisa
import pdfkit

logger = logging.getLogger(__name__)

# ==================== HELPER FUNCTIONS ====================

def load_table_to_dataframe(table_name, sample_size=None):
    """Load a DB table into a DataFrame with optional random sampling."""
    try:
        with connection.cursor() as cursor:
            # Get column names
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            columns_info = cursor.fetchall()
            columns = [col[1] for col in columns_info]

            # Get row count
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            row_count = cursor.fetchone()[0]

            # Load all data (no sampling)
            cursor.execute(f'SELECT * FROM "{table_name}"')
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=columns)
            
            # Store original row IDs
            df['original_row_id'] = df.index
            
            return df, True, row_count, False

    except Exception as e:
        logger.error(f"Error loading table {table_name}: {e}")
        return pd.DataFrame(), False, 0, False

def create_visualizations(df, target_col, features, results):
    """Create various visualizations for the results"""
    visualizations = {}
    
    try:
        # 1. Target variable distribution
        plt.figure(figsize=(10, 6))
        if df[target_col].dtype in ['int64', 'float64'] and df[target_col].nunique() > 10:
            plt.hist(df[target_col].dropna(), bins=30, alpha=0.7, color='skyblue')
            plt.title(f'Distribution of {target_col}')
            plt.xlabel(target_col)
            plt.ylabel('Frequency')
        else:
            value_counts = df[target_col].value_counts().head(10)
            plt.bar(value_counts.index.astype(str), value_counts.values, alpha=0.7, color='lightcoral')
            plt.title(f'Top 10 Values in {target_col}')
            plt.xlabel(target_col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        visualizations['target_dist'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 2. Correlation heatmap (for numeric features only)
        numeric_features = df[features].select_dtypes(include=np.number).columns.tolist()
        if len(numeric_features) > 1:
            plt.figure(figsize=(12, 8))
            corr_matrix = df[numeric_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            visualizations['correlation_heatmap'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
        
        # 3. Model performance comparison (if multiple models succeeded)
        successful_models = {}
        for model_type, result in results.items():
            if not result.get('error') and not result.get('is_anomaly_detection'):
                successful_models[model_type.upper()] = result['metrics']
        
        if successful_models:
            metrics_df = pd.DataFrame(successful_models).T
            plt.figure(figsize=(12, 6))
            metrics_df[['accuracy', 'precision', 'recall', 'f1']].plot(kind='bar', figsize=(12, 6))
            plt.title('Model Performance Comparison')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            visualizations['model_comparison'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
        
        # 4. PCA visualization for high-dimensional data
        if len(features) > 2:
            numeric_data = df[features].select_dtypes(include=np.number)
            if len(numeric_data.columns) > 1:
                # Handle missing values
                numeric_data = numeric_data.fillna(numeric_data.mean())
                
                # Standardize
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                
                # Apply PCA
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(scaled_data)
                
                plt.figure(figsize=(10, 8))
                plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=30)
                plt.title('PCA Visualization of Features')
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.tight_layout()
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                visualizations['pca_visualization'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
                
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
    
    return visualizations

def create_binary_target_from_continuous(df, continuous_col, threshold_quantile=0.95):
    """
    Create a binary target from a continuous column by setting a threshold
    based on quantile (values above threshold are considered anomalies/fraud)
    """
    threshold = df[continuous_col].quantile(threshold_quantile)
    binary_target = (df[continuous_col] > threshold).astype(int)
    fraud_ratio = binary_target.mean()
    
    logger.info(f"Created binary target from {continuous_col}. Threshold: {threshold:.2f}, Fraud ratio: {fraud_ratio:.4f}")
    
    return binary_target, fraud_ratio

def generate_feature_importance_plot(feature_importances, feature_names):
    """Generate a feature importance plot"""
    try:
        # Sort features by importance
        indices = np.argsort(feature_importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = [feature_importances[i] for i in indices]
        
        # Take top 20 features
        top_features = sorted_features[:20]
        top_importances = sorted_importances[:20]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_importances[::-1], align='center')
        plt.yticks(range(len(top_features)), top_features[::-1])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_data
    except Exception as e:
        logger.error(f"Error generating feature importance plot: {e}")
        return None

# ==================== MODEL MANAGER ====================

class FraudDetectionModelManager:
    def __init__(self):
        self.data = None
        self.cleaned_df = None
        self.fraud_ratio = None
        self.row_count = 0
        self.sampled = False
        self.target_info = {}
        self.visualizations = {}
        self.binary_target_created = False
        self.feature_importances = {}
        self.predictions = {}
        self.full_predictions = {}  # Store predictions for entire dataset
        self.original_data = None   # Store original data for reporting

    def import_data(self, table_name, sample_size=None):
        try:
            df, ok, row_count, sampled = load_table_to_dataframe(table_name, sample_size)
            if not ok:
                return False, 0, False

            self.row_count = row_count
            self.sampled = sampled
            self.original_data = df.copy()  # Store original data

            # Quick heuristic for fraud ratio from plausible columns
            fraud_columns = [c for c in df.columns if any(k in c.lower() for k in ['fraud', 'label', 'target', 'is_fraud', 'flag'])]
            if fraud_columns:
                fc = fraud_columns[0]
                try:
                    df[fc] = pd.to_numeric(df[fc], errors='coerce')
                    self.fraud_ratio = float(df[fc].mean(skipna=True))
                except Exception:
                    self.fraud_ratio = None
            else:
                self.fraud_ratio = None

            self.data = df
            return True, row_count, sampled
        except Exception as e:
            logger.error(f"import_data error: {e}\n{traceback.format_exc()}")
            return False, 0, False

    def prepare_data(self, features, target):
        try:
            if self.data is None:
                return False

            df = self.data.copy()

            needed = list(set(features + ([target] if target in df.columns else [])))
            df = df[[c for c in needed if c in df.columns]]

            # Handle missing values
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('Unknown')
                else:
                    df[col] = df[col].fillna(df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 0)

            # Check if we need to create a binary target from continuous variable
            if (target in df.columns and 
                pd.api.types.is_numeric_dtype(df[target]) and 
                df[target].nunique() > 20):
                
                # Create binary target for classification
                df['fraud_binary_target'], self.fraud_ratio = create_binary_target_from_continuous(df, target)
                self.binary_target_created = True
                target = 'fraud_binary_target'
                
                self.target_info = {
                    "name": target,
                    "dtype": str(df[target].dtype),
                    "unique_values": int(df[target].nunique()),
                    "is_binary": bool(df[target].nunique() == 2),
                    "class_counts": {str(k): int(v) for k, v in df[target].value_counts(dropna=False).to_dict().items()},
                }
            elif target in df.columns:
                self.target_info = {
                    "name": target,
                    "dtype": str(df[target].dtype),
                    "unique_values": int(df[target].nunique()),
                    "is_binary": bool(df[target].nunique() == 2),
                    "class_counts": {str(k): int(v) for k, v in df[target].value_counts(dropna=False).to_dict().items()},
                }
            else:
                self.target_info = {}

            self.cleaned_df = df
            return True, target  # Return modified target if binary was created
        except Exception as e:
            logger.error(f"prepare_data error: {e}\n{traceback.format_exc()}")
            return False, target

    def _get_model(self, mtype, params):
        if mtype == "rf":
            return RandomForestClassifier(
                n_estimators=min(100, int(params.get("rf_trees", 100))),  # Reduced for speed
                max_depth=10,
                n_jobs=-1,
                random_state=42,
            )
        if mtype == "lr":
            return LogisticRegression(
                solver="liblinear",  # Changed for better performance
                penalty="l2",
                max_iter=1000,
                random_state=42,
            )
        if mtype == "gb":
            return GradientBoostingClassifier(
                n_estimators=100,  # Reduced for speed
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )
        if mtype == "iforest":
            return IsolationForest(
                n_estimators=100,  # Reduced for memory
                contamination=float(params.get("contamination", 0.01)),
                random_state=42,
                n_jobs=-1,
            )
        raise ValueError(f"Unknown model type: {mtype}")

    def _get_preprocessor(self, numeric_features, categorical_features):
        """Create a preprocessor with dimensionality control"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Limit categorical features to avoid dimensionality explosion
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', max_categories=10))  # Limit categories
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ],
            remainder='drop'
        )
        
        return preprocessor

    def build_model(self, model_types, target, features, params):
        try:
            t0 = time.time()

            if self.data is None:
                return {"success": False, "error": "No dataset loaded"}

            # Prepare data and get potentially modified target
            success, modified_target = self.prepare_data(features, target)
            if not success:
                return {"success": False, "error": "Data preparation failed"}
            
            # Use modified target if binary was created
            if self.binary_target_created:
                target = modified_target

            X = self.cleaned_df[features]
            y = self.cleaned_df[target] if target in self.cleaned_df.columns else None

            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(include="object").columns.tolist()
            
            # Limit number of features if too many
            if len(numeric_features) + len(categorical_features) > 50:
                # Select top features by variance for numeric
                if len(numeric_features) > 20:
                    numeric_var = X[numeric_features].var()
                    numeric_features = numeric_var.nlargest(20).index.tolist()
                
                # Select top features by cardinality for categorical
                if len(categorical_features) > 30:
                    cat_cardinality = X[categorical_features].nunique()
                    categorical_features = cat_cardinality.nlargest(30).index.tolist()
            
            preprocessor = self._get_preprocessor(numeric_features, categorical_features)

            results = {}
            self.feature_importances = {}
            self.predictions = {}
            self.full_predictions = {}

            for mtype in model_types:
                try:
                    model = self._get_model(mtype, params)

                    if mtype == "iforest":
                        # For Isolation Forest, use only numeric features to avoid memory issues
                        isolation_features = numeric_features if numeric_features else features[:10]  # Use first 10 if no numeric
                        
                        if not isolation_features:
                            results[mtype] = {"error": "No suitable numeric features found for Isolation Forest"}
                            continue
                            
                        isolation_preprocessor = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())
                        ])
                        
                        pipe = Pipeline([
                            ("preprocessor", isolation_preprocessor),
                            ("model", model),
                        ])
                        
                        X_iso = X[isolation_features]
                        pipe.fit(X_iso)
                        scores = pipe.decision_function(X_iso)
                        predictions = pipe.predict(X_iso)
                        
                        # Store predictions for entire dataset with original row IDs
                        self.full_predictions[mtype] = {
                            'scores': scores.tolist(),
                            'anomalies': (scores < 0).astype(int).tolist(),
                            'row_ids': self.cleaned_df['original_row_id'].tolist()
                        }
                        
                        # For evaluation, use a subset if dataset is too large
                        if X_iso.shape[0] > 10000:
                            X_sample, _, y_sample, _ = train_test_split(
                                X_iso, scores, test_size=0.8, random_state=42
                            )
                            sample_scores = pipe.decision_function(X_sample)
                            results[mtype] = {
                                "is_anomaly_detection": True,
                                "metrics": {
                                    "anomaly_score_mean": float(np.mean(sample_scores)),
                                    "anomaly_score_std": float(np.std(sample_scores)),
                                    "n_anomalies": int((sample_scores < 0).sum()),
                                },
                            }
                        else:
                            results[mtype] = {
                                "is_anomaly_detection": True,
                                "metrics": {
                                    "anomaly_score_mean": float(np.mean(scores)),
                                    "anomaly_score_std": float(np.std(scores)),
                                    "n_anomalies": int((scores < 0).sum()),
                                },
                            }
                        continue

                    # supervised models
                    if y is None:
                        results[mtype] = {"error": "Target variable required for supervised learning"}
                        continue

                    # Encode y if it is not numeric/binary
                    if not pd.api.types.is_numeric_dtype(y):
                        try:
                            y_str = y.astype(str)
                            y_encoded, _ = pd.factorize(y_str)
                            y = pd.Series(y_encoded, index=y.index)
                        except Exception:
                            y = pd.to_numeric(y, errors="coerce")

                    # filter out rows with NaN y after coercion
                    valid_mask = ~pd.isna(y)
                    X_use = X.loc[valid_mask]
                    y_use = y.loc[valid_mask]

                    unique_classes, class_counts = np.unique(y_use, return_counts=True)
                    if len(unique_classes) < 2:
                        results[mtype] = {"error": f"Need at least 2 classes for classification, found: {unique_classes.tolist()}."}
                        continue
                    if min(class_counts) < 2:
                        results[mtype] = {"error": f"One class has only {int(min(class_counts))} sample(s). Need ≥2 per class."}
                        continue

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_use, y_use, test_size=0.2, random_state=42, stratify=y_use
                    )

                    # SMOTE if viable
                    unique_train, train_counts = np.unique(y_train, return_counts=True)
                    minority_n = int(min(train_counts))
                    if minority_n >= 2:
                        pipe = make_imb_pipeline(
                            preprocessor,
                            SMOTE(
                                sampling_strategy="minority",
                                random_state=42,
                                k_neighbors=max(1, min(5, minority_n - 1)),
                            ),
                            model,
                        )
                    else:
                        pipe = Pipeline([
                            ("preprocessor", preprocessor),
                            ("model", model),
                        ])

                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)
                    y_proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None

                    # Store test set predictions for evaluation
                    self.predictions[mtype] = {
                        'true': y_test.tolist(),
                        'predicted': y_pred.tolist(),
                        'probabilities': y_proba.tolist() if y_proba is not None else None
                    }

                    # Generate predictions for entire dataset with original row IDs
                    try:
                        if hasattr(pipe, "predict_proba"):
                            full_proba = pipe.predict_proba(X_use)
                            full_pred = pipe.predict(X_use)
                            self.full_predictions[mtype] = {
                                'true': y_use.tolist(),
                                'predicted': full_pred.tolist(),
                                'probabilities': full_proba.tolist(),
                                'features': features,
                                'row_ids': self.cleaned_df.loc[valid_mask, 'original_row_id'].tolist()
                            }
                        else:
                            full_pred = pipe.predict(X_use)
                            self.full_predictions[mtype] = {
                                'true': y_use.tolist(),
                                'predicted': full_pred.tolist(),
                                'probabilities': None,
                                'features': features,
                                'row_ids': self.cleaned_df.loc[valid_mask, 'original_row_id'].tolist()
                            }
                    except Exception as e:
                        logger.error(f"Error generating full predictions for {mtype}: {e}")
                        # Fallback to test predictions
                        self.full_predictions[mtype] = self.predictions[mtype]

                    metrics = {
                        "accuracy": float(accuracy_score(y_test, y_pred)),
                        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                    }

                    # ROC-AUC if binary and predict_proba exists
                    if hasattr(pipe, "predict_proba") and len(unique_classes) == 2:
                        y_proba = pipe.predict_proba(X_test)[:, 1]
                        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))

                    # Calculate feature importance for tree-based models
                    if mtype in ['rf', 'gb'] and hasattr(pipe.steps[-1][1], 'feature_importances_'):
                        try:
                            # Get feature names after preprocessing
                            feature_names = []
                            for name, trans, cols in preprocessor.transformers_:
                                if name == 'num':
                                    feature_names.extend(cols)
                                elif name == 'cat':
                                    # For categorical features, get the one-hot encoded names
                                    encoder = trans.named_steps['encoder']
                                    cat_features = []
                                    for i, col in enumerate(cols):
                                        if i < len(encoder.categories_):
                                            for cat in encoder.categories_[i]:
                                                cat_features.append(f"{col}_{cat}")
                                    feature_names.extend(cat_features)
                            
                            # Get importances
                            importances = pipe.steps[-1][1].feature_importances_
                            self.feature_importances[mtype] = {
                                'features': feature_names,
                                'importances': importances.tolist(),
                                'plot': generate_feature_importance_plot(importances, feature_names)
                            }
                        except Exception as e:
                            logger.error(f"Error calculating feature importance for {mtype}: {e}")

                    results[mtype] = {
                        "is_anomaly_detection": False,
                        "metrics": metrics,
                    }

                except Exception as me:
                    error_msg = str(me)
                    if "MemoryError" in error_msg or "allocate" in error_msg:
                        error_msg = "Memory error. Try with fewer features or a smaller dataset."
                    elif "only 1 member" in error_msg or "too few" in error_msg:
                        error_msg += ". Try a different target variable or use Isolation Forest."
                    elif "classes" in error_msg:
                        error_msg += ". Target variable needs at least 2 different values."
                    logger.error(f"Model {mtype} failed: {error_msg}\n{traceback.format_exc()}")
                    results[mtype] = {"error": error_msg}

            training_time = time.time() - t0
            
            # Create visualizations
            self.visualizations = create_visualizations(self.cleaned_df, target, features, results)

            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "target": target,
                "original_target": target if not self.binary_target_created else self.target_info.get('name', target),
                "features": features,
                "model_types": model_types,
                "fraud_ratio": self.fraud_ratio,
                "target_info": self.target_info,
                "results": results,
                "training_time": training_time,
                "sample_size": int(X.shape[0]),
                "visualizations": self.visualizations,
                "binary_target_created": self.binary_target_created,
                "feature_importances": self.feature_importances,
                "predictions": self.predictions,
                "full_predictions": self.full_predictions,
                "original_row_ids": self.cleaned_df['original_row_id'].tolist() if 'original_row_id' in self.cleaned_df.columns else list(range(len(self.cleaned_df))),
            }

        except Exception as e:
            logger.error(f"build_model failed: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": str(e)}

# ==================== VIEWS ====================

@login_required
def fraud_detection_dashboard(request):
    """Main dashboard"""
    try:
        sess = request.session.get("fraud_ml_results", {})
        context = {
            "page_title": "Machine Learning Fraud Detection",
            "data_tables": connection.introspection.table_names(),
            "results": sess if sess else {},
            "has_results": bool(sess.get("results")),
            "current_table": sess.get("current_table", ""),
            "training_time": sess.get("training_time"),
        }
        return render(request, "myapp/fraud_ml_based.html", context)
    except Exception as e:
        logger.error(f"Error in dashboard: {e}\n{traceback.format_exc()}")
        return render(request, "myapp/fraud_ml_based.html", {
            "page_title": "Error",
            "error_message": "Failed to load dashboard",
            "data_tables": connection.introspection.table_names(),
        })

@login_required
@require_GET
def fetch_table_columns(request):
    """AJAX helper to list columns & preview"""
    try:
        table_name = (request.GET.get("table") or "").strip()
        if not table_name:
            return JsonResponse({"success": False, "error": "No table specified"}, status=400)

        sample_size = None  # No sampling - use full dataset

        manager = FraudDetectionModelManager()
        ok, row_count, sampled = manager.import_data(table_name, sample_size)
        if not ok:
            return JsonResponse({"success": False, "error": "Failed to import data"}, status=500)

        preview = []
        if manager.data is not None and not manager.data.empty:
            preview = manager.data.head(5).replace({np.nan: None}).to_dict("records")

        # Store light metadata for continuity
        request.session["current_table"] = table_name
        request.session["fraud_ratio"] = manager.fraud_ratio
        request.session["sampled"] = sampled
        request.session.modified = True

        return JsonResponse({
            "success": True,
            "columns": list(manager.data.columns),
            "fraud_ratio": manager.fraud_ratio,
            "table": table_name,
            "row_count": row_count,
            "sampled": sampled,
            "cleaned_df_preview": preview,
            "sample_size": int(manager.data.shape[0]),
        })
    except Exception as e:
        logger.error(f"fetch_table_columns error: {e}\n{traceback.format_exc()}")
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@login_required
@require_POST
def train_fraud_models(request):
    """Train requested fraud detection models"""
    try:
        manager = FraudDetectionModelManager()

        # Collect & validate input
        features = request.POST.getlist("features")
        model_types = request.POST.getlist("model_types[]") or request.POST.getlist("model_types")
        current_table = request.POST.get("current_table")
        target = request.POST.get("target")
        sample_size = None  # No sampling - use full dataset

        errors = []
        if not current_table:
            errors.append("No dataset specified")
        if not target:
            errors.append("Target variable not specified")
        if not features:
            errors.append("No features selected")
        if not model_types:
            errors.append("No models selected")

        if errors:
            return JsonResponse({
                "success": False,
                "error": "Validation errors: " + ", ".join(errors)
            }, status=400)

        ok, row_count, sampled = manager.import_data(current_table, sample_size)
        if not ok:
            return JsonResponse({"success": False, "error": "Failed to load dataset"}, status=500)

        # Hyperparams
        try:
            rf_trees = int(request.POST.get("rf_trees", 100))  # Reduced default
            contamination = float(request.POST.get("contamination", 0.01))
        except ValueError:
            rf_trees, contamination = 100, 0.01

        # Train models
        result = manager.build_model(
            model_types=model_types,
            target=target,
            features=features,
            params={
                "rf_trees": rf_trees,
                "contamination": contamination,
            },
        )

        if not result.get("success"):
            return JsonResponse(result, status=400)

        # Annotate + persist to session
        result["current_table"] = current_table
        result["row_count"] = row_count
        result["sampled"] = sampled
        result["sample_size"] = result.get("sample_size") or int(manager.data.shape[0])
        result["feature_importances"] = manager.feature_importances
        result["predictions"] = manager.predictions
        result["full_predictions"] = manager.full_predictions
        result["original_row_ids"] = manager.original_data.index.tolist() if manager.original_data is not None else []

        request.session["fraud_ml_results"] = result
        request.session.modified = True

        return JsonResponse({
            "success": True,
            "redirect_url": reverse("display_fraud_results"),  
            "training_time": result.get("training_time"),
            "sampled": sampled,
            "row_count": row_count,
        })

    except Exception as e:
        logger.error(f"train_fraud_models critical error: {e}\n{traceback.format_exc()}")
        return JsonResponse({
            "success": False,
            "error": f"Critical server error: {e}"
        }, status=500)

@login_required
def display_fraud_results(request):
    """Dedicated results display endpoint"""
    try:
        results = request.session.get("fraud_ml_results")
        if not results or not results.get("success"):
            return redirect("fraud_ml_dashboard")

        context = {
            "page_title": "Fraud Detection Results",
            "data_tables": connection.introspection.table_names(),
            "results": results,
            "has_results": True,
            "current_table": results.get("current_table", ""),
            "training_time": results.get("training_time"),
        }
        return render(request, "myapp/fraud_ml_based.html", context)

    except Exception as e:
        logger.error(f"display_fraud_results error: {e}\n{traceback.format_exc()}")
        return render(request, "myapp/fraud_ml_based.html", {
            "page_title": "Error",
            "error_message": "Failed to display results",
            "data_tables": connection.introspection.table_names(),
        })

@require_GET
def session_keepalive(request):
    """Keep session alive during long-running operations"""
    if request.user.is_authenticated:
        request.session.modified = True
        return JsonResponse({"status": "ok", "session_key": request.session.session_key})
    return JsonResponse({"status": "error"}, status=401)

@login_required
def download_fraud_report(request, format_type):
    """Download fraud detection report in various formats"""
    try:
        results = request.session.get("fraud_ml_results")
        if not results or not results.get("success"):
            return JsonResponse({"success": False, "error": "No results available for download"})
        
        # Get the best performing model
        best_model = None
        best_score = -1
        for model_type, result in results.get("results", {}).items():
            if not result.get("error") and not result.get("is_anomaly_detection"):
                f1_score = result.get("metrics", {}).get("f1", 0)
                if f1_score > best_score:
                    best_score = f1_score
                    best_model = model_type
        
        # If no supervised model worked, try to use isolation forest
        if best_model is None:
            for model_type, result in results.get("results", {}).items():
                if not result.get("error") and result.get("is_anomaly_detection"):
                    best_model = model_type
                    break
        
        if best_model is None:
            return JsonResponse({"success": False, "error": "No valid models available for reporting"})
        
        # Prepare context for report
        context = {
            "results": results,
            "best_model": best_model,
            "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": request.user.get_username(),
        }
        
        if format_type == "pdf":
            # Generate PDF report using xhtml2pdf
            html_string = render_to_string('myapp/fraud_report_template.html', context)
            
            response = HttpResponse(content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="fraud_detection_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf"'
            
            # Create PDF
            pisa_status = pisa.CreatePDF(html_string, dest=response)
            if pisa_status.err:
                return HttpResponse('We had some errors <pre>' + html_string + '</pre>')
            return response
            
        elif format_type == "csv":
            # Generate CSV with predictions for the entire dataset
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="fraud_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
            
            writer = csv.writer(response)
            
            # Get the best model's predictions
            predictions = results.get("full_predictions", {}).get(best_model, {})
            
            if not predictions:
                return JsonResponse({"success": False, "error": "No prediction data available for download"})
            
            # Write header
            header = ["Original Row ID", "True Value", "Predicted Value", "Is Fraud"]
            
            # Add probability columns if available
            if predictions.get('probabilities') and len(predictions['probabilities']) > 0:
                if len(predictions['probabilities'][0]) == 2:  # Binary classification
                    header.extend(["Probability Class 0", "Probability Class 1"])
                else:
                    for i in range(len(predictions['probabilities'][0])):
                        header.append(f"Probability Class {i}")
            
            header.extend(["Model", "Confidence Score"])
            writer.writerow(header)
            
            # Write data
            for i in range(len(predictions.get('predicted', []))):
                row_id = predictions.get('row_ids', [])[i] if i < len(predictions.get('row_ids', [])) else i+1
                
                # Get true and predicted values
                true_val = predictions.get('true', [])[i] if i < len(predictions.get('true', [])) else "N/A"
                pred_val = predictions.get('predicted', [])[i] if i < len(predictions.get('predicted', [])) else "N/A"
                
                # Determine if it's fraud
                is_fraud = "Yes" if (pred_val == 1 or (isinstance(pred_val, (int, float)) and pred_val < 0)) else "No"
                
                row = [row_id, true_val, pred_val, is_fraud]
                
                # Add probabilities if available
                if predictions.get('probabilities') and i < len(predictions['probabilities']):
                    if isinstance(predictions['probabilities'][i], (list, np.ndarray)):
                        for prob in predictions['probabilities'][i]:
                            row.append(f"{prob:.6f}")
                    else:
                        # Add placeholders if probabilities aren't available
                        row.extend(["N/A", "N/A"])
                
                row.append(best_model.upper())
                
                # Calculate confidence score
                confidence = "N/A"
                if predictions.get('probabilities') and i < len(predictions['probabilities']):
                    if isinstance(predictions['probabilities'][i], (list, np.ndarray)):
                        max_prob = max(predictions['probabilities'][i])
                        confidence = f"{max_prob:.4f}"
                
                row.append(confidence)
                
                writer.writerow(row)
            
            return response
            
        elif format_type == "json":
            # Generate JSON report
            response = HttpResponse(content_type='application/json')
            response['Content-Disposition'] = f'attachment; filename="fraud_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"'
            
            # Create a comprehensive report for JSON export
            export_data = {
                "metadata": {
                    "generated_date": datetime.now().isoformat(),
                    "user": request.user.get_username(),
                    "dataset": results.get("current_table"),
                    "target_variable": results.get("target"),
                    "original_target": results.get("original_target"),
                    "sample_size": results.get("sample_size"),
                    "fraud_ratio": results.get("fraud_ratio"),
                    "training_time": results.get("training_time"),
                    "binary_target_created": results.get("binary_target_created", False),
                },
                "model_performance": {},
                "feature_importances": results.get("feature_importances", {}),
                "best_model": best_model,
            }
            
            # Add model performance
            for model_type, result in results.get("results", {}).items():
                if not result.get("error"):
                    export_data["model_performance"][model_type] = {
                        "metrics": result.get("metrics", {}),
                        "is_anomaly_detection": result.get("is_anomaly_detection", False),
                        "error": result.get("error")
                    }
            
            # Add summary statistics
            export_data["summary"] = {
                "total_records": results.get("sample_size", 0),
                "fraudulent_records": int(results.get("sample_size", 0) * (results.get("fraud_ratio", 0) or 0)),
                "non_fraudulent_records": int(results.get("sample_size", 0) * (1 - (results.get("fraud_ratio", 0) or 0))),
                "best_model_accuracy": export_data["model_performance"].get(best_model, {}).get("metrics", {}).get("accuracy", 0) if best_model else 0,
            }
            
            response.write(json.dumps(export_data, indent=2, default=str))
            return response
            
        else:
            return JsonResponse({"success": False, "error": "Unsupported format"})
            
    except Exception as e:
        logger.error(f"Error generating report: {e}\n{traceback.format_exc()}")
        return JsonResponse({"success": False, "error": str(e)})

@login_required
def forecasting(request):
    return render(request, 'forecasting.html')

@login_required
def ml_claims(request):
    return render(request, 'ml_claims.html')


###### Data profiling 

import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.db import connection
from django.contrib.auth.decorators import login_required
import json
import matplotlib.pyplot as plt
import io
import base64
from scipy import stats
from collections import Counter
import re
from wordcloud import WordCloud
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
from textblob import TextBlob

# Add the missing function
def describe_distribution(series):
    """Describe the distribution of a numeric series"""
    # Handle cases with no data
    if len(series.dropna()) == 0:
        return "No data"
    
    # Calculate moments
    skew = series.skew()
    kurt = series.kurtosis()
    
    # Describe based on skewness and kurtosis
    if abs(skew) < 0.5:
        shape = "approximately symmetric"
    elif skew > 0:
        shape = "right-skewed"
    else:
        shape = "left-skewed"
    
    if kurt > 3.5:
        tail = "heavy-tailed"
    elif kurt < 2.5:
        tail = "light-tailed"
    else:
        tail = "moderate-tailed"
    
    return f"{shape}, {tail}"

def calculate_entropy(series):
    """Calculate entropy of a categorical variable"""
    if len(series.dropna()) == 0:
        return 0
        
    value_counts = series.value_counts(normalize=True)
    return -sum(p * np.log2(p) for p in value_counts if p > 0)

def calculate_quality_score(df):
    """Calculate an overall data quality score (0-100)"""
    if len(df) == 0:
        return 0
        
    total_metrics = 0
    score = 0
    
    # Completeness score
    completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
    score += completeness * 25  # 25% of total score
    total_metrics += 1
    
    # Uniqueness score
    uniqueness = 1 - (df.duplicated().sum() / len(df))
    score += uniqueness * 20  # 20% of total score
    total_metrics += 1
    
    # Validity score (for numeric columns - check for outliers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        outlier_scores = []
        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Avoid division by zero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    outlier_scores.append(1 - (len(outliers) / len(df)))
        
        if outlier_scores:
            validity = sum(outlier_scores) / len(outlier_scores)
            score += validity * 25  # 25% of total score
            total_metrics += 1
    
    # Consistency score (check for consistent data types)
    type_consistency = 0
    for col in df.columns:
        # Check if all values in the column have the same type (excluding NaN)
        non_null_values = df[col].dropna()
        if len(non_null_values) > 0:
            first_type = type(non_null_values.iloc[0])
            consistent = all(isinstance(x, first_type) for x in non_null_values)
            if consistent:
                type_consistency += 1
    
    type_consistency = type_consistency / len(df.columns) if len(df.columns) > 0 else 0
    score += type_consistency * 30  # 30% of total score
    total_metrics += 1
    
    # Normalize score based on actual metrics calculated
    normalized_score = (score / total_metrics) * (4 / 4)  # 4 is the max possible if all metrics were calculated
    
    return round(normalized_score, 2)

@login_required(login_url='login')
def profiling_report_view(request):
    """View to generate profiling report with available dataset tables."""
    vendor = connection.vendor  # 'sqlite', 'postgresql', 'mysql', etc.

    with connection.cursor() as cursor:
        if vendor == "sqlite":
            cursor.execute("""
                SELECT name 
                FROM sqlite_master 
                WHERE type='table' 
                AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
        elif vendor == "postgresql":
            cursor.execute("""
                SELECT tablename 
                FROM pg_catalog.pg_tables 
                WHERE schemaname='public'
                ORDER BY tablename
            """)
        elif vendor == "mysql":
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = DATABASE()
                ORDER BY table_name
            """)
        else:
            raise NotImplementedError(f"Database vendor '{vendor}' is not supported yet.")

        dataset_ids = [row[0] for row in cursor.fetchall()]
    
    selected_id = request.GET.get('dataset_id', dataset_ids[0] if dataset_ids else None)
    compare_id = request.GET.get('compare_id')
    context = {
        'dataset_ids': dataset_ids,
        'selected_id': selected_id,
        'compare_id': compare_id,
    }
    
    # Handle PDF download request
    if 'download_pdf' in request.GET and selected_id:
        return generate_pdf_report(selected_id, compare_id)
    
    try:
        if selected_id and selected_id in dataset_ids:
            # Load ALL data without limit
            df = pd.read_sql(f'SELECT * FROM "{selected_id}"', connection)
            
            # Generate comprehensive statistics
            context.update(generate_comprehensive_stats(df, selected_id))
            
            # If comparison dataset is selected
            if compare_id and compare_id in dataset_ids and compare_id != selected_id:
                compare_df = pd.read_sql(f'SELECT * FROM "{compare_id}"', connection)
                context.update(generate_comparison_stats(df, compare_df, selected_id, compare_id))
            
            # Generate profile report (minimal for performance)
            if 'full_report' in request.GET:
                profile = ProfileReport(
                    df,
                    minimal=False,
                    explorative=True,
                    correlations={"auto": {"calculate": True}},
                    interactions={"continuous": False},
                    progress_bar=False
                )
                context['profile_html'] = profile.to_html()
            
        # Generate dataset comparison overview
        context['dataset_comparison'] = generate_dataset_comparison(dataset_ids)
            
    except Exception as e:
        context['error'] = f"Error processing {selected_id}: {str(e)}"
    
    return render(request, 'myapp/minet_profiling_report.html', context)

def generate_comprehensive_stats(df, dataset_name):
    """Generate comprehensive statistics and data for visualizations"""
    stats = {
        'dataset_name': dataset_name,
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': int(df.isnull().sum().sum()),
        'missing_pct': round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2) if len(df) > 0 else 0,
        'duplicate_rows': int(df.duplicated().sum()),
        'duplicate_pct': round((df.duplicated().sum() / len(df)) * 100, 2) if len(df) > 0 else 0,
    }
    
    # Get column types
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
    date_cols = len(df.select_dtypes(include=['datetime', 'datetime64']).columns)
    boolean_cols = len(df.select_dtypes(include='bool').columns)
    
    stats.update({
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'date_cols': date_cols,
        'boolean_cols': boolean_cols,
    })
    
    # Memory usage
    stats['memory_usage'] = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)  # MB
    
    # Column types breakdown
    stats['column_types'] = {
        'Numeric': numeric_cols,
        'Categorical': categorical_cols,
        'Date_Time': date_cols,
        'Boolean': boolean_cols,
        'Other': len(df.columns) - numeric_cols - categorical_cols - date_cols - boolean_cols
    }
    
    # Top columns with most missing values
    missing_values = df.isnull().sum().sort_values(ascending=False).head(10)
    stats['missing_values_top'] = {
        'columns': missing_values.index.tolist(),
        'counts': missing_values.values.tolist(),
        'percentages': [round((count / len(df)) * 100, 2) if len(df) > 0 else 0 for count in missing_values.values]
    }
    
    # Data sample for display
    stats['sample_data'] = df.head(10).to_dict('records')
    
    # Column-wise detailed analysis
    stats['column_analysis'] = []
    text_analytics_results = {}
    
    for col in df.columns:
        col_stats = generate_column_stats(df[col], col)
        stats['column_analysis'].append(col_stats)
        
        # Text analytics for categorical columns with text data
        if col_stats['type'] == 'categorical' and col_stats.get('is_text_data', False):
            text_analytics_results[col] = generate_text_analytics(df[col])
    
    stats['text_analytics'] = text_analytics_results
    
    # Correlation analysis
    if numeric_cols > 1:
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        stats['correlation_matrix'] = {
            'columns': corr_matrix.columns.tolist(),
            'data': corr_matrix.values.tolist()
        }
        
        # Top correlations
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        top_correlations = upper_triangle.stack().sort_values(ascending=False).head(10)
        
        stats['top_correlations'] = [
            {'pair': f"{pair[0]} - {pair[1]}", 'value': round(value, 3)}
            for pair, value in top_correlations.items() if not pd.isna(value)
        ]
    
    # Outlier detection for numeric columns
    stats['outlier_analysis'] = {}
    numeric_cols_list = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols_list:
        if len(df[col].dropna()) > 0:  # Only process if there's data
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:  # Avoid division by zero
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                
                stats['outlier_analysis'][col] = {
                    'count': len(outliers),
                    'percentage': round((len(outliers) / len(df)) * 100, 2) if len(df) > 0 else 0,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
    
    # Data quality score
    stats['quality_score'] = calculate_quality_score(df)
    
    # Generate chart data
    stats['chart_data'] = generate_chart_data(df, stats)
    
    return stats

def generate_column_stats(series, col_name):
    """Generate detailed statistics for a single column"""
    col_type = series.dtype
    stats = {
        'name': col_name,
        'type': 'numeric' if np.issubdtype(col_type, np.number) else 
                'categorical' if col_type == 'object' or col_type.name == 'category' else
                'datetime' if np.issubdtype(col_type, np.datetime64) else
                'boolean' if col_type == 'bool' else 'other',
        'missing_count': series.isnull().sum(),
        'missing_pct': round((series.isnull().sum() / len(series)) * 100, 2) if len(series) > 0 else 0,
        'unique_count': series.nunique(),
        'unique_pct': round((series.nunique() / len(series)) * 100, 2) if len(series) > 0 else 0,
    }
    
    # Type-specific statistics
    if np.issubdtype(col_type, np.number):
        stats.update({
            'min': series.min() if len(series.dropna()) > 0 else None,
            'max': series.max() if len(series.dropna()) > 0 else None,
            'mean': series.mean() if len(series.dropna()) > 0 else None,
            'median': series.median() if len(series.dropna()) > 0 else None,
            'std': series.std() if len(series.dropna()) > 0 else None,
            'q1': series.quantile(0.25) if len(series.dropna()) > 0 else None,
            'q3': series.quantile(0.75) if len(series.dropna()) > 0 else None,
            'skewness': round(series.skew(), 4) if len(series.dropna()) > 0 else None,
            'kurtosis': round(series.kurtosis(), 4) if len(series.dropna()) > 0 else None,
            'zeros_count': (series == 0).sum() if len(series.dropna()) > 0 else 0,
            'zeros_pct': round(((series == 0).sum() / len(series)) * 100, 2) if len(series) > 0 else 0,
            'negative_count': (series < 0).sum() if len(series.dropna()) > 0 and (series < 0).any() else 0,
            'negative_pct': round(((series < 0).sum() / len(series)) * 100, 2) if len(series) > 0 and (series < 0).any() else 0,
        })
        
        # Distribution characteristics
        stats['distribution'] = describe_distribution(series)
        
    elif col_type == 'object' or col_type.name == 'category':
        if len(series.dropna()) > 0:
            value_counts = series.value_counts()
            stats.update({
                'top_values': [
                    {'value': str(val), 'count': count, 'pct': round((count / len(series)) * 100, 2)}
                    for val, count in value_counts.head(5).items()
                ],
                'entropy': calculate_entropy(series),
                'is_mostly_unique': (series.nunique() / len(series)) > 0.9 if len(series) > 0 else False,
                'is_mostly_constant': (series.nunique() / len(series)) < 0.1 if len(series) > 0 else False,
            })
            
            # Check if this is text data (strings with average length > 3)
            try:
                if series.dropna().empty:
                    str_lengths = pd.Series([])
                else:
                    # Only calculate string lengths if the series contains strings
                    sample = series.dropna().iloc[0] if not series.dropna().empty else ""
                    if isinstance(sample, str):
                        str_lengths = series.str.len()
                        avg_length = str_lengths.mean()
                        stats.update({
                            'avg_length': avg_length,
                            'min_length': str_lengths.min(),
                            'max_length': str_lengths.max(),
                            'is_text_data': avg_length > 3  # Consider it text data if average length > 3
                        })
            except (AttributeError, TypeError):
                # If .str accessor fails, skip string length analysis
                stats['is_text_data'] = False
    
    elif np.issubdtype(col_type, np.datetime64):
        if len(series.dropna()) > 0:
            stats.update({
                'min_date': series.min(),
                'max_date': series.max(),
                'date_range_days': (series.max() - series.min()).days,
            })
    
    return stats

def generate_text_analytics(series):
    """Generate text analytics for text columns"""
    # Combine all text values
    text_data = ' '.join([str(x) for x in series.dropna() if isinstance(x, str)])
    
    if not text_data:
        return None
        
    # Word frequency analysis
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text_data.lower())
    word_freq = Counter(words)
    top_words = word_freq.most_common(20)
    
    # Sentiment analysis
    blob = TextBlob(text_data)
    sentiment = blob.sentiment
    
    # Generate word cloud image - with error handling
    wordcloud = None
    img_str = None
    
    try:
        if len(words) > 0:  # Only generate word cloud if we have words
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
            
            # Save word cloud to base64
            img_buffer = io.BytesIO()
            wordcloud.to_image().save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    except Exception as e:
        # If word cloud generation fails, just continue without it
        img_str = None
    
    return {
        'total_words': len(words),
        'unique_words': len(set(words)),
        'top_words': top_words,
        'sentiment': {
            'polarity': round(sentiment.polarity, 3),
            'subjectivity': round(sentiment.subjectivity, 3)
        },
        'wordcloud': img_str
    }

def generate_chart_data(df, stats):
    """Generate data for various charts"""
    chart_data = {}
    
    # Distribution charts for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:5]:  # Limit to first 5 numeric columns for performance
        if len(df[col].dropna()) > 0:
            # Create histogram data
            hist, bins = np.histogram(df[col].dropna(), bins=10)
            chart_data[f'{col}_histogram'] = {
                'labels': [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins)-1)],
                'data': hist.tolist()
            }
    
    # Bar charts for categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
        if len(df[col].dropna()) > 0:
            value_counts = df[col].value_counts().head(10)
            chart_data[f'{col}_barchart'] = {
                'labels': value_counts.index.tolist(),
                'data': value_counts.values.tolist()
            }
    
    return chart_data

def generate_comparison_stats(df1, df2, name1, name2):
    """Generate comparison statistics between two datasets"""
    comparison = {
        'comparison_datasets': f"{name1} vs {name2}",
        'row_count_diff': len(df1) - len(df2),
        'col_count_diff': len(df1.columns) - len(df2.columns),
        'common_columns': list(set(df1.columns) & set(df2.columns)),
        'unique_to_first': list(set(df1.columns) - set(df2.columns)),
        'unique_to_second': list(set(df2.columns) - set(df1.columns)),
    }
    
    # Compare schema
    comparison['schema_comparison'] = []
    all_columns = set(df1.columns) | set(df2.columns)
    for col in all_columns:
        col_comparison = {
            'column': col,
            'in_first': col in df1.columns,
            'in_second': col in df2.columns,
        }
        
        if col in df1.columns and col in df2.columns:
            col_comparison['type_match'] = str(df1[col].dtype) == str(df2[col].dtype)
            col_comparison['type_first'] = str(df1[col].dtype)
            col_comparison['type_second'] = str(df2[col].dtype)
            
            # For numeric columns, compare basic statistics
            if (np.issubdtype(df1[col].dtype, np.number) and 
                np.issubdtype(df2[col].dtype, np.number) and
                len(df1[col].dropna()) > 0 and len(df2[col].dropna()) > 0):
                col_comparison['mean_diff'] = df1[col].mean() - df2[col].mean()
                if df1[col].mean() != 0:
                    col_comparison['mean_diff_pct'] = round((col_comparison['mean_diff'] / df1[col].mean()) * 100, 2)
                else:
                    col_comparison['mean_diff_pct'] = float('inf')
        
        comparison['schema_comparison'].append(col_comparison)
    
    # Compare distributions for common numeric columns
    numeric_cols = (set(df1.select_dtypes(include=[np.number]).columns) & 
                    set(df2.select_dtypes(include=[np.number]).columns))
    comparison['distribution_comparison'] = []
    
    for col in numeric_cols:
        if len(df1[col].dropna()) > 0 and len(df2[col].dropna()) > 0:
            ks_stat, p_value = stats.ks_2samp(df1[col].dropna(), df2[col].dropna())
            comparison['distribution_comparison'].append({
                'column': col,
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'different_distribution': p_value < 0.05  # Significant difference at 5% level
            })
    
    return {'comparison': comparison}

def generate_dataset_comparison(dataset_ids):
    """Generate overview comparison of all datasets in the database"""
    comparison = []
    
    for dataset_id in dataset_ids:
        with connection.cursor() as cursor:
            cursor.execute(f'SELECT COUNT(*) FROM "{dataset_id}"')
            row_count = cursor.fetchone()[0]
            
            cursor.execute(f'PRAGMA table_info("{dataset_id}")')
            columns = [row[1] for row in cursor.fetchall()]
            
            comparison.append({
                'name': dataset_id,
                'row_count': row_count,
                'column_count': len(columns),
                'columns': columns
            })
    
    return comparison

def generate_pdf_report(dataset_name, compare_name=None):
    """Generate a PDF report of the data profiling"""
    # Create a file-like buffer to receive PDF data
    buffer = io.BytesIO()
    
    # Create the PDF object, using the buffer as its "file"
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    
    # Container for the 'Flowable' objects
    elements = []
    styles = getSampleStyleSheet()
    
    # Add title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center aligned
    )
    
    elements.append(Paragraph(f"Data Profiling Report: {dataset_name}", title_style))
    
    # Add dataset overview
    elements.append(Paragraph("Dataset Overview", styles['Heading2']))
    
    # Load data and generate stats
    df = pd.read_sql(f'SELECT * FROM "{dataset_name}"', connection)
    stats = generate_comprehensive_stats(df, dataset_name)
    
    # Create overview table
    overview_data = [
        ['Metric', 'Value'],
        ['Total Rows', stats['rows']],
        ['Total Columns', stats['columns']],
        ['Missing Values', stats['missing_values']],
        ['Missing Percentage', f"{stats['missing_pct']}%"],
        ['Duplicate Rows', stats['duplicate_rows']],
        ['Data Quality Score', f"{stats['quality_score']}/100"]
    ]
    
    overview_table = Table(overview_data)
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(overview_table)
    elements.append(Spacer(1, 20))
    
    # Add column analysis
    elements.append(Paragraph("Column Analysis", styles['Heading2']))
    
    for col_stats in stats['column_analysis'][:10]:  # Limit to first 10 columns
        elements.append(Paragraph(f"Column: {col_stats['name']} ({col_stats['type']})", styles['Heading3']))
        
        col_data = [
            ['Metric', 'Value'],
            ['Missing Values', f"{col_stats['missing_count']} ({col_stats['missing_pct']}%)"],
            ['Unique Values', f"{col_stats['unique_count']} ({col_stats['unique_pct']}%)"]
        ]
        
        if col_stats['type'] == 'numeric':
            col_data.extend([
                ['Minimum', col_stats['min']],
                ['Maximum', col_stats['max']],
                ['Average', col_stats['mean']],
                ['Standard Deviation', col_stats['std']]
            ])
        
        col_table = Table(col_data)
        col_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(col_table)
        elements.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(elements)
    
    # FileResponse sets the Content-Disposition header so that browsers
    # present the option to save the file.
    buffer.seek(0)
    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{dataset_name}_profiling_report.pdf"'
    
    return response


