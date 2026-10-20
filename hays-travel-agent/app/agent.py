# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hays Travel Personal Assistant Multi-Agent System on Google Cloud ADK.

Architecture:
- Model: `gemini-4-argon` (via Vertex AI global endpoint)
- Security: Pure Workload/Workforce Identity Federation (WIF) + OIDC token exchange
  (`sts.googleapis.com/v1/token`). No custom Service Account impersonation in the
  agent query path; BigQuery natively enforces 4-tier Row-Level Security (RLS)
  (`rls_4_tier_wif_policy` / `v_bookings`) using `SESSION_USER()`.
- Sub-Agents:
  1. `databricks_ops_agent` (Pillar 1): Digital Productivity & Earnings Assistant
     (Monday morning weekly planner, 1WD Agent Report with 7th-of-month pay-period MTD
     reconciliation, T+7 post-holiday Welcome Home calls with Mosaic enrichment, and
     balance-due tracker)
  2. `databricks_analyst_agent` (Pillar 2): Customer Personalisation & Tailored Offer Curation
     (Customer 360 with probabilistic match >= 425, Experian Mosaic segmentation,
     tailored offer curation + high-margin Hays Vista, Cruise, and Long-Haul wildcards,
     and pre-departure Forex / Airport Parking / Insurance ancillaries)
  3. `databricks_research_agent` (Pillar 3): Product Positioning & "Be the Influencer" Studio
     (regional All-Inclusive top-seller queries, Google Places API + BigQuery 292k
     hotel/resort catalog standardization, regional tone-of-voice adaptation for
     North East / Scotland / Manchester, and CMA/ABTA-compliant marketing copy)
"""

import glob
import json
import os
import socket
import subprocess
import time
from typing import Any
import urllib.request
import urllib3.util.connection

# Ensure local Cloudtop sessions connect directly to https://bigquery.googleapis.com
# over IPv4 rather than stalling on IPv6 blackholes or bigquery.mtls.googleapis.com.
os.environ.setdefault("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")
os.environ.setdefault("GOOGLE_API_USE_MTLS_ENDPOINT", "never")
if os.getenv("K_SERVICE") is None:
    urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.auth import AuthConfig, OpenIdConnectWithConfig
from google.adk.models import Gemini
from google.adk.tools import ToolContext
from google.api_core import exceptions as google_exceptions
import google.auth
from google.auth import default as google_auth_default
from google.auth import identity_pool
from google.auth.transport import requests as google_requests
from google.cloud import bigquery
from google.genai import types
import google.genai._api_client
from google.oauth2 import id_token
from google.oauth2.credentials import Credentials
import requests.adapters

MODEL = "gemini-4-argon"
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "ewans-demo-project")
BQ_DATASET = os.getenv("HAYS_BQ_DATASET", "hays_travel_sharing")
BQ_LOCATION = os.getenv("HAYS_BQ_LOCATION", "europe-west2")


def _build_plain_auth_request() -> google_requests.Request:
    """Returns a non-mTLS google-auth Request to avoid expired gnubby mTLS stalls on Cloudtop."""
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter())
    return google_requests.Request(session)


def _load_local_dev_credentials() -> Credentials | None:
    """Loads refreshable non-CBA OAuth2 credentials for local development."""
    adc_candidates = [os.getenv("HAYS_LOCAL_ADC_PATH", "")] + sorted(
        glob.glob(
            os.path.expanduser("~/.config/gcloud/legacy_credentials/*/adc.json")
        )
    )
    for adc_path in adc_candidates:
        if adc_path and os.path.isfile(adc_path):
            try:
                creds = Credentials.from_authorized_user_file(
                    adc_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                creds.refresh(_build_plain_auth_request())
                return creds
            except Exception:
                pass
    try:
        token = (
            subprocess.check_output(
                ["gcloud", "auth", "print-access-token"],
                env={
                    **os.environ,
                    "CLOUDSDK_METRICS_ENVIRONMENT": "datacloud.jetski",
                },
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        if token:
            return Credentials(token=token)
    except Exception:
        pass
    return None


def _gcloud_fallback_default(
    scopes: Any = None,
    request: Any = None,
    quota_project_id: Any = None,
    default_scopes: Any = None,
) -> tuple[Any, str | None]:
    """Ensures local Cloudtop runs use active gcloud credentials when ADC differs."""
    if os.getenv("K_SERVICE") is None:
        local_creds = _load_local_dev_credentials()
        if local_creds is not None:
            return local_creds, PROJECT_ID
    return google_auth_default(
        scopes=scopes,
        request=request,
        quota_project_id=quota_project_id,
        default_scopes=default_scopes,
    )


google.auth.default = _gcloud_fallback_default
google.genai._api_client.google.auth.default = _gcloud_fallback_default

# Customer Microsoft Entra ID OIDC + GCP Workload/Workforce Identity Federation (WIF) Config
ENTRA_TENANT_ID = os.getenv("HAYS_ENTRA_TENANT_ID", "common")
ENTRA_CLIENT_ID = os.getenv("HAYS_ENTRA_CLIENT_ID", "api://hays-gemini-enterprise")
WIF_AUDIENCE = os.getenv(
    "HAYS_WIF_AUDIENCE",
    "//iam.googleapis.com/locations/global/workforcePools/hays-oidc-pool/providers/entra-oidc",
)

OIDC_AUTH_CONFIG = AuthConfig(
    auth_scheme=OpenIdConnectWithConfig(
        openIdConnectUrl=(
            f"https://login.microsoftonline.com/{ENTRA_TENANT_ID}/v2.0/"
            ".well-known/openid-configuration"
        ),
        authorization_endpoint=(
            f"https://login.microsoftonline.com/{ENTRA_TENANT_ID}/oauth2/v2.0/authorize"
        ),
        token_endpoint=(
            f"https://login.microsoftonline.com/{ENTRA_TENANT_ID}/oauth2/v2.0/token"
        ),
    )
)


class OidcJwtSupplier(identity_pool.SubjectTokenSupplier):
    """Supplies the caller's verified OIDC ID Token to GCP Security Token Service (STS)."""

    def __init__(self, oidc_jwt: str) -> None:
        self._jwt = oidc_jwt

    def get_subject_token(self, context: Any, request: Any) -> str:
        return self._jwt


def _extract_oidc_jwt(tool_context: ToolContext | None) -> str | None:
    """Extracts and optionally verifies the caller's OIDC ID token from ADK AuthConfig or state."""
    if tool_context is None:
        return None
    raw_jwt = tool_context.state.get("auth:oidc_id_token")
    if not raw_jwt:
        try:
            auth_resp = tool_context.get_auth_response(OIDC_AUTH_CONFIG)
            if auth_resp and getattr(auth_resp, "oauth2", None):
                raw_jwt = auth_resp.oauth2.id_token
        except Exception:
            raw_jwt = None

    if raw_jwt and os.getenv("HAYS_VERIFY_OIDC_SIGNATURE", "false").lower() == "true":
        certs_url = (
            f"https://login.microsoftonline.com/{ENTRA_TENANT_ID}/discovery/v2.0/keys"
        )
        id_token.verify_token(
            raw_jwt,
            google_requests.Request(),
            audience=ENTRA_CLIENT_ID,
            certs_url=certs_url,
        )
    return raw_jwt


def _build_bq_client(creds: Any) -> bigquery.Client:
    """Builds a BigQuery client using standard HTTPS (bypassing local Cloudtop mTLS stalls)."""
    http = google_requests.AuthorizedSession(
        creds, auth_request=_build_plain_auth_request()
    )
    http.mount("https://", requests.adapters.HTTPAdapter())
    http._is_mtls = False
    return bigquery.Client(
        project=PROJECT_ID,
        location=BQ_LOCATION,
        credentials=creds,
        _http=http,
    )


_LOCAL_DEV_BQ_CLIENT: bigquery.Client | None = None


def get_caller_bq_client(tool_context: ToolContext | None = None) -> bigquery.Client:
    """Returns a BigQuery client authenticated via WIF OIDC token exchange.

    In production, exchanges the caller's Entra ID OIDC JWT at `sts.googleapis.com/v1/token`
    so BigQuery enforces native Row Access Policies (`rls_4_tier_wif_policy`) and `v_bookings`
    against the caller's federated `SESSION_USER()` identity.
    In local sandbox development (when no OIDC JWT is present), falls back to local gcloud/ADC.
    """
    global _LOCAL_DEV_BQ_CLIENT
    oidc_jwt = _extract_oidc_jwt(tool_context)
    if oidc_jwt:
        wif_creds = identity_pool.Credentials(
            audience=WIF_AUDIENCE,
            subject_token_type="urn:ietf:params:oauth:token-type:jwt",
            token_url="https://sts.googleapis.com/v1/token",
            subject_token_supplier=OidcJwtSupplier(oidc_jwt),
            scopes=["https://www.googleapis.com/auth/bigquery"],
        )
        return _build_bq_client(wif_creds)

    if _LOCAL_DEV_BQ_CLIENT is not None:
        return _LOCAL_DEV_BQ_CLIENT

    local_creds = _load_local_dev_credentials()
    if local_creds is not None:
        _LOCAL_DEV_BQ_CLIENT = _build_bq_client(local_creds)
        return _LOCAL_DEV_BQ_CLIENT

    creds, _ = google_auth_default(
        scopes=["https://www.googleapis.com/auth/bigquery"]
    )
    _LOCAL_DEV_BQ_CLIENT = _build_bq_client(creds)
    return _LOCAL_DEV_BQ_CLIENT


def _serialize_rows(rows: Any) -> list[dict[str, Any]]:
    """Converts BigQuery RowIterator results into JSON-serializable dictionaries."""
    out: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for k, v in dict(row).items():
            if hasattr(v, "isoformat"):
                item[k] = v.isoformat()
            elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                item[k] = v
            else:
                item[k] = str(v)
        out.append(item)
    return out


def _execute_bq_query(
    client: bigquery.Client,
    sql: str,
    job_config: bigquery.QueryJobConfig,
    max_attempts: int = 5,
) -> list[dict[str, Any]]:
    """Executes a BigQuery query with automatic retry on transient backend overload or PENDING stalls."""
    if job_config.job_timeout_ms is None:
        job_config.job_timeout_ms = 20000
    for attempt in range(max_attempts):
        job = None
        try:
            job = client.query(sql, job_config=job_config, timeout=15)
            return _serialize_rows(job.result(timeout=18))
        except Exception as exc:
            if job is not None:
                try:
                    job.cancel()
                except Exception:
                    pass
            msg = str(exc)
            is_retryable = isinstance(
                exc, (TimeoutError, requests.exceptions.RequestException)
            ) or any(
                k in msg
                for k in (
                    "jobBackendError",
                    "backendError",
                    "Retrying may solve the problem",
                    "Retrying the job may solve the problem",
                    "Operation did not complete within the designated timeout",
                    "timed out",
                    "Timeout",
                )
            )
            if attempt < max_attempts - 1 and is_retryable:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise
    return []


# ==============================================================================
# PILLAR 1: Digital Productivity & Earnings Assistant (`databricks_ops_agent`)
# ==============================================================================


def get_morning_sales_briefing(
    booking_date: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Generates the 1WD Agent Report morning sales briefing and pay-period MTD summary.

    Replicates the 1WD Agent Report metrics (bookings count, total passengers,
    gross transaction value, profit, commission earned toward the 7th-of-month
    payday cut-off, and outstanding balance) for both `booking_date` and the active
    monthly commission pay period (8th of prior/current month -> 7th cut-off),
    scoped by BigQuery WIF/OIDC RLS.

    Args:
        booking_date: The booking date in YYYY-MM-DD format (e.g., '2026-10-17').

    Returns:
        Dictionary containing the single-day 1WD summary, pay-period MTD summary,
        and top individual bookings.
    """
    client = get_caller_bq_client(tool_context)
    sql = f"""
        SELECT
            BookingReference,
            SaleBranchID,
            SellingBranch.BranchName AS BranchName,
            Username,
            AgentName,
            LeadPassenger,
            DestinationType.Destination AS Destination,
            DestinationType.DestinationType AS PackageType,
            DepartureDate,
            Duration,
            Passengers,
            GrossPrice,
            CAST(Profit AS FLOAT64) AS Profit,
            CAST(Commission AS FLOAT64) AS Commission,
            Outstanding,
            BalanceDue
        FROM `{PROJECT_ID}.{BQ_DATASET}.v_bookings`
        WHERE BookingDate = @booking_date
          AND Cancelled = 'N'
        ORDER BY GrossPrice DESC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("booking_date", "DATE", booking_date)
        ],
        labels={"datacloud": "jetski", "agent_pillar": "ops"},
    )
    rows = _execute_bq_query(client, sql, job_config)
    total_gross = sum(float(r.get("GrossPrice") or 0.0) for r in rows)
    total_profit = sum(float(r.get("Profit") or 0.0) for r in rows)
    total_commission = sum(float(r.get("Commission") or 0.0) for r in rows)
    total_pax = sum(int(r.get("Passengers") or 0) for r in rows)

    # Calculate 1WD Agent Report monthly commission pay-period (8th -> 7th cut-off)
    mtd_sql = f"""
        WITH bounds AS (
            SELECT
                CASE
                    WHEN EXTRACT(DAY FROM @booking_date) <= 7
                        THEN DATE_ADD(DATE_TRUNC(DATE_SUB(@booking_date, INTERVAL 1 MONTH), MONTH), INTERVAL 7 DAY)
                    ELSE DATE_ADD(DATE_TRUNC(@booking_date, MONTH), INTERVAL 7 DAY)
                END AS period_start,
                CASE
                    WHEN EXTRACT(DAY FROM @booking_date) <= 7
                        THEN DATE_ADD(DATE_TRUNC(@booking_date, MONTH), INTERVAL 6 DAY)
                    ELSE DATE_ADD(DATE_TRUNC(DATE_ADD(@booking_date, INTERVAL 1 MONTH), MONTH), INTERVAL 6 DAY)
                END AS period_cutoff_date
        )
        SELECT
            CAST(ANY_VALUE(bounds.period_start) AS STRING) AS pay_period_start,
            CAST(ANY_VALUE(bounds.period_cutoff_date) AS STRING) AS payday_cutoff_date,
            COUNT(*) AS mtd_bookings,
            COALESCE(SUM(b.Passengers), 0) AS mtd_passengers,
            ROUND(COALESCE(SUM(b.GrossPrice), 0.0), 2) AS mtd_gross_sales_gbp,
            ROUND(COALESCE(SUM(CAST(b.Profit AS FLOAT64)), 0.0), 2) AS mtd_profit_gbp,
            ROUND(COALESCE(SUM(CAST(b.Commission AS FLOAT64)), 0.0), 2) AS mtd_commission_gbp
        FROM `{PROJECT_ID}.{BQ_DATASET}.v_bookings` b
        CROSS JOIN bounds
        WHERE b.Cancelled = 'N'
          AND b.BookingDate BETWEEN bounds.period_start AND @booking_date
    """
    mtd_rows = _execute_bq_query(client, mtd_sql, job_config)
    mtd_summary = mtd_rows[0] if mtd_rows else {}

    return {
        "status": "success",
        "booking_date": booking_date,
        "summary_1wd": {
            "total_bookings": len(rows),
            "total_passengers": total_pax,
            "total_gross_sales_gbp": round(total_gross, 2),
            "total_profit_gbp": round(total_profit, 2),
            "total_commission_gbp": round(total_commission, 2),
        },
        "pay_period_mtd_summary": mtd_summary,
        "top_bookings": rows[:25],
    }


def get_welcome_home_call_list(
    return_date: str, tool_context: ToolContext, lookback_days: int = 7
) -> dict[str, Any]:
    """Retrieves the T+7 post-holiday Welcome Home call list for customers returning recently.

    Joins RLS-scoped bookings with `customermaster` (`confidence_score >= 425`) and
    `customermosaicdemographics` (`MosaicGroup`, `MosaicType`) across a rolling T+7
    return window (`return_date - lookback_days` through `return_date`) so consultants
    can prioritize high-value welcome-back calls (e.g., 'Mrs Smith returned from
    Tenerife last week, give her a welcome-home call').

    Args:
        return_date: The reference return date in YYYY-MM-DD format (e.g., '2026-10-16').
        lookback_days: Number of days to look back from return_date (default 7 for T+7).

    Returns:
        Dictionary with returning customers, trip details, and Experian Mosaic segments.
    """
    client = get_caller_bq_client(tool_context)
    sql = f"""
        WITH recent_returns AS (
            SELECT
                b.BookingReference,
                b.SaleBranchID,
                b.SellingBranch.BranchName AS BranchName,
                b.Username,
                b.LeadPassenger,
                b.DebtorID,
                b.ReturnDate,
                DATE_DIFF(@return_date, b.ReturnDate, DAY) AS DaysSinceReturn,
                b.DestinationType.Destination AS Destination,
                b.DestinationType.DestinationType AS PackageType,
                b.GrossPrice,
                b.HotelDetails[SAFE_OFFSET(0)].HotelName AS HotelName,
                b.HotelDetails[SAFE_OFFSET(0)].Resort AS Resort,
                b.HotelDetails[SAFE_OFFSET(0)].StarRating AS StarRating
            FROM `{PROJECT_ID}.{BQ_DATASET}.v_bookings` b
            WHERE b.ReturnDate BETWEEN DATE_SUB(@return_date, INTERVAL @lookback_days DAY) AND @return_date
              AND b.Cancelled = 'N'
            ORDER BY b.ReturnDate DESC, b.GrossPrice DESC
            LIMIT 50
        )
        SELECT
            r.BookingReference,
            r.SaleBranchID,
            r.BranchName,
            r.Username,
            r.LeadPassenger,
            r.ReturnDate,
            r.DaysSinceReturn,
            r.Destination,
            r.PackageType,
            r.GrossPrice,
            r.HotelName,
            r.Resort,
            r.StarRating,
            c.CustomerID,
            c.confidence_score,
            c.TextMessaging,
            c.MailingAllowed,
            m.MosaicGroup,
            m.MosaicType,
            m.EstimatedMosaicGroup
        FROM recent_returns r
        LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.customermaster` c
          ON r.DebtorID = c.CustomerID AND c.confidence_score >= 425
        LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.customermosaicdemographics` m
          ON c.CustomerID = m.CustomerID AND m.IsCurrent = TRUE
        ORDER BY r.ReturnDate DESC, r.GrossPrice DESC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("return_date", "DATE", return_date),
            bigquery.ScalarQueryParameter(
                "lookback_days", "INT64", int(lookback_days)
            ),
        ],
        labels={"datacloud": "jetski", "agent_pillar": "ops"},
    )
    rows = _execute_bq_query(client, sql, job_config)
    return {
        "status": "success",
        "return_date": return_date,
        "lookback_days": lookback_days,
        "welcome_home_calls": rows,
    }


def get_balance_due_tracker(
    due_before_date: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Lists active bookings with outstanding balances due on or before due_before_date.

    Args:
        due_before_date: Upper bound BalanceDue date in YYYY-MM-DD format (e.g., '2026-11-01').

    Returns:
        Dictionary of bookings requiring balance collection follow-up.
    """
    client = get_caller_bq_client(tool_context)
    sql = f"""
        SELECT
            BookingReference,
            SaleBranchID,
            Username,
            LeadPassenger,
            DestinationType.Destination AS Destination,
            DepartureDate,
            BalanceDue,
            GrossPrice,
            TotalPayment,
            Outstanding,
            IsActiveDirectDebit
        FROM `{PROJECT_ID}.{BQ_DATASET}.v_bookings`
        WHERE Cancelled = 'N'
          AND Outstanding > 0
          AND BalanceDue >= CURRENT_DATE()
          AND BalanceDue <= @due_before_date
        ORDER BY BalanceDue ASC
        LIMIT 50
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "due_before_date", "DATE", due_before_date
            )
        ],
        labels={"datacloud": "jetski", "agent_pillar": "ops"},
    )
    rows = _execute_bq_query(client, sql, job_config)
    return {
        "status": "success",
        "due_before_date": due_before_date,
        "outstanding_bookings": rows,
    }


def get_monday_morning_weekly_plan(
    reference_date: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Generates a unified Monday morning weekly action plan for a branch consultant or PTC.

    Supports the 'Day in the Life of a Branch Agent' workflow by combining:
    1. 1WD Agent Report daily sales & monthly pay-period (7th cut-off) earnings summary,
    2. High-value T+7 Welcome Home calls (customers returning in the past 7 days, enriched
       with `customermaster` confidence_score >= 425 and Experian `MosaicGroup`),
    3. Urgent balance-due collection reminders for the next 14 days (highlighting bookings
       without an active Direct Debit).

    Args:
        reference_date: Reference date in YYYY-MM-DD format (e.g., '2026-10-17').

    Returns:
        Dictionary containing the consultant's prioritized weekly plan across earnings,
        T+7 welcome-home calls, and balance reminders.
    """
    briefing = get_morning_sales_briefing(reference_date, tool_context)
    welcome_calls = get_welcome_home_call_list(
        reference_date, tool_context, lookback_days=7
    )
    client = get_caller_bq_client(tool_context)
    bal_sql = f"""
        SELECT
            BookingReference,
            SaleBranchID,
            SellingBranch.BranchName AS BranchName,
            Username,
            LeadPassenger,
            DestinationType.Destination AS Destination,
            DepartureDate,
            BalanceDue,
            GrossPrice,
            Outstanding,
            IsActiveDirectDebit
        FROM `{PROJECT_ID}.{BQ_DATASET}.v_bookings`
        WHERE Cancelled = 'N'
          AND Outstanding > 0
          AND BalanceDue BETWEEN @reference_date AND DATE_ADD(@reference_date, INTERVAL 14 DAY)
        ORDER BY (IsActiveDirectDebit = 'N') DESC, BalanceDue ASC, Outstanding DESC
        LIMIT 25
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "reference_date", "DATE", reference_date
            )
        ],
        labels={"datacloud": "jetski", "agent_pillar": "ops"},
    )
    urgent_balances = _execute_bq_query(client, bal_sql, job_config)
    return {
        "status": "success",
        "reference_date": reference_date,
        "earnings_and_1wd_summary": briefing.get("summary_1wd"),
        "pay_period_mtd_summary": briefing.get("pay_period_mtd_summary"),
        "priority_1_t_plus_7_welcome_home_calls": welcome_calls.get(
            "welcome_home_calls", []
        )[:15],
        "priority_2_urgent_balance_reminders": urgent_balances,
    }


# ==============================================================================
# PILLAR 2: Customer Personalisation & Tailored Offer Curation (`databricks_analyst_agent`)
# ==============================================================================


def get_customer_360_and_ancillary_gaps(
    branch_id: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Surfaces high-propensity Customer 360 profiles, Hays Vista margin opportunities, and ancillary gaps.

    Enforces `confidence_score >= 425` on `customermaster`, joins Experian Mosaic
    demographics (`customermosaicdemographics`), `supplier` (highlighting in-house
    Hays Vista vs. 3rd-party tour operators), and `branch` (`HasForex`) to identify:
    1. Customers missing Foreign Exchange (`AllFXOrders` empty where branch `HasForex = 'Yes'`),
    2. Upcoming departures missing Airport Parking (`CarParkDetails`) or Insurance (`InsuranceDetails`),
    3. Tailored high-margin Hays Vista (`VIS0001`) upgrade / next-holiday recommendations.

    Args:
        branch_id: Branch identifier to filter on (e.g., '0015', '9024', or 'ALL').

    Returns:
        Dictionary of enriched Customer 360 profiles, Mosaic segments, and cross-sell actions.
    """
    client = get_caller_bq_client(tool_context)
    sql = f"""
        WITH upcoming_bookings AS (
            SELECT
                b.BookingReference,
                b.SaleBranchID,
                b.LeadPassenger,
                b.DebtorID,
                b.AE_SupplierKey,
                b.DepartureDate,
                b.Duration,
                b.DestinationType.Destination AS Destination,
                b.DestinationType.ForexStocked AS RecommendedCurrency,
                b.GrossPrice,
                CAST(b.Profit AS FLOAT64) AS Profit,
                CAST(b.Commission AS FLOAT64) AS Commission,
                ARRAY_LENGTH(b.CarParkDetails) AS CarParkCount,
                ARRAY_LENGTH(b.InsuranceDetails) AS InsuranceCount
            FROM `{PROJECT_ID}.{BQ_DATASET}.v_bookings` b
            WHERE b.Cancelled = 'N'
              AND b.DepartureDate >= CURRENT_DATE()
              AND (@branch_id = 'ALL' OR b.SaleBranchID = @branch_id)
            ORDER BY b.DepartureDate ASC
            LIMIT 100
        )
        SELECT
            ub.BookingReference,
            ub.SaleBranchID,
            br.BranchName,
            br.HasForex AS BranchHasForex,
            ub.LeadPassenger,
            ub.DepartureDate,
            ub.Duration,
            ub.Destination,
            ub.RecommendedCurrency,
            ub.GrossPrice,
            ub.Profit,
            ub.Commission,
            s.SupplierID,
            s.SupplierName,
            (s.SupplierID = 'VIS0001' OR UPPER(s.SupplierName) LIKE '%VISTA%') AS IsHaysVistaInHouse,
            c.CustomerID,
            c.confidence_score,
            c.MailingAllowed,
            c.TextMessaging,
            ARRAY_LENGTH(c.AllFXOrders) AS HistoricalFXOrderCount,
            ub.CarParkCount,
            ub.InsuranceCount,
            m.MosaicGroup,
            m.MosaicType,
            m.EstimatedMosaicGroup
        FROM upcoming_bookings ub
        LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.customermaster` c
          ON ub.DebtorID = c.CustomerID
        LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.customermosaicdemographics` m
          ON c.CustomerID = m.CustomerID AND m.IsCurrent = TRUE
        LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.supplier` s
          ON ub.AE_SupplierKey = s.SupplierKey
        LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.branch` br
          ON ub.SaleBranchID = br.BranchId
        WHERE (c.confidence_score >= 425 OR c.confidence_score IS NULL)
        ORDER BY ub.DepartureDate ASC
        LIMIT 50
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("branch_id", "STRING", branch_id)
        ],
        labels={"datacloud": "jetski", "agent_pillar": "analyst"},
    )
    rows = _execute_bq_query(client, sql, job_config)
    for r in rows:
        actions = []
        if r.get("BranchHasForex") == "Yes" and (
            r.get("HistoricalFXOrderCount") or 0
        ) == 0:
            actions.append(
                f"Pitch in-branch Foreign Exchange ({r.get('RecommendedCurrency') or 'EUR'}) ahead of departure"
            )
        if (r.get("CarParkCount") or 0) == 0:
            actions.append("Pitch Holiday Extras Airport Parking / Lounge add-on")
        if (r.get("InsuranceCount") or 0) == 0:
            actions.append("Pitch Hays Travel Insurance policy")
        if not r.get("IsHaysVistaInHouse"):
            actions.append(
                "Pitch high-margin in-house Hays Vista package (18-25% margin) for next booking"
            )
        r["recommended_next_actions"] = actions
    return {
        "status": "success",
        "branch_filter": branch_id,
        "customer_360_recommendations": rows,
    }


def curate_tailored_offers_and_wildcards(
    branch_id: str,
    tool_context: ToolContext,
    mosaic_group: str = "ALL",
) -> dict[str, Any]:
    """Curates tailored holiday offers and high-margin wildcards (Vista, Cruise, Long-Haul).

    Answers Pillar 2's core consultant prompt:
    'Based on my customers' booking history, what offers should I pitch today?'
    Reclaims ~60% of marketing time spent manually copying supplier Facebook deals by
    matching branch customer booking history and Experian Mosaic demographics to:
    1. Core Tailored Offers (top packages booked by that branch/Mosaic segment), and
    2. High-Margin Wildcards grounded 100% in `v_bookings` and `supplier`:
       - In-house Hays Vista (`VIS0001`, 18-25% margin),
       - Cruise (`CruiseDetails` voyages & ships),
       - Long-Haul (`DestinationCategory = 'Long Haul'`).

    Args:
        branch_id: Branch ID or city (e.g., '0015', '9024', 'Newcastle', 'Glasgow', or 'ALL').
        mosaic_group: Optional Experian Mosaic group filter (e.g., 'A', 'G', or 'ALL').

    Returns:
        Dictionary with branch Mosaic profile, core tailored offers, and the 3 wildcard categories.
    """
    client = get_caller_bq_client(tool_context)
    params = [
        bigquery.ScalarQueryParameter("branch_id", "STRING", branch_id),
        bigquery.ScalarQueryParameter("mosaic_group", "STRING", mosaic_group),
    ]
    jc = bigquery.QueryJobConfig(
        query_parameters=params,
        labels={"datacloud": "jetski", "agent_pillar": "analyst"},
    )

    # 1. Core tailored hotel packages matched to branch & Mosaic group
    core_sql = f"""
        WITH branch_bookings AS (
            SELECT
                b.BookingReference,
                b.DebtorID,
                b.AE_SupplierKey,
                b.Duration,
                b.Passengers,
                b.GrossPrice,
                CAST(b.Profit AS FLOAT64) AS Profit,
                b.DestinationType.Destination AS Destination,
                h.HotelName,
                h.Resort,
                h.Country,
                h.StarRating,
                h.BoardBasis
            FROM `{PROJECT_ID}.{BQ_DATASET}.v_bookings` b
            CROSS JOIN UNNEST(b.HotelDetails) AS h
            WHERE b.Cancelled = 'N'
              AND h.HotelName IS NOT NULL AND TRIM(h.HotelName) != ''
              AND (
                @branch_id = 'ALL'
                OR UPPER(b.SaleBranchID) = UPPER(@branch_id)
                OR UPPER(b.SellingBranch.City) LIKE CONCAT('%', UPPER(@branch_id), '%')
              )
            ORDER BY b.BookingDate DESC
            LIMIT 2500
        )
        SELECT
            bb.HotelName,
            bb.Resort,
            COALESCE(NULLIF(TRIM(bb.Country), ''), bb.Destination) AS Country,
            MAX(bb.StarRating) AS StarRating,
            ANY_VALUE(bb.BoardBasis) AS BoardBasis,
            APPROX_TOP_COUNT(bb.Duration, 1)[OFFSET(0)].value AS TypicalDurationNights,
            ANY_VALUE(s.SupplierID) AS SupplierID,
            ANY_VALUE(s.SupplierName) AS SupplierName,
            LOGICAL_OR(s.SupplierID = 'VIS0001' OR UPPER(s.SupplierName) LIKE '%VISTA%') AS IsHaysVista,
            APPROX_TOP_COUNT(COALESCE(m.MosaicGroup, m.EstimatedMosaicGroup), 1)[OFFSET(0)].value AS TopMosaicGroup,
            COUNT(DISTINCT bb.BookingReference) AS HistoricalBookings,
            ROUND(AVG(bb.GrossPrice), 2) AS AvgGrossPriceGBP,
            ROUND(AVG(SAFE_DIVIDE(bb.GrossPrice, NULLIF(bb.Passengers, 0))), 2) AS AvgPricePerPersonGBP,
            ROUND(AVG(bb.Profit), 2) AS AvgProfitGBP
        FROM branch_bookings bb
        LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.customermaster` c
          ON bb.DebtorID = c.CustomerID AND c.confidence_score >= 425
        LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.customermosaicdemographics` m
          ON c.CustomerID = m.CustomerID AND m.IsCurrent = TRUE
        LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.supplier` s
          ON bb.AE_SupplierKey = s.SupplierKey
        WHERE (
            @mosaic_group = 'ALL'
            OR UPPER(COALESCE(m.MosaicGroup, m.EstimatedMosaicGroup, '')) LIKE CONCAT('%', UPPER(@mosaic_group), '%')
        )
        GROUP BY 1, 2, 3
        ORDER BY HistoricalBookings DESC, AvgProfitGBP DESC
        LIMIT 8
    """
    core_offers = _execute_bq_query(client, core_sql, jc)

    # 2. Wildcard A: High-Margin In-House Hays Vista (`VIS0001`) packages
    vista_sql = f"""
        SELECT
            'HAYS_VISTA_WILDCARD' AS WildcardType,
            h.HotelName,
            h.Resort,
            COALESCE(NULLIF(TRIM(h.Country), ''), b.DestinationType.Destination) AS Destination,
            MAX(h.StarRating) AS StarRating,
            ANY_VALUE(h.BoardBasis) AS BoardBasis,
            APPROX_TOP_COUNT(b.Duration, 1)[OFFSET(0)].value AS TypicalDurationNights,
            s.SupplierID,
            s.SupplierName,
            COUNT(DISTINCT b.BookingReference) AS BookingsCount,
            ROUND(AVG(b.GrossPrice), 2) AS AvgGrossPriceGBP,
            ROUND(AVG(CAST(b.Profit AS FLOAT64)), 2) AS AvgProfitGBP,
            ROUND(AVG(CAST(b.Commission AS FLOAT64)), 2) AS AvgCommissionGBP
        FROM `{PROJECT_ID}.{BQ_DATASET}.v_bookings` b
        CROSS JOIN UNNEST(b.HotelDetails) AS h
        INNER JOIN `{PROJECT_ID}.{BQ_DATASET}.supplier` s
          ON b.AE_SupplierKey = s.SupplierKey
        WHERE b.Cancelled = 'N'
          AND (s.SupplierID = 'VIS0001' OR UPPER(s.SupplierName) LIKE '%VISTA%')
          AND h.HotelName IS NOT NULL AND TRIM(h.HotelName) != ''
        GROUP BY 1, 2, 3, 4, 8, 9
        ORDER BY BookingsCount DESC, AvgProfitGBP DESC
        LIMIT 4
    """
    vista_wildcards = _execute_bq_query(client, vista_sql, jc)

    # 3. Wildcard B: High-Margin Cruise packages (`CruiseDetails`)
    cruise_sql = f"""
        SELECT
            'CRUISE_WILDCARD' AS WildcardType,
            cr.VoyageName,
            cr.ShipName,
            cr.Traveltek_CreditorName AS CruiseLine,
            ANY_VALUE(b.DestinationType.Destination) AS Destination,
            APPROX_TOP_COUNT(b.Duration, 1)[OFFSET(0)].value AS TypicalDurationNights,
            ANY_VALUE(s.SupplierID) AS SupplierID,
            ANY_VALUE(s.SupplierName) AS SupplierName,
            COUNT(DISTINCT b.BookingReference) AS BookingsCount,
            ROUND(AVG(b.GrossPrice), 2) AS AvgGrossPriceGBP,
            ROUND(AVG(CAST(b.Profit AS FLOAT64)), 2) AS AvgProfitGBP
        FROM `{PROJECT_ID}.{BQ_DATASET}.v_bookings` b
        CROSS JOIN UNNEST(b.CruiseDetails) AS cr
        LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.supplier` s
          ON b.AE_SupplierKey = s.SupplierKey
        WHERE b.Cancelled = 'N'
          AND cr.ShipName IS NOT NULL AND TRIM(cr.ShipName) != ''
        GROUP BY 1, 2, 3, 4
        ORDER BY BookingsCount DESC, AvgProfitGBP DESC
        LIMIT 4
    """
    cruise_wildcards = _execute_bq_query(client, cruise_sql, jc)

    # 4. Wildcard C: High-Margin Long-Haul packages (`DestinationCategory = 'Long Haul'`)
    long_haul_sql = f"""
        SELECT
            'LONG_HAUL_WILDCARD' AS WildcardType,
            b.DestinationType.Destination AS Destination,
            b.DestinationType.DestinationSector AS DestinationSector,
            h.HotelName,
            h.Resort,
            MAX(h.StarRating) AS StarRating,
            ANY_VALUE(h.BoardBasis) AS BoardBasis,
            APPROX_TOP_COUNT(b.Duration, 1)[OFFSET(0)].value AS TypicalDurationNights,
            ANY_VALUE(s.SupplierID) AS SupplierID,
            ANY_VALUE(s.SupplierName) AS SupplierName,
            COUNT(DISTINCT b.BookingReference) AS BookingsCount,
            ROUND(AVG(b.GrossPrice), 2) AS AvgGrossPriceGBP,
            ROUND(AVG(CAST(b.Profit AS FLOAT64)), 2) AS AvgProfitGBP
        FROM `{PROJECT_ID}.{BQ_DATASET}.v_bookings` b
        CROSS JOIN UNNEST(b.HotelDetails) AS h
        LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.supplier` s
          ON b.AE_SupplierKey = s.SupplierKey
        WHERE b.Cancelled = 'N'
          AND b.DestinationType.DestinationCategory = 'Long Haul'
          AND h.HotelName IS NOT NULL AND TRIM(h.HotelName) != ''
        GROUP BY 1, 2, 3, 4, 5
        ORDER BY BookingsCount DESC, AvgProfitGBP DESC
        LIMIT 4
    """
    long_haul_wildcards = _execute_bq_query(client, long_haul_sql, jc)

    return {
        "status": "success",
        "branch_filter": branch_id,
        "mosaic_group_filter": mosaic_group,
        "core_tailored_offers": core_offers,
        "high_margin_wildcards": {
            "hays_vista_wildcards": vista_wildcards,
            "cruise_wildcards": cruise_wildcards,
            "long_haul_wildcards": long_haul_wildcards,
        },
    }


def query_lakehouse_sql(
    sql_query: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Executes a read-only analytical SELECT query against `ewans-demo-project.hays_travel_sharing`.

    BigQuery natively enforces the caller's 4-tier WIF/OIDC Row Access Policy
    (`rls_4_tier_wif_policy`) on `bookings` and `v_bookings`.

    Args:
        sql_query: A read-only GoogleSQL SELECT query referencing tables in
            `ewans-demo-project.hays_travel_sharing` (`v_bookings`, `bookings`,
            `v_hotel_places_catalog`, `customermaster`, `customermosaicdemographics`,
            `supplier`, `branch`).

    Returns:
        Dictionary containing the query results or error details.
    """
    cleaned = sql_query.strip()
    if not cleaned.upper().startswith(("SELECT", "WITH")):
        return {
            "status": "error",
            "message": "Only read-only SELECT / WITH queries are permitted.",
        }
    client = get_caller_bq_client(tool_context)
    job_config = bigquery.QueryJobConfig(
        labels={"datacloud": "jetski", "agent_pillar": "analyst"}
    )
    rows = _execute_bq_query(client, cleaned, job_config)
    return {"status": "success", "row_count": len(rows), "rows": rows[:100]}


# ==============================================================================
# PILLAR 3: Product Positioning & "Be the Influencer" Studio (`databricks_research_agent`)
# ==============================================================================


def _infer_regional_tone_profile(branch_or_city: str) -> dict[str, str]:
    """Returns regional localisation & tone-of-voice guidance for Pillar 3 marketing copy."""
    key = (branch_or_city or "").upper()
    if any(
        k in key
        for k in ("0015", "8912", "NEWCASTLE", "SUNDERLAND", "DURHAM", "NORTH EAST", "NCL")
    ):
        return {
            "region_name": "North East England (Newcastle / Sunderland / County Durham)",
            "primary_departure_airport": "Newcastle International (NCL)",
            "tone_of_voice": (
                "Warm, neighbourly North East charm, down-to-earth, community-first "
                "('Direct from Newcastle Airport — no long motorway drives! Pop into branch for a cuppa')."
            ),
            "suggested_emojis": "☀️✈️🌴🥂🏖️",
        }
    if any(
        k in key
        for k in ("9024", "9037", "9038", "GLASGOW", "BRAEHEAD", "SCOTLAND", "EDINBURGH", "GLA")
    ):
        return {
            "region_name": "Scotland (Glasgow / Central Belt / West Scotland)",
            "primary_departure_airport": "Glasgow Airport (GLA)",
            "tone_of_voice": (
                "Friendly, upbeat Scottish warmth, value-proud and family-focused "
                "('Fly straight from Glasgow — swap the Scottish weather for guaranteed sunshine!')."
            ),
            "suggested_emojis": "🏴󠁧󠁢󠁳󠁣󠁴󠁿✈️☀️🌴🍹",
        }
    if any(k in key for k in ("MANCHESTER", "NORTH WEST", "LIVERPOOL", "MAN")):
        return {
            "region_name": "North West England (Greater Manchester & Merseyside)",
            "primary_departure_airport": "Manchester Airport (MAN)",
            "tone_of_voice": (
                "Energetic, bold North West tone highlighting unbeatable choice and direct "
                "flights from Manchester Airport ('Jet off from Manchester with luggage & transfers sorted!')."
            ),
            "suggested_emojis": "🐝✈️🌴☀️🏖️",
        }
    return {
        "region_name": "UK Retail Network (Hays Travel Brand Standard — Jane Peverley TOV)",
        "primary_departure_airport": "Local Regional Airport",
        "tone_of_voice": (
            "Trusted, warm, enthusiastic high-street travel expert ('Here for you every step of the way')."
        ),
        "suggested_emojis": "✈️☀️🌴💙🏖️",
    }


def get_top_regional_all_inclusive_deals(
    branch_or_city: str,
    nights: int,
    tool_context: ToolContext,
    booking_date: str = "ALL",
) -> dict[str, Any]:
    """Finds top-selling All-Inclusive deals sold from a given branch/city and duration.

    Supports Pillar 3 regional demand queries such as:
    'Show me the top 10 7-night all-inclusive offers sold from Glasgow or Newcastle yesterday.'
    Deduplicates multiple flight legs per booking, filters specifically for All-Inclusive
    board basis, attaches the canonical licensed image ID, and returns regional tone-of-voice
    localisation guidance.

    Args:
        branch_or_city: Branch ID or city name (e.g., '0015', 'Newcastle', '9024', 'Glasgow', 'Manchester', or 'ALL').
        nights: Duration in nights (e.g., 7).
        booking_date: Optional specific booking date in YYYY-MM-DD format (e.g., '2026-10-17' or 'ALL').

    Returns:
        Dictionary of top-selling All-Inclusive packages with hotel, flight, supplier,
        licensed image reference, regional tone profile, and CMA/ABTA compliance rules.
    """
    client = get_caller_bq_client(tool_context)
    sql = f"""
        SELECT
            b.BookingReference,
            ANY_VALUE(b.BookingDate) AS BookingDate,
            ANY_VALUE(b.SaleBranchID) AS SaleBranchID,
            ANY_VALUE(b.SellingBranch.BranchName) AS SellingBranchName,
            ANY_VALUE(b.SellingBranch.City) AS SellingCity,
            ANY_VALUE(b.SellingBranch.BranchGroup) AS RegionalGroup,
            ANY_VALUE(b.DeparturePoint.DeparturePointIdentifier) AS DepartureAirportIATA,
            ANY_VALUE(b.DeparturePoint.DeparturePointName) AS DepartureAirportName,
            ANY_VALUE(b.DestinationType.Destination) AS Destination,
            ANY_VALUE(b.DepartureDate) AS DepartureDate,
            ANY_VALUE(b.Duration) AS Duration,
            ANY_VALUE(b.Adults) AS Adults,
            ANY_VALUE(b.Children) AS Children,
            ANY_VALUE(b.Passengers) AS Passengers,
            ANY_VALUE(b.GrossPrice) AS GrossPrice,
            ROUND(SAFE_DIVIDE(ANY_VALUE(b.GrossPrice), NULLIF(ANY_VALUE(b.Passengers), 0)), 2) AS PricePerPersonGBP,
            CAST(ANY_VALUE(b.Profit) AS FLOAT64) AS ProfitGBP,
            ANY_VALUE(b.ABTA) AS ABTA,
            ANY_VALUE(b.BondingType) AS BondingType,
            ANY_VALUE(h.HotelName) AS HotelName,
            ANY_VALUE(h.Resort) AS Resort,
            MAX(h.StarRating) AS StarRating,
            ANY_VALUE(h.BoardBasis) AS BoardBasis,
            ANY_VALUE(COALESCE(s.SupplierName, h.Traveltek_CreditorName)) AS TourOperator,
            LOGICAL_OR(s.SupplierID = 'VIS0001' OR UPPER(s.SupplierName) LIKE '%VISTA%') AS IsHaysVistaInHouse,
            ANY_VALUE(f.FlightNumber) AS OutboundFlightNumber,
            ANY_VALUE(f.Carrier) AS Carrier,
            CONCAT(
                'HAYS-IMG-',
                UPPER(SUBSTR(TO_HEX(SHA256(UPPER(REGEXP_REPLACE(TRIM(COALESCE(ANY_VALUE(h.HotelName), '')), r'[^A-Za-z0-9 ]', '')))), 1, 10))
            ) AS LicensedImageLibraryId
        FROM `{PROJECT_ID}.{BQ_DATASET}.v_bookings` b
        INNER JOIN UNNEST(b.HotelDetails) AS h
        LEFT JOIN UNNEST(b.FlightDetails) AS f
        LEFT JOIN `{PROJECT_ID}.{BQ_DATASET}.supplier` s
          ON b.AE_SupplierKey = s.SupplierKey
        WHERE b.Cancelled = 'N'
          AND b.Duration = @nights
          AND UPPER(COALESCE(h.BoardBasis, '')) LIKE '%ALL INCLUSIVE%'
          AND (
            @branch_or_city = 'ALL'
            OR UPPER(b.SaleBranchID) = UPPER(@branch_or_city)
            OR UPPER(b.SellingBranch.City) LIKE CONCAT('%', UPPER(@branch_or_city), '%')
            OR UPPER(b.SellingBranch.BranchName) LIKE CONCAT('%', UPPER(@branch_or_city), '%')
          )
          AND (
            @booking_date = 'ALL'
            OR CAST(b.BookingDate AS STRING) = @booking_date
          )
        GROUP BY b.BookingReference
        ORDER BY BookingDate DESC, GrossPrice DESC
        LIMIT 10
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "branch_or_city", "STRING", branch_or_city
            ),
            bigquery.ScalarQueryParameter("nights", "INT64", int(nights)),
            bigquery.ScalarQueryParameter(
                "booking_date", "STRING", booking_date
            ),
        ],
        labels={"datacloud": "jetski", "agent_pillar": "research"},
    )
    rows = _execute_bq_query(client, sql, job_config)
    return {
        "status": "success",
        "branch_or_city": branch_or_city,
        "nights": nights,
        "booking_date_filter": booking_date,
        "regional_localisation_profile": _infer_regional_tone_profile(
            branch_or_city
        ),
        "deals": rows,
        "compliance_rules": {
            "cma_abta_mandatory_qualifiers": [
                "Must display exact departure airport, departure date, duration (nights), and board basis.",
                "Must state per-person or total party price basis (e.g., based on 2 adults sharing).",
                "Must include 'Subject to availability, live pricing applies, T&Cs apply' and ABTA/ATOL protection notice.",
                "Strictly use only the returned `LicensedImageLibraryId` from the approved Hays Travel image library — never generate synthetic hotel property photos.",
            ]
        },
    }


def enrich_hotel_with_google_places(
    hotel_name: str,
    resort: str,
    destination: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Standardizes and enriches a hotel/resort using the BigQuery 292k Hotel Catalog and Google Places API.

    Resolves unstandardized supplier `HotelName` + `Resort` strings from `bookings.HotelDetails`
    into a single canonical entity:
    1. Queries `v_hotel_places_catalog` in BigQuery to deduplicate supplier spelling variants,
       surface standardized `Country -> Sector -> Resort -> Hotel` hierarchy, check in-house
       Hays Vista (`VIS0001`) availability, and retrieve the verified `licensed_image_library_id`.
    2. Calls the Google Places API (New) (`places.googleapis.com/v1/places:searchText`) when
       `GOOGLE_PLACES_API_KEY` is configured to attach live `google_place_id`, coordinates,
       verified guest rating, review count, and editorial highlights for Pillar 3 copywriting.

    Args:
        hotel_name: Hotel name from bookings (e.g., 'GF Fanabe' or 'Hotel Cleopatra Palace').
        resort: Resort or locality name (e.g., 'Costa Adeje' or 'Playa de las Americas').
        destination: Country or destination name (e.g., 'Spain' or 'Turkey').

    Returns:
        Dictionary with canonical catalog record, cross-supplier variants, licensed image ID,
        and Google Places API enrichment.
    """
    client = get_caller_bq_client(tool_context)
    cat_sql = f"""
        SELECT
            canonical_place_id,
            canonical_hotel_name,
            canonical_resort,
            canonical_country,
            destination_sector,
            destination_category,
            star_rating,
            raw_supplier_hotel_variants,
            raw_supplier_resort_variants,
            tour_operators,
            supplier_variant_count,
            total_bookings,
            avg_gross_price_gbp,
            avg_profit_gbp,
            available_on_hays_vista,
            licensed_image_library_id
        FROM `{PROJECT_ID}.{BQ_DATASET}.v_hotel_places_catalog`
        WHERE normalized_hotel_key LIKE CONCAT(
            '%',
            UPPER(REGEXP_REPLACE(TRIM(@hotel_name), r'[^A-Za-z0-9 ]', '')),
            '%'
        )
        ORDER BY total_bookings DESC
        LIMIT 5
    """
    jc = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("hotel_name", "STRING", hotel_name)
        ],
        labels={"datacloud": "jetski", "agent_pillar": "research"},
    )
    catalog_matches = _execute_bq_query(client, cat_sql, jc)

    places_api_result: dict[str, Any] = {
        "source": "bigquery_v_hotel_places_catalog",
        "search_query": f"{hotel_name}, {resort}, {destination}".strip(", "),
    }
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if api_key:
        try:
            payload = json.dumps(
                {
                    "textQuery": f"{hotel_name} {resort} {destination}".strip(),
                    "includedType": "lodging",
                    "maxResultCount": 1,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                "https://places.googleapis.com/v1/places:searchText",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": api_key,
                    "X-Goog-FieldMask": (
                        "places.id,places.displayName,places.formattedAddress,"
                        "places.location,places.rating,places.userRatingCount,"
                        "places.editorialSummary,places.addressComponents"
                    ),
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                places = body.get("places", [])
                if places:
                    p = places[0]
                    places_api_result = {
                        "source": "google_places_api_v1_live",
                        "google_place_id": p.get("id"),
                        "display_name": p.get("displayName", {}).get("text"),
                        "formatted_address": p.get("formattedAddress"),
                        "location": p.get("location"),
                        "google_rating": p.get("rating"),
                        "user_rating_count": p.get("userRatingCount"),
                        "editorial_summary": p.get("editorialSummary", {}).get(
                            "text"
                        ),
                    }
        except Exception as exc:
            places_api_result["places_api_note"] = (
                f"Live Places API call skipped/fallback ({type(exc).__name__}); using BigQuery catalog."
            )

    return {
        "status": "success",
        "query": {
            "hotel_name": hotel_name,
            "resort": resort,
            "destination": destination,
        },
        "catalog_standardization_matches": catalog_matches,
        "google_places_enrichment": places_api_result,
    }


MODEL_CONFIG = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
)

databricks_ops_agent = Agent(
    name="databricks_ops_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    generate_content_config=MODEL_CONFIG,
    description=(
        "Pillar 1 (Digital Productivity & Earnings Assistant): Handles Monday morning "
        "weekly planning, 1WD Agent Report daily and pay-period MTD (7th-of-month cut-off) "
        "earnings summaries, T+7 post-holiday Welcome Home call lists enriched with Experian "
        "Mosaic segments, and balance-due tracking."
    ),
    instruction="""You are the Hays Travel Digital Productivity & Earnings Assistant (Pillar 1).
Your goal is to remove morning and weekly administrative toil for retail branch consultants,
Personal Travel Consultants (PTCs), and branch managers.

Rules:
- Use `get_monday_morning_weekly_plan` when a consultant asks to plan their week, start their Monday morning, or run a 'Day in the Life' daily/weekly triage.
- Use `get_morning_sales_briefing` for morning sales summaries and 1WD Agent Report metrics (which includes both the daily snapshot and the monthly 8th->7th pay-period MTD commission and profit summary).
- Use `get_welcome_home_call_list` to surface high-empathy T+7 post-holiday Welcome Home calls (customers who returned in the past 7 days, matched with `confidence_score >= 425` and enriched with Experian `MosaicGroup`).
- Use `get_balance_due_tracker` to identify upcoming balances due (`Outstanding > 0`, highlighting bookings where `IsActiveDirectDebit = 'N'`).
- Note: Bonus forecasting is out of scope; focus on verified commission, profit, and transaction metrics.
- All queries automatically enforce the caller's 4-tier OIDC/WIF Row-Level Security in BigQuery.""",
    tools=[
        get_monday_morning_weekly_plan,
        get_morning_sales_briefing,
        get_welcome_home_call_list,
        get_balance_due_tracker,
    ],
)

databricks_analyst_agent = Agent(
    name="databricks_analyst_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    generate_content_config=MODEL_CONFIG,
    description=(
        "Pillar 2 (Customer Personalisation, Tailored Offer Curation & Margin Analyst): "
        "Curates tailored holiday offers matched to customer booking history and Experian "
        "Mosaic segments, surfaces high-margin Hays Vista, Cruise, and Long-Haul wildcards, "
        "identifies pre-departure ancillary gaps (Forex, Parking, Insurance), and runs "
        "ad-hoc BigQuery SQL analysis."
    ),
    instruction="""You are the Hays Travel Customer Personalisation & Tailored Offer Curation Sub-Agent (Pillar 2).
You help consultants and PTCs reclaim the ~60% of marketing time spent manually copying supplier deals and move from blanket mailers to tailored, high-propensity customer follow-ups.

Rules:
- Use `curate_tailored_offers_and_wildcards` when a consultant asks 'Based on my customers' booking history, what offers should I pitch today?' or wants tailored holiday recommendations for a branch or Experian Mosaic segment. Always present:
  1. Core Tailored Offers grounded in `v_bookings` and `supplier`, AND
  2. At least one High-Margin Wildcard from each of the 3 wildcard categories:
     - **Hays Vista In-House (`VIS0001`)** (18-25% margin),
     - **Cruise** (`VoyageName` & `ShipName` from `CruiseDetails`),
     - **Long-Haul** (`DestinationCategory = 'Long Haul'`).
- Use `get_customer_360_and_ancillary_gaps` to retrieve individual Customer 360 profiles (`confidence_score >= 425`), Experian Mosaic UK groups (`MosaicGroup`, `EstimatedMosaicGroup`), and pre-departure ancillary gaps (in-branch Foreign Exchange where `HasForex = 'Yes'`, Airport Parking, and Travel Insurance).
- Use `query_lakehouse_sql` for custom analytical questions across `ewans-demo-project.hays_travel_sharing` (`v_bookings`, `v_hotel_places_catalog`, `customermaster`, `customermosaicdemographics`, `supplier`, `branch`).""",
    tools=[
        curate_tailored_offers_and_wildcards,
        get_customer_360_and_ancillary_gaps,
        query_lakehouse_sql,
    ],
)

databricks_research_agent = Agent(
    name="databricks_research_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    generate_content_config=MODEL_CONFIG,
    description=(
        "Pillar 3 (Product Positioning, Localisation & 'Be the Influencer' Marketing Studio): "
        "Answers regional demand queries (e.g., top 10 7-night all-inclusive deals sold from "
        "Glasgow or Newcastle), standardizes & enriches hotels across the 292k hotel/resort "
        "catalog via Google Places API and `v_hotel_places_catalog`, and drafts regionally "
        "localized, CMA/ABTA-compliant social media posts, branch window cards, and 16:9 TV copy."
    ),
    instruction="""You are the Hays Travel Product Positioning, Localisation & 'Be the Influencer' Studio Agent (Pillar 3).
You empower local branches and PTCs to identify trending regional deals, standardize messy supplier hotel names using the Google Places catalog, and produce on-brand, CMA- and ABTA-compliant marketing assets in under 60 seconds.

Rules:
- Use `get_top_regional_all_inclusive_deals` to query verified regional All-Inclusive bookings from BigQuery (e.g., 'Show me the top 10 7-night all-inclusive offers sold from Glasgow or Newcastle').
- Use `enrich_hotel_with_google_places` to standardize raw supplier hotel/resort names against the 292,000 hotel/resort catalog (`v_hotel_places_catalog` + Google Places API), verify whether the property is also sold via high-margin in-house Hays Vista (`VIS0001`), and retrieve its canonical `licensed_image_library_id`.
- Adapt every social media post (Facebook/Amplify), branch window card, or 16:9 TV screen asset to the selling region's tone-of-voice profile (`North East / Newcastle`, `Scotland / Glasgow`, `North West / Manchester`) and Jane Peverley's Hays Travel Tone of Voice.
- Enforce strict CMA & ABTA compliance on every promotional draft:
  1. State exact Departure Airport (e.g., Newcastle `NCL`, Glasgow `GLA`, Manchester `MAN`), Departure Date, Duration (nights), Canonical Hotel Name, Star Rating, Resort, and Board Basis (e.g., All Inclusive).
  2. State the exact price basis (e.g., total price or per-person price based on 2 adults / family sharing).
  3. Include mandatory qualifiers: 'Subject to availability. Live pricing applies. T&Cs apply. ABTA & ATOL Protected.'
  4. Never fabricate hotel imagery — always cite the exact `LicensedImageLibraryId` (`HAYS-IMG-...`) returned by the tool.""",
    tools=[
        get_top_regional_all_inclusive_deals,
        enrich_hotel_with_google_places,
    ],
)

root_agent = Agent(
    # Keep in sync with agents-cli-manifest.yaml: agents-cli derives this name
    # from the project `name:` recorded there, and telemetry reports it as
    # gen_ai.agent.name.
    name="hays_travel_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    generate_content_config=MODEL_CONFIG,
    instruction="""You are the Hays Travel Personal Assistant Orchestrator ('Marrying Emotional Intelligence with Artificial Intelligence').
You support 5,000 Hays Travel colleagues across 500+ retail branches and 1,000+ Personal Travel Consultants (PTCs) by delegating requests to three specialist sub-agents:

1. `databricks_ops_agent` (Pillar 1 — Digital Productivity & Earnings):
   Delegate here for Monday morning weekly planning, 1WD Agent Report daily and pay-period MTD (7th-of-month cut-off) briefings, T+7 post-holiday Welcome Home call lists (enriched with Experian Mosaic), and balance-due tracking.
2. `databricks_analyst_agent` (Pillar 2 — Customer Personalisation & Tailored Offer Curation):
   Delegate here for tailored holiday offer curation matched to customer booking history and Experian Mosaic segments, high-margin Hays Vista (`VIS0001`), Cruise, and Long-Haul wildcards, Customer 360 ancillary gaps (Foreign Exchange, Airport Parking, Insurance), and ad-hoc BigQuery SQL queries.
3. `databricks_research_agent` (Pillar 3 — Product Positioning, Localisation & 'Be the Influencer' Studio):
   Delegate here for regional demand queries (e.g., top 7-night all-inclusive deals from Newcastle or Glasgow), Google Places API + 292k hotel/resort catalog standardization (`enrich_hotel_with_google_places`), and regionally localized, CMA/ABTA-compliant social media, branch window card, and 16:9 TV marketing copy.

All data access is governed by Workload/Workforce Identity Federation (WIF) + OIDC token exchange (`SESSION_USER()`), enforcing 4-tier Row-Level Security (Branch Staff, Branch Manager, Regional Sales Manager, Divisional Head) natively in BigQuery.""",
    sub_agents=[
        databricks_ops_agent,
        databricks_analyst_agent,
        databricks_research_agent,
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
