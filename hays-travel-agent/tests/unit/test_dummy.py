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
"""Unit and integration verification tests for the Hays Travel ADK Multi-Agent System."""

from unittest.mock import MagicMock

from app.agent import (
    MODEL,
    curate_tailored_offers_and_wildcards,
    databricks_analyst_agent,
    databricks_ops_agent,
    databricks_research_agent,
    enrich_hotel_with_google_places,
    get_balance_due_tracker,
    get_customer_360_and_ancillary_gaps,
    get_monday_morning_weekly_plan,
    get_morning_sales_briefing,
    get_top_regional_all_inclusive_deals,
    get_welcome_home_call_list,
    query_lakehouse_sql,
    root_agent,
)


def test_agent_hierarchy_and_model() -> None:
    """Verifies the root_agent + 3 sub_agents architecture and gemini-4-argon model."""
    assert root_agent.name == "hays_travel_agent"
    assert root_agent.model.model == MODEL == "gemini-4-argon"
    sub_names = [sa.name for sa in root_agent.sub_agents]
    assert sub_names == [
        "databricks_ops_agent",
        "databricks_analyst_agent",
        "databricks_research_agent",
    ]
    for sa in [
        databricks_ops_agent,
        databricks_analyst_agent,
        databricks_research_agent,
    ]:
        assert sa.model.model == "gemini-4-argon"


def test_bigquery_tools_live_execution() -> None:
    """Verifies all Pillar 1, 2, and 3 BigQuery tools against the hydrated 11.6M-row dataset."""
    mock_ctx = MagicMock()
    mock_ctx.state = {}
    mock_ctx.get_auth_response.return_value = None

    # Action 2 (Pillar 1): 1WD briefing + pay-period MTD summary, T+7 Welcome Home + Mosaic, Monday Planner
    briefing = get_morning_sales_briefing("2026-10-17", mock_ctx)
    assert briefing["status"] == "success"
    assert briefing["summary_1wd"]["total_bookings"] >= 2000
    assert briefing["pay_period_mtd_summary"]["mtd_bookings"] >= 2000

    welcome = get_welcome_home_call_list("2026-10-16", mock_ctx, lookback_days=7)
    assert welcome["status"] == "success"
    assert welcome["lookback_days"] == 7
    assert len(welcome["welcome_home_calls"]) >= 10
    assert "MosaicGroup" in welcome["welcome_home_calls"][0]

    monday_plan = get_monday_morning_weekly_plan("2026-10-17", mock_ctx)
    assert monday_plan["status"] == "success"
    assert len(monday_plan["priority_1_t_plus_7_welcome_home_calls"]) >= 5

    balance = get_balance_due_tracker("2026-12-31", mock_ctx)
    assert balance["status"] == "success"
    assert len(balance["outstanding_bookings"]) >= 10

    # Action 3 (Pillar 2): Customer 360 + Tailored Offer Curation & 3 High-Margin Wildcard categories
    c360 = get_customer_360_and_ancillary_gaps("0015", mock_ctx)
    assert c360["status"] == "success"
    assert len(c360["customer_360_recommendations"]) >= 10

    curated = curate_tailored_offers_and_wildcards("0015", mock_ctx, mosaic_group="ALL")
    assert curated["status"] == "success"
    assert len(curated["core_tailored_offers"]) >= 5
    wildcards = curated["high_margin_wildcards"]
    assert len(wildcards["hays_vista_wildcards"]) >= 1
    assert len(wildcards["cruise_wildcards"]) >= 1
    assert len(wildcards["long_haul_wildcards"]) >= 1

    # Action 4 (Pillar 3): Deduplicated All-Inclusive Regional Deals + Regional TOV + Google Places Catalog
    deals = get_top_regional_all_inclusive_deals("Glasgow", 7, mock_ctx)
    assert deals["status"] == "success"
    assert len(deals["deals"]) == 10
    assert "ALL INCLUSIVE" in deals["deals"][0]["BoardBasis"].upper()
    assert deals["deals"][0]["LicensedImageLibraryId"].startswith("HAYS-IMG-")
    assert "Scotland" in deals["regional_localisation_profile"]["region_name"]

    top_hotel = deals["deals"][0]["HotelName"]
    top_resort = deals["deals"][0].get("Resort") or ""
    top_dest = deals["deals"][0].get("Destination") or ""
    places_res = enrich_hotel_with_google_places(
        top_hotel, top_resort, top_dest, mock_ctx
    )
    assert places_res["status"] == "success"
    assert len(places_res["catalog_standardization_matches"]) >= 1
    assert places_res["catalog_standardization_matches"][0][
        "licensed_image_library_id"
    ].startswith("HAYS-IMG-")

    sql_res = query_lakehouse_sql(
        "SELECT COUNT(*) AS cnt FROM `ewans-demo-project.hays_travel_sharing.v_bookings`",
        mock_ctx,
    )
    assert sql_res["status"] == "success"
    assert sql_res["rows"][0]["cnt"] >= 867461
