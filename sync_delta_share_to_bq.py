#!/usr/bin/env python3
"""Hydrates the 5 Hays Travel Azure Databricks Delta Sharing tables into BigQuery.

Target Dataset: ewans-demo-project:hays_travel_sharing (europe-west2)
Source Share:   google_hays_marketing_agent_share.google_sharing
Tables:         bookings, customermaster, customermosaicdemographics, supplier,
branch

Handles:
1. Delta Column Mapping (`delta.columnMapping.physicalName` -> logical column
   names, both top-level and nested inside STRUCT / ARRAY<STRUCT> columns).
2. Spark 3-level Parquet LIST inference (`ParquetOptions(enable_list_inference=True)`).
3. Preserving Unity Catalog column descriptions across `WRITE_TRUNCATE`.
4. Re-applying the 4-tier WIF/OIDC Row Access Policy (`rls_4_tier_wif_policy`) on
   `bookings` after hydration.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any
import urllib.request

from google.cloud import bigquery
from google.oauth2.credentials import Credentials
import pyarrow as pa
import pyarrow.parquet as pq

def _load_share_credentials() -> tuple[str, str]:
  """Loads Delta Sharing endpoint and bearer token from env vars or .share profile JSON."""
  profile_path = os.getenv("DELTA_SHARING_PROFILE_PATH")
  if profile_path and os.path.isfile(profile_path):
    with open(profile_path, "r", encoding="utf-8") as f:
      profile = json.load(f)
    return profile["endpoint"].rstrip("/"), profile["bearerToken"]
  endpoint = os.getenv("DELTA_SHARING_ENDPOINT", "").rstrip("/")
  token = os.getenv("DELTA_SHARING_BEARER_TOKEN", "")
  if not endpoint or not token:
    raise RuntimeError(
        "Set DELTA_SHARING_PROFILE_PATH=/path/to/profile.share or both "
        "DELTA_SHARING_ENDPOINT and DELTA_SHARING_BEARER_TOKEN environment variables."
    )
  return endpoint, token


SHARE = os.getenv("DELTA_SHARING_SHARE", "google_hays_marketing_agent_share")
SCHEMA = os.getenv("DELTA_SHARING_SCHEMA", "google_sharing")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "ewans-demo-project")
DATASET_ID = os.getenv("HAYS_BQ_DATASET", "hays_travel_sharing")
LOCATION = os.getenv("HAYS_BQ_LOCATION", "europe-west2")
TABLES = [
    "supplier",
    "branch",
    "customermaster",
    "customermosaicdemographics",
    "bookings",
]


def sanitize_col_name(name: str) -> str:
  if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
    return name
  m = re.search(r"substring\(([a-zA-Z0-9_]+)", name)
  if m:
    return m.group(1)
  cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_")
  return cleaned if cleaned and not cleaned[0].isdigit() else f"col_{cleaned}"


def build_phys_to_logical_map(
    delta_fields: list[dict[str, Any]],
) -> dict[str, tuple[str, dict[str, Any]]]:
  """Builds a recursive physicalName -> (logicalName, nestedMap) lookup from Delta schemaString."""
  mapping: dict[str, tuple[str, dict[str, Any]]] = {}
  for f in delta_fields:
    phys = f.get("metadata", {}).get(
        "delta.columnMapping.physicalName", f["name"]
    )
    logical = sanitize_col_name(f["name"])
    sub_map: dict[str, Any] = {}
    ftype = f["type"]
    if isinstance(ftype, dict):
      if ftype.get("type") == "struct":
        sub_map = build_phys_to_logical_map(ftype.get("fields", []))
      elif (
          ftype.get("type") == "array"
          and isinstance(ftype.get("elementType"), dict)
          and ftype["elementType"].get("type") == "struct"
      ):
        sub_map = build_phys_to_logical_map(
            ftype["elementType"].get("fields", [])
        )
    mapping[phys] = (logical, sub_map)
  return mapping


def build_description_map(delta_fields: list[dict[str, Any]]) -> dict[str, str]:
  """Extracts top-level column descriptions from the Delta schemaString."""
  desc_map: dict[str, str] = {}
  for f in delta_fields:
    logical = sanitize_col_name(f["name"])
    comment = f.get("metadata", {}).get("comment")
    if comment:
      desc_map[logical] = comment
  return desc_map


def rename_pa_array(arr: pa.Array, sub_map: dict[str, Any]) -> pa.Array:
  """Recursively renames nested struct fields inside a PyArrow StructArray or ListArray."""
  if not sub_map:
    return arr
  t = arr.type
  if pa.types.is_struct(t):
    new_fields = []
    children = []
    for i in range(t.num_fields):
      sf = t[i]
      logical_name, nested_map = sub_map.get(sf.name, (sf.name, {}))
      child_arr = rename_pa_array(arr.field(i), nested_map)
      new_fields.append(
          pa.field(logical_name, child_arr.type, nullable=sf.nullable)
      )
      children.append(child_arr)
    return pa.StructArray.from_arrays(
        children, fields=new_fields, mask=arr.is_null()
    )
  elif pa.types.is_list(t) and pa.types.is_struct(t.value_type):
    flat_renamed = rename_pa_array(arr.values, sub_map)
    list_type = pa.list_(
        pa.field(
            t.value_field.name, flat_renamed.type, nullable=t.value_field.nullable
        )
    )
    return pa.ListArray.from_arrays(
        arr.offsets, flat_renamed, type=list_type, mask=arr.is_null()
    )
  return arr


def rename_pa_table(
    table_pa: pa.Table, delta_schema: dict[str, Any]
) -> pa.Table:
  """Renames physical `col-*` column names (top-level and nested structs) to logical names."""
  mapping = build_phys_to_logical_map(delta_schema.get("fields", []))
  new_cols = []
  new_fields = []
  for i, col_name in enumerate(table_pa.column_names):
    col = table_pa.column(i)
    logical_name, sub_map = mapping.get(
        col_name, (sanitize_col_name(col_name), {})
    )
    if sub_map:
      chunks = [rename_pa_array(chunk, sub_map) for chunk in col.chunks]
      col = pa.chunked_array(chunks)
    new_cols.append(col)
    new_fields.append(
        pa.field(
            logical_name, col.type, nullable=table_pa.schema.field(i).nullable
        )
    )
  return pa.Table.from_arrays(new_cols, schema=pa.schema(new_fields))


def get_bq_client() -> bigquery.Client:
  token = (
      subprocess.check_output(
          ["gcloud", "auth", "print-access-token"],
          env={**os.environ, "CLOUDSDK_METRICS_ENVIRONMENT": "datacloud.jetski"},
      )
      .decode()
      .strip()
  )
  return bigquery.Client(
      project=PROJECT_ID, location=LOCATION, credentials=Credentials(token=token)
  )


def apply_bookings_rls_policy(client: bigquery.Client) -> None:
  """Re-applies the 4-tier WIF/OIDC Row Access Policy on bookings after WRITE_TRUNCATE."""
  rls_sql = f"""
    CREATE OR REPLACE ROW ACCESS POLICY rls_4_tier_wif_policy
    ON `{PROJECT_ID}.{DATASET_ID}.bookings`
    GRANT TO ('allAuthenticatedUsers')
    FILTER USING (
      (SaleBranchID IN (
         SELECT branch_id FROM `{PROJECT_ID}.{DATASET_ID}.user_entitlements`
         WHERE (wif_subject = SESSION_USER() OR user_email = SESSION_USER()) AND role = 'BRANCH_STAFF'
       )
       AND Username IN (
         SELECT username FROM `{PROJECT_ID}.{DATASET_ID}.user_entitlements`
         WHERE (wif_subject = SESSION_USER() OR user_email = SESSION_USER()) AND role = 'BRANCH_STAFF'
       ))
      OR SaleBranchID IN (
         SELECT branch_id FROM `{PROJECT_ID}.{DATASET_ID}.user_entitlements`
         WHERE (wif_subject = SESSION_USER() OR user_email = SESSION_USER()) AND role = 'BRANCH_MANAGER'
       )
      OR SellingBranch.BranchGroup IN (
         SELECT region_group FROM `{PROJECT_ID}.{DATASET_ID}.user_entitlements`
         WHERE (wif_subject = SESSION_USER() OR user_email = SESSION_USER()) AND role = 'REGIONAL_MANAGER'
       )
      OR SellingBranch.BranchBusinessDivision IN (
         SELECT division FROM `{PROJECT_ID}.{DATASET_ID}.user_entitlements`
         WHERE (wif_subject = SESSION_USER() OR user_email = SESSION_USER()) AND role = 'DIVISIONAL_HEAD'
       )
      OR EXISTS (
         SELECT 1 FROM `{PROJECT_ID}.{DATASET_ID}.user_entitlements`
         WHERE (wif_subject = SESSION_USER() OR user_email = SESSION_USER())
           AND role = 'DIVISIONAL_HEAD' AND division = 'ALL'
       )
    );
  """
  client.query(
      rls_sql,
      job_config=bigquery.QueryJobConfig(labels={"datacloud": "jetski"}),
  ).result()
  print("  Re-applied ROW ACCESS POLICY rls_4_tier_wif_policy on bookings.")


def hydrate_table(client: bigquery.Client, table_name: str) -> None:
  endpoint, token = _load_share_credentials()
  print(f"\n--- Hydrating {SHARE}.{SCHEMA}.{table_name} ---")
  query_url = (
      f"{endpoint}/shares/{SHARE}/schemas/{SCHEMA}/tables/{table_name}/query"
  )
  req = urllib.request.Request(
      query_url,
      data=b"{}",
      headers={
          "Authorization": f"Bearer {token}",
          "Content-Type": "application/json",
          "delta-sharing-capabilities": (
              "responseformat=delta;readerfeatures=columnmapping,deletionvectors,timestampntz"
          ),
      },
      method="POST",
  )
  with urllib.request.urlopen(req, timeout=120) as resp:
    lines = [
        json.loads(line)
        for line in resp.read().decode().strip().split("\n")
        if line.strip()
    ]

  meta_obj = next(l["metaData"] for l in lines if "metaData" in l)[
      "deltaMetadata"
  ]
  delta_schema = json.loads(meta_obj["schemaString"])
  desc_map = build_description_map(delta_schema.get("fields", []))

  file_urls = []
  for obj in lines:
    if "file" in obj:
      add = obj["file"].get("deltaSingleAction", {}).get("add", {})
      path = add.get("path") or obj["file"].get("url")
      if path:
        file_urls.append(path)

  print(f"Found {len(file_urls)} Parquet file(s) for {table_name}.")
  table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
  if table_name == "bookings":
    client.query(
        f"DROP ALL ROW ACCESS POLICIES ON `{table_id}`",
        job_config=bigquery.QueryJobConfig(labels={"datacloud": "jetski"}),
    ).result()

  parquet_opts = bigquery.ParquetOptions()
  parquet_opts.enable_list_inference = True

  with tempfile.TemporaryDirectory() as tmpdir:
    local_files = []
    for idx, furl in enumerate(file_urls):
      raw_path = os.path.join(tmpdir, f"raw_{idx}.parquet")
      clean_path = os.path.join(tmpdir, f"clean_{idx}.parquet")
      urllib.request.urlretrieve(furl, raw_path)
      table_pa = pq.read_table(raw_path)
      table_pa = rename_pa_table(table_pa, delta_schema)
      pq.write_table(table_pa, clean_path)
      local_files.append(clean_path)

    for idx, lpath in enumerate(local_files):
      disposition = (
          bigquery.WriteDisposition.WRITE_TRUNCATE
          if idx == 0
          else bigquery.WriteDisposition.WRITE_APPEND
      )
      job_config = bigquery.LoadJobConfig(
          source_format=bigquery.SourceFormat.PARQUET,
          write_disposition=disposition,
          parquet_options=parquet_opts,
          labels={"datacloud": "jetski"},
      )
      with open(lpath, "rb") as f:
        job = client.load_table_from_file(f, table_id, job_config=job_config)
      job.result()
      print(
          f"  Loaded shard {idx + 1}/{len(local_files)} into {table_id}"
      )

  # Restore Unity Catalog column descriptions onto the loaded BigQuery table
  if desc_map:
    table_ref = client.get_table(table_id)
    new_schema = []
    for field in table_ref.schema:
      desc = desc_map.get(field.name, field.description)
      new_schema.append(
          bigquery.SchemaField(
              name=field.name,
              field_type=field.field_type,
              mode=field.mode,
              description=desc,
              fields=field.fields,
          )
      )
    table_ref.schema = new_schema
    client.update_table(table_ref, ["schema"])
    print(f"  Restored {len(desc_map)} column descriptions on {table_id}")

  if table_name == "bookings":
    apply_bookings_rls_policy(client)


if __name__ == "__main__":
  bq_client = get_bq_client()
  target_tables = sys.argv[1:] if len(sys.argv) > 1 else TABLES
  for tbl in target_tables:
    hydrate_table(bq_client, tbl)
