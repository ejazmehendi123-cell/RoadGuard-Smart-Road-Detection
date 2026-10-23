"""
RoadGuard — Supabase Connection Verifier
Replaces the old PostgreSQL init_db.py.
Run this to verify your Supabase credentials and check table counts.
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://eitvwhbivmsqyrnkvqzv.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY", "")

def verify_connection():
    if not SUPABASE_KEY:
        print("ERROR: No Supabase key found.")
        print("Set SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY in server/.env")
        return False

    print(f"Connecting to Supabase: {SUPABASE_URL} ...")
    try:
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)

        tables = ["hazards", "hazard_observations", "trips", "devices", "profiles", "user_settings"]
        print("\nTable row counts:")
        for table in tables:
            try:
                resp = sb.table(table).select("*", count="exact").limit(1).execute()
                count = resp.count or 0
                print(f"  {table:<30} {count} rows")
            except Exception as e:
                print(f"  {table:<30} ERROR: {e}")

        print("\nSupabase connection verified successfully!")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


if __name__ == "__main__":
    verify_connection()
