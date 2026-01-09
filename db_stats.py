#!/usr/bin/env python3
"""
Database Statistics and Management Tool for MindfulWatch

Commands:
  stats     - Show database statistics
  sample    - Show sample items from database
  clear     - Clear all items from database
  export    - Export database to JSON file
"""

import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, '.')

from utils import get_vector_collection
from logging_config import logger


def get_stats():
    """Get comprehensive database statistics."""
    collection = get_vector_collection()
    if not collection:
        print("Error: Could not connect to database")
        return
    
    try:
        result = collection.get()
        total = len(result["ids"]) if result["ids"] else 0
        
        # Count by type
        type_counts = {}
        for meta in result.get("metadatas", []):
            item_type = meta.get("type", "unknown")
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
        
        print("\n" + "=" * 50)
        print("DATABASE STATISTICS")
        print("=" * 50)
        print(f"\nTotal items: {total}")
        print("\nBy type:")
        for item_type, count in sorted(type_counts.items()):
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {item_type}: {count} ({pct:.1f}%)")
        
        return {"total": total, "by_type": type_counts}
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def show_sample(count=10, item_type=None):
    """Show sample items from the database."""
    collection = get_vector_collection()
    if not collection:
        print("Error: Could not connect to database")
        return
    
    try:
        result = collection.get(limit=count * 3)  # Get more to filter
        
        print("\n" + "=" * 50)
        print(f"SAMPLE ITEMS (showing up to {count})")
        print("=" * 50)
        
        shown = 0
        for i, (item_id, meta) in enumerate(zip(result["ids"], result["metadatas"])):
            if item_type and meta.get("type") != item_type:
                continue
                
            if shown >= count:
                break
            
            print(f"\n[{shown + 1}] {meta.get('title', 'Unknown')}")
            print(f"    Type: {meta.get('type', 'unknown')}")
            print(f"    ID: {item_id}")
            overview = meta.get("overview", "")[:100]
            if overview:
                print(f"    Overview: {overview}...")
            
            shown += 1
        
    except Exception as e:
        print(f"Error: {e}")


def clear_database():
    """Clear all items from the database."""
    collection = get_vector_collection()
    if not collection:
        print("Error: Could not connect to database")
        return
    
    try:
        result = collection.get()
        ids = result.get("ids", [])
        
        if not ids:
            print("Database is already empty.")
            return
        
        print(f"\nThis will delete {len(ids)} items.")
        confirm = input("Are you sure? (yes/no): ")
        
        if confirm.lower() == "yes":
            collection.delete(ids=ids)
            print(f"Deleted {len(ids)} items.")
        else:
            print("Cancelled.")
            
    except Exception as e:
        print(f"Error: {e}")


def export_database(filename=None):
    """Export database contents to JSON."""
    collection = get_vector_collection()
    if not collection:
        print("Error: Could not connect to database")
        return
    
    if filename is None:
        filename = f"db_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        result = collection.get()
        
        items = []
        for item_id, meta, doc in zip(
            result.get("ids", []),
            result.get("metadatas", []),
            result.get("documents", [])
        ):
            items.append({
                "id": item_id,
                "metadata": meta,
                "document": doc
            })
        
        with open(filename, "w") as f:
            json.dump(items, f, indent=2)
        
        print(f"Exported {len(items)} items to {filename}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Database management tool")
    parser.add_argument("command", choices=["stats", "sample", "clear", "export"],
                       help="Command to run")
    parser.add_argument("--count", type=int, default=10,
                       help="Number of items for sample command")
    parser.add_argument("--type", choices=["movie", "video"],
                       help="Filter by type for sample command")
    parser.add_argument("--output", help="Output filename for export")
    
    args = parser.parse_args()
    
    if args.command == "stats":
        get_stats()
    elif args.command == "sample":
        show_sample(count=args.count, item_type=args.type)
    elif args.command == "clear":
        clear_database()
    elif args.command == "export":
        export_database(args.output)


if __name__ == "__main__":
    main()
