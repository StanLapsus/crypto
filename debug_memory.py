#!/usr/bin/env python3
"""
Tool for diagnosing memory issues in the crypto prediction system.
"""
import os
import sys
import tracemalloc
import gc
import argparse
from datetime import datetime

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_memory_stats(snapshot=None, top_n=20, group_by='lineno', filters=None):
    """Print memory statistics and top allocations"""
    if snapshot is None:
        snapshot = tracemalloc.take_snapshot()
        
    print(f"Memory snapshot taken at {datetime.now().strftime('%H:%M:%S')}")
    
    # Get current and peak memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**6:.2f}MB; Peak: {peak / 10**6:.2f}MB")
    
    # Group statistics
    if group_by == 'filename':
        stats = snapshot.statistics('filename')
    elif group_by == 'traceback':
        stats = snapshot.statistics('traceback')
    else:
        stats = snapshot.statistics('lineno')
    
    # Apply filters if provided
    if filters:
        filtered_stats = []
        for stat in stats:
            trace_str = str(stat.traceback)
            if any(f in trace_str for f in filters):
                filtered_stats.append(stat)
        if filtered_stats:
            stats = filtered_stats
        else:
            print(f"No stats match the filters: {filters}")
    
    # Print the top allocations
    print(f"\nTop {top_n} memory allocations by {group_by}:")
    for i, stat in enumerate(stats[:top_n], 1):
        print(f"#{i}: {stat.size / 1024:.1f} KB")
        if group_by == 'traceback':
            for line in stat.traceback.format():
                print(f"    {line}")
        else:
            print(f"    {stat}")
    
    return snapshot

def compare_snapshots(snapshot1, snapshot2):
    """Compare two memory snapshots to find leaks"""
    print("\n=== Memory Comparison ===")
    
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("\nTop 10 memory increases:")
    for i, stat in enumerate(stats[:10], 1):
        if stat.size_diff > 0:  # Only show increases
            print(f"#{i}: {stat.size_diff / 1024:.1f} KB increase at:")
            print(f"    {stat}")
    
    return stats

def run_memory_profile(test_func, iterations=3, interval=1.0):
    """Profile memory usage during repeated function calls"""
    import time
    
    # Start tracing if not already started
    if not tracemalloc.is_tracing():
        tracemalloc.start(25)
    
    snapshots = []
    try:
        # Take initial snapshot
        snapshots.append(tracemalloc.take_snapshot())
        
        # Run the function repeatedly and take snapshots
        for i in range(iterations):
            clear_screen()
            print(f"Running iteration {i+1}/{iterations}...")
            
            test_func()
            
            # Force garbage collection
            gc.collect()
            
            # Sleep to allow async operations to complete
            time.sleep(interval)
            
            # Take a new snapshot
            snapshots.append(tracemalloc.take_snapshot())
            
            # Compare with previous
            if i > 0:
                compare_snapshots(snapshots[i], snapshots[i+1])
        
        # Final comparison between first and last snapshot
        clear_screen()
        print("\n=== Overall Memory Change ===")
        compare_snapshots(snapshots[0], snapshots[-1])
        
        # Print detailed stats from final snapshot
        print("\n=== Final Memory State ===")
        print_memory_stats(snapshots[-1], top_n=15)
        
    except KeyboardInterrupt:
        print("\nMemory profiling interrupted by user")
    finally:
        return snapshots

def main():
    parser = argparse.ArgumentParser(description="Memory debugging tool for crypto prediction system")
    parser.add_argument('--mode', choices=['snapshot', 'profile', 'compare'], default='snapshot',
                        help='Debug mode: take snapshot, profile execution, or compare')
    parser.add_argument('--filter', type=str, help='Filter results to specified modules')
    parser.add_argument('--top', type=int, default=20, help='Number of top allocations to show')
    parser.add_argument('--group', choices=['lineno', 'filename', 'traceback'], default='lineno',
                        help='How to group memory allocation statistics')
    
    args = parser.parse_args()
    
    if not tracemalloc.is_tracing():
        tracemalloc.start(25)
    
    if args.mode == 'snapshot':
        filters = [args.filter] if args.filter else None
        print_memory_stats(top_n=args.top, group_by=args.group, filters=filters)
    
    elif args.mode == 'profile':
        # Import here to avoid startup overhead
        from main import CryptoPredictionSystem
        
        def test_prediction():
            system = CryptoPredictionSystem()
            system.run_prediction_cycle()
        
        print("Profiling prediction cycles...")
        run_memory_profile(test_prediction, iterations=3, interval=2.0)
    
    elif args.mode == 'compare':
        print("Taking first snapshot. Press Enter when ready to take second snapshot for comparison...")
        snapshot1 = tracemalloc.take_snapshot()
        input()
        snapshot2 = tracemalloc.take_snapshot()
        compare_snapshots(snapshot1, snapshot2)

if __name__ == "__main__":
    main()
