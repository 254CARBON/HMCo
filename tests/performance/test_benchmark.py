"""
Basic performance benchmark tests for the platform.
"""
import pytest
import time


def fibonacci(n):
    """Calculate Fibonacci number (for benchmark testing)."""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def test_fibonacci_benchmark(benchmark):
    """Benchmark Fibonacci calculation."""
    result = benchmark(fibonacci, 10)
    assert result == 55


def test_list_operations_benchmark(benchmark):
    """Benchmark list operations."""
    def list_ops():
        data = []
        for i in range(1000):
            data.append(i)
        return sum(data)
    
    result = benchmark(list_ops)
    assert result == 499500


def test_dict_operations_benchmark(benchmark):
    """Benchmark dictionary operations."""
    def dict_ops():
        data = {}
        for i in range(1000):
            data[f"key_{i}"] = i
        return len(data)
    
    result = benchmark(dict_ops)
    assert result == 1000


def test_string_operations_benchmark(benchmark):
    """Benchmark string operations."""
    def string_ops():
        text = ""
        for i in range(100):
            text += f"item_{i} "
        return len(text.split())
    
    result = benchmark(string_ops)
    assert result == 100
