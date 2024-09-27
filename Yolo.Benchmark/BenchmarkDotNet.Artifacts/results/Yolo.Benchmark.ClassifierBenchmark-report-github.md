```

BenchmarkDotNet v0.14.0, KDE neon 6.1
AMD Ryzen 5 5600X, 1 CPU, 12 logical and 6 physical cores
.NET SDK 8.0.108
  [Host]     : .NET 8.0.8 (8.0.824.36612), X64 RyuJIT AVX2
  DefaultJob : .NET 8.0.8 (8.0.824.36612), X64 RyuJIT AVX2


```
| Method  | Mean     | Error     | StdDev    | Allocated |
|-------- |---------:|----------:|----------:|----------:|
| Predict | 2.520 ms | 0.0494 ms | 0.0607 ms |    8.6 KB |
