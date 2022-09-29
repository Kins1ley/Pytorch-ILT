# CU-ILT

## Prerequest

### For MacOS

1. install automake `brew install automake`
2. install libtool `brew install libtool`
3. install protobuf `brew install protobuf`

## command line options

```shell
usage: ./cuilt -input ../Benchmarks/M1_test1.glp -output M1_test_mask.glp -gpu [options] ...
options:
  -input,    input file path
  -output,   output file path
  -gpu,      use cuda when compute
```