
## Simple Pattern
### pattern: close
``` 
python3 src/aaa_main.py --dataset close --device_id 0
```
### pattern: diagonal
``` 
python3 src/aaa_main.py --dataset diagonal --device_id 0
```
### pattern: two_pairs
``` 
python3 src/aaa_main.py --dataset two_pairs --device_id 0
```
### pattern: three_same
``` 
python3 src/aaa_main.py --dataset three_same_3 --device_id 0
```

## Complex Patterns

### pattern: check_mark

``` 
# pattern: check mark with 4 objects
python3 src/aaa_main.py --dataset check_mark_4 --dataset-type single_pattern --device_id 0

# pattern: check mark with 6 objects
python3 src/aaa_main.py --dataset check_mark_6 --dataset-type single_pattern --device_id 0
```
### pattern: parallel
``` 
python3 src/aaa_main.py --dataset parallel --dataset-type custom_scenes --sn_th 0.9 --with_pi  --device_id 0
```

### pattern: perpendicular

``` 
python3 src/aaa_main.py --dataset perpendicular --sn_th 0.9 --with_pi  --device_id 0
```

### pattern: square
``` 
python3 src/aaa_main.py --dataset square --sn_th 0.9 --with_pi  --device_id 0
```

###### Alphabet Patterns

To run experiment about letter `X`, replace the arg of `--dataset_A` to `letter_X`.
``` 
python3 src/aaa_main.py --dataset letter_A --dataset-type alphabet --device_id 0
```



