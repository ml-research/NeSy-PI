# NeSy-PI

The implementation of NeSy-PI

## Docker
If run the experiment on a docker, you can use the following command.
###### Build docker

``` 
docker build -t nesy_pi_docker .
```

###### Run docker

``` 
docker run --gpus all -it -v path/to/storage:/nesypi/storage --rm nesy_pi_docker
```

## Experiments

#### Dataset
The dataset is saved in the extern folder `storage`. 


###### Simple Pattern

``` 
# pattern: close
python3 src/aaa_main.py --dataset close --dataset-type custom_scenes --device_id 7
```




###### Complex Patterns

``` 
# pattern: check mark with 4 objects
python3 src/aaa_main.py --dataset check_mark_4 --dataset-type single_pattern --device_id 6

# pattern: check mark with 6 objects
python3 src/aaa_main.py --dataset check_mark_6 --dataset-type single_pattern --device_id 6


python3 src/aaa_main.py --dataset square --dataset-type custom_scenes --device_id 9 --sn_th 0.9 --with_pi
python3 src/aaa_main.py --dataset parallel --dataset-type custom_scenes --device_id 6 --sn_th 0.9 --with_pi
python3 src/aaa_main.py --dataset parallel --dataset-type custom_scenes --device_id 7 --sn_th 0.9
python3 src/aaa_main.py --dataset check_mark --dataset-type custom_scenes --device_id 4 --sn_th 0.9 --with_pi
python3 src/aaa_main.py --dataset check_mark --dataset-type custom_scenes --device_id 5 --sn_th 0.9
```



###### Alphabet Pattern

``` 
python3 src/aaa_main.py --dataset letter_G --dataset-type alphabet --device_id 7
python3 src/aaa_main.py --dataset letter_B --dataset-type alphabet --device_id 7
python3 src/aaa_main.py --dataset letter_Q --dataset-type alphabet --device_id 7
python3 src/aaa_main.py --dataset letter_W --dataset-type alphabet --device_id 3
python3 src/aaa_main.py --dataset-type alphabet --device_id 12 --sn_th 0.92 --dataset letter_E --with_pi
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.92 --dataset letter_E --device_id 15
python3 src/aaa_main.py --dataset-type alphabet --device_id 14 --sn_th 0.9 --dataset letter_D --with_pi
python3 src/aaa_main.py --dataset-type alphabet --device_id 12 --sn_th 0.9 --dataset perpendicular --with_pi

python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_G --device_id 1 --with_pi
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_H --device_id 2 --with_pi
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_H --device_id 1
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_I --device_id 1 --with_pi
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_I --device_id 1
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_J --device_id 1 --with_pi
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_P --device_id 1 --with_pi
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_P --device_id 2 --with_pi
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_Q --device_id 8 --with_pi
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_Q --device_id 9
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_S --device_id 10 --with_pi
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_S --device_id 12
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_T --device_id 1 --with_pi
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_T --device_id 2
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_U --device_id 1 --with_pi
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_U --device_id 2
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_V --device_id 4 --with_pi
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_V --device_id 5
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_W --device_id 6 --with_pi
python3 src/aaa_main.py --dataset-type alphabet --sn_th 0.9 --dataset letter_W --device_id 7
```



