# ER: Equivariance Regularizer for Knowledge Graph Completion

This is the code of paper "ER: Equivariance Regularizer for Knowledge Graph Completion". 


## Reproduce the Results

### 1. Preprocess the Datasets
To preprocess the datasets, run the following commands.

```shell script
cd code
python3 process_datasets.py
```

Now, the processed datasets are in the `data` directory.

### 2. Reproduce the Results 
To reproduce the results of CP, ComplEx and RESCAL with
the ER regularizer on WN18RR, FB15k237 and YAGO3-10,
please run the following commands.

```shell script
#################################### WN18RR ####################################
# CP
CUDA_VISIBLE_DEVICES=0 python3 learn.py --dataset WN18RR --model CP --rank 2000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 100 --regularizer ER --reg 1e-1 --max_epochs 200 \
--valid 5 -train -id 0 -save -weight


# ComplEx
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset WN18RR --model ComplEx --rank 2000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 100 --regularizer ER --reg 1e-1 --max_epochs 50 \
--valid 5 -train -id 0 -save -weight

# RESCAL
CUDA_VISIBLE_DEVICES=3 python3 learn.py --dataset WN18RR --model RESCAL --rank 256 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 1024 --regularizer ER_RESCAL --reg 1e-1 --max_epochs 200 \
--valid 5 -train -id 0 -save -weight

#################################### FB237 ####################################
# CP
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset FB237 --model CP --rank 2000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 100 --regularizer ER --reg 5e-2 --max_epochs 200 \
--valid 5 -train -id 0 -save

# ComplEx
CUDA_VISIBLE_DEVICES=7 python3 learn.py --dataset FB237 --model ComplEx --rank 2000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 2000 --regularizer ER --reg 5e-2 --max_epochs 200 \
--valid 5 -train -id 0 -save

# RESCAL
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset FB237 --model RESCAL --rank 512 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 512 --regularizer ER_RESCAL --reg 5e-2 --max_epochs 200 \
--valid 5 -train -id 0 -save


#################################### YAGO3-10 ####################################
# CP
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset YAGO3-10 --model CP --rank 1000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 1000 --regularizer ER --reg 5e-3 --max_epochs 200 \
--valid 5 -train -id 0 -save -weight

# ComplEx
CUDA_VISIBLE_DEVICES=2 python3 learn.py --dataset YAGO3-10 --model ComplEx --rank 1000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 1000 --regularizer ER --reg 5e-3 --max_epochs 200 \
--valid 5 -train -id 0 -save

# RESCAL
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset YAGO3-10 --model RESCAL --rank 512 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 1024 --regularizer ER_RESCAL --reg 5e-3 --max_epochs 200 \
--valid 5 -train -id 0 -save -weight
```

## Acknowledgement
We refer to the code of [kbc](https://github.com/facebookresearch/kbc). Thanks for their contributions.

