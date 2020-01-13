for i in `seq $1 $2`;
do
        python cifar100.py --seed $i
done    
