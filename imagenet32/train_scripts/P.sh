for i in `seq $1 $2`;
do
        python cifar100.py --beta 0.01 --seed $i
done    
