for i in `seq $1 $2`;
do
        python cifar100.py --beta 0.01 --gamma 0.01 -m 2 --seed $i
done    
