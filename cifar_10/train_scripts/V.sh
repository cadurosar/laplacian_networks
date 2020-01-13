for i in `seq $1 $2`;
do
        python cifar10.py --seed $i
done    
