path = ''

python get_list.py ./formal/$path

for i in `less pdb_list.dat`;
    do awk -F " " '{if($3=="C"||$3=="N"||$3=="CA")print}' ./formal/$path/$i > ./processed/$path/$i;
done
sed -i "s/HIE/HIS/g" `grep -rl HIE ./processed/$path/`
