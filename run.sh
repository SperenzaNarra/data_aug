set -e
n=1000
# python label_yolo.py -n 4 -T 100 --objs 8 --save yolo
python label_yolo.py -n 4 -T $n --objs 1 --save yolo
python label_yolo.py -n 4 -T $((n*2)) --objs 2 --save yolo
python label_yolo.py -n 4 -T $((n*3)) --objs 3 --save yolo
python label_yolo.py -n 4 -T $((n*4)) --objs 4 --save yolo
python label_yolo.py -n 4 -T $((n*5)) --objs 5 --save yolo
python label_yolo.py -n 4 -T $((n*6)) --objs 6 --save yolo
python label_yolo.py -n 4 -T $((n*7)) --objs 7 --save yolo
python label_yolo.py -n 4 -T $((n*8)) --objs 8 --save yolo
python label_yolo.py -n 4 -T $((n*9)) --objs 1 -a --save yolo
python label_yolo.py -n 4 -T $((n*10)) --objs 2 -a --save yolo
python label_yolo.py -n 4 -T $((n*11)) --objs 3 -a --save yolo
python label_yolo.py -n 4 -T $((n*12)) --objs 4 -a --save yolo
python label_yolo.py -n 4 -T $((n*13)) --objs 5 -a --save yolo
python label_yolo.py -n 4 -T $((n*14)) --objs 6 -a --save yolo
python label_yolo.py -n 4 -T $((n*15)) --objs 7 -a --save yolo
python label_yolo.py -n 4 -T $((n*16)) --objs 8 -a --save yolo