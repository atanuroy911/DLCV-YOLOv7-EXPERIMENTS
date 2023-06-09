# DLCV-YOLOv7-EXPERIMENTS

```
https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-seg.pt

python segment/train.py --data coco.yaml --batch 16 --weights '' --cfg yolov7-seg.yaml --epochs 300 --name yolov7-seg --img 640 --hyp hyp.scratch-high.yaml


python3 train.py --data ../datasets/sewage-defects/data.yaml --cfg ../configs/yolov7-tiny-ecanet-c3n2.yaml --weights weights/yolov7_training.pt --hyp data/hyp.scratch.tiny.yaml --name sewage-c3-eca --device 1 --epoch 300

python3 test.py --data ../datasets/precision-ag/data.yaml --weights runs/train/precision-ag-yolov7-tiny-n/weights/best.pt --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --name precision-ag-tiny-n

python3 test.py --data ../datasets/inpipe/data.yaml --weights ../results/inpipe-yolov7-tiny-n/weights/best.pt --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --name inpipe-tiny-n

python3 test.py --data ../datasets/inpipe/data.yaml --weights ../results/inpipe-tiny-eca-c3/weights/best.pt --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --name inpipe-tiny-n

python3 train.py --data ../datasets/inpipe/data.yaml --weights weights/yolov7_training.pt --cfg ../configs/yolov7-tiny-ecanet-cbam-c3.yaml --name inpipe-tiny-eca-cbam-c3-new --device 0 --epochs 300

python3 train.py --data ../datasets/inpipe/data.yaml --cfg ../configs/yolov7.yaml --weights weights/yolov7_training.pt --hyp data/hyp.scratch.tiny.yaml --name inpipe-yolov7 --device 1 --epoch 300

python3 train.py --data ../datasets/inpipe/data.yaml --cfg ../configs/yolov7-tiny.yaml --weights weights/yolov7_training.pt --hyp data/hyp.scratch.tiny.yaml --name inpipe-yolov7 --device 1 --epoch 300


python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --workers 8 --device 0,1 --sync-bn --data ../datasets/inpipe/data.yaml --weights weights/yolov7_training.pt --cfg ../configs/yolov7-tiny-ecanet-c3n3.yaml --name inpipe-tiny-eca-c3-nnn --epochs 300

python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --workers 32 --device 0,1 --sync-bn --data ../datasets/coco128/data.yaml --weights weights/yolov7_training.pt --cfg ../configs/yolov7-tiny-ecanet-c3n3.yaml --name test --epochs 300
```

