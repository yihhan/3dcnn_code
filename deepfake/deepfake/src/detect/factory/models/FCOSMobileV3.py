# model settings
model = dict(
    type='FCOS',
    pretrained=None,
    backbone=dict(
        type='MobileNetV3',
        pretrained=True),
    neck=dict(
        type='NASFPN',
        in_channels=[72, 120, 672, 960],
        out_channels=256,
        num_outs=5,
        stack_times=3,
        start_level=1,
        add_extra_convs=True,
        norm_cfg=dict(type='BN', requires_grad=True)),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=81,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05, # keep the threshold low, we will tune during validation
    nms=dict(type='nms', iou_thr=0.5), 
    max_per_img=100)
