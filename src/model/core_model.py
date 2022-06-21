from ..model.core.fasterrcnnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor


def get_fu_model():
    '''
        返回实体福字检测模型 2 分类目标检测模型
    '''
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_fu2_model():
    '''
        返回实体福字和手写福字目标检测模型 3分类目标检测模型
    '''
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model